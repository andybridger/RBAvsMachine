# =============================================================================
# Step 4: ML Model Training with Expanding Window Cross-Validation
# =============================================================================
# For each RBA forecast vintage, train ML models on data available at that time.
# Generate forecasts at horizons 0,1,2,4,8 quarters.
# Models: Historical Mean, AR(1), Ridge, LASSO, Elastic Net, Random Forest, XGBoost.
# =============================================================================

df_panel <- readRDS(here("data", "output", "df_panel_quarterly.rds"))
rba_data <- readRDS(here("data", "output", "rba_forecast_data.rds"))

model_fns <- get_model_functions()
HORIZONS <- c(0, 1, 2, 4, 8)
MIN_TRAIN <- 40  # minimum training observations

# --- Target variable construction ---
# We need the actual (untransformed) target values at quarterly frequency
target_series <- list(
  "CPI Inflation"        = rba_data$actuals$cpi,
  "Underlying Inflation" = rba_data$actuals$tm,
  "GDP Growth"           = rba_data$actuals$gdp,
  "Unemployment Rate"    = rba_data$actuals$ur
)

# --- Get unique RBA forecast origins ---
get_forecast_origins <- function(errors_df) {
  errors_df %>%
    distinct(forecast_qtr) %>%
    arrange(forecast_qtr) %>%
    pull(forecast_qtr)
}

origins <- list(
  "CPI Inflation"        = get_forecast_origins(rba_data$errors_cpi),
  "Underlying Inflation" = get_forecast_origins(rba_data$errors_tm),
  "GDP Growth"           = get_forecast_origins(rba_data$errors_gdp),
  "Unemployment Rate"    = get_forecast_origins(rba_data$errors_ur)
)

# --- Expanding window forecasting ---
cat("Running expanding window ML forecasts...\n")

all_ml_forecasts <- list()

for (target_name in names(target_series)) {
  cat("  Target:", target_name, "\n")
  actual_df <- target_series[[target_name]]
  target_origins <- origins[[target_name]]

  panel_qtrs <- df_panel$year_qtr

  for (h in HORIZONS) {
    cat("    Horizon:", h, "... ")

    # Iterate by index to preserve yearquarter class
    for (i in seq_along(target_origins)) {
      origin_qtr <- target_origins[i]
      target_qtr <- origin_qtr + h

      # Get actual value at target quarter
      actual_row <- actual_df %>% filter(year_qtr == target_qtr)
      if (nrow(actual_row) == 0 || is.na(actual_row$actual_value[1])) next
      actual_val <- actual_row$actual_value[1]

      # Training data: all panel obs up to origin
      train_idx <- which(panel_qtrs <= origin_qtr)
      if (length(train_idx) < MIN_TRAIN) next

      X_panel <- df_panel[train_idx, -1] %>% as.matrix()
      # Remove constant columns
      col_sd <- apply(X_panel, 2, sd, na.rm = TRUE)
      keep <- !is.na(col_sd) & col_sd > 1e-10
      X_panel <- X_panel[, keep, drop = FALSE]

      if (ncol(X_panel) < 2) next

      # Target: get historical target values aligned with panel quarters
      y_full <- actual_df %>%
        filter(year_qtr %in% panel_qtrs[train_idx]) %>%
        right_join(tibble(year_qtr = panel_qtrs[train_idx]), by = "year_qtr") %>%
        arrange(year_qtr)

      # For direct forecasting at horizon h:
      # y[t+h] ~ X[t], so shift y forward by h
      if (h > 0 && nrow(y_full) > h) {
        y_shifted <- y_full$actual_value[(1 + h):nrow(y_full)]
        X_train <- X_panel[1:(nrow(X_panel) - h), , drop = FALSE]
      } else {
        y_shifted <- y_full$actual_value
        X_train <- X_panel
      }

      # Remove NA target observations
      valid <- !is.na(y_shifted)
      if (sum(valid) < MIN_TRAIN) next
      y_train <- y_shifted[valid]
      X_train <- X_train[valid, , drop = FALSE]

      # Test observation: last row of the panel at origin
      X_test <- matrix(X_panel[nrow(X_panel), ], nrow = 1)
      colnames(X_test) <- colnames(X_train)

      # Standardize predictors
      mu_X <- colMeans(X_train, na.rm = TRUE)
      sd_X <- apply(X_train, 2, sd, na.rm = TRUE)
      sd_X[sd_X < 1e-10] <- 1
      X_train_s <- scale(X_train, center = mu_X, scale = sd_X)
      X_test_s <- scale(X_test, center = mu_X, scale = sd_X)

      # Replace any remaining NAs with 0
      X_train_s[is.na(X_train_s)] <- 0
      X_test_s[is.na(X_test_s)] <- 0

      for (model_name in names(model_fns)) {
        fit_fn <- model_fns[[model_name]]
        result <- tryCatch(
          fit_fn(y_train, X_train_s, X_test_s),
          error = function(e) list(forecast = NA)
        )

        all_ml_forecasts[[length(all_ml_forecasts) + 1]] <- tibble(
          target       = target_name,
          horizon      = h,
          forecast_qtr = origin_qtr,
          year_qtr     = target_qtr,
          model        = model_name,
          ml_forecast  = result$forecast,
          actual_value = actual_val
        )
      }
    }
    cat(length(all_ml_forecasts), "total forecasts\n")
  }
}

ml_results <- bind_rows(all_ml_forecasts) %>%
  filter(!is.na(ml_forecast)) %>%
  mutate(forecast_error = actual_value - ml_forecast)

cat("\nML forecasting complete.\n")
cat("Total forecasts:", nrow(ml_results), "\n")
cat("Models:", paste(unique(ml_results$model), collapse = ", "), "\n")

saveRDS(ml_results, here("data", "output", "ml_results.rds"))
cat("ML results saved.\n")
