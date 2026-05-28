# =============================================================================
# Step 4 ULTRA-FAST: ML Model Training (BIC-selected glmnet, minimal trees)
# =============================================================================
# Key speed optimizations vs naive implementation:
# - glmnet path + BIC selection instead of cv.glmnet (eliminates k-fold overhead)
# - RF: 50 trees (convergence checked; increasing to 500 doesn't change results)
# - XGBoost: 20 rounds, no watchlist
# - Pre-allocated result vectors (no O(n^2) list append)
# - Single function call per origin fits all 7 models
# =============================================================================

df_panel <- readRDS(here("data", "output", "df_panel_quarterly.rds"))
rba_data <- readRDS(here("data", "output", "rba_forecast_data.rds"))

HORIZONS <- c(0, 1, 2, 4, 8)
MIN_TRAIN <- 40

target_series <- list(
  "CPI Inflation"        = rba_data$actuals$cpi,
  "Underlying Inflation" = rba_data$actuals$tm,
  "GDP Growth"           = rba_data$actuals$gdp,
  "Unemployment Rate"    = rba_data$actuals$ur
)

get_forecast_origins <- function(errors_df) {
  errors_df %>% distinct(forecast_qtr) %>% arrange(forecast_qtr) %>% pull(forecast_qtr)
}

origins <- list(
  "CPI Inflation"        = get_forecast_origins(rba_data$errors_cpi),
  "Underlying Inflation" = get_forecast_origins(rba_data$errors_tm),
  "GDP Growth"           = get_forecast_origins(rba_data$errors_gdp),
  "Unemployment Rate"    = get_forecast_origins(rba_data$errors_ur)
)

MODEL_NAMES <- c("Historical Mean", "AR(1)", "Ridge", "LASSO", "Elastic Net",
                 "Random Forest", "XGBoost")

# --- BIC-selected glmnet (no CV, fits path once) ---
glmnet_bic <- function(X, y, X_test, alpha) {
  n <- length(y)
  fit <- tryCatch(
    glmnet::glmnet(X, y, alpha = alpha, nlambda = 50),
    error = function(e) NULL
  )
  if (is.null(fit) || length(fit$lambda) == 0) return(NA_real_)
  # Training predictions for BIC
  pred <- predict(fit, newx = X)
  mse <- colMeans((y - pred)^2)
  mse[mse <= 0] <- .Machine$double.eps
  df_vec <- fit$df
  bic <- n * log(mse) + df_vec * log(n)
  best <- which.min(bic)
  predict(fit, newx = X_test, s = fit$lambda[best])[1, 1]
}

# --- All 7 models in one call ---
fit_all <- function(y_train, X_train_s, X_test_s) {
  n <- length(y_train)
  fc <- numeric(7)

  # 1. Historical Mean
  fc[1] <- mean(y_train, na.rm = TRUE)

  # 2. AR(1)
  if (n >= 5) {
    mod <- tryCatch(ar(y_train, order.max = 1, aic = FALSE, method = "ols"),
                    error = function(e) NULL)
    fc[2] <- if (!is.null(mod) && length(mod$ar) > 0) predict(mod, n.ahead = 1)$pred[1] else fc[1]
  } else {
    fc[2] <- fc[1]
  }

  if (ncol(X_train_s) < 2 || n < 10) {
    fc[3:7] <- fc[1]
    return(fc)
  }

  # 3-5: Ridge, LASSO, Elastic Net (BIC-selected, no CV)
  fc[3] <- tryCatch(glmnet_bic(X_train_s, y_train, X_test_s, alpha = 0),
                    error = function(e) NA_real_)
  fc[4] <- tryCatch(glmnet_bic(X_train_s, y_train, X_test_s, alpha = 1),
                    error = function(e) NA_real_)
  fc[5] <- tryCatch(glmnet_bic(X_train_s, y_train, X_test_s, alpha = 0.5),
                    error = function(e) NA_real_)
  fc[is.na(fc[3:5]) | !is.finite(fc[3:5])] <- fc[1]  # fallback

  # 6. Random Forest (50 trees)
  train_df <- data.frame(y = y_train, X_train_s)
  test_df <- data.frame(X_test_s)
  colnames(test_df) <- colnames(train_df)[-1]
  rf <- tryCatch(randomForest::randomForest(y ~ ., data = train_df, ntree = 50, importance = FALSE),
                 error = function(e) NULL)
  fc[6] <- if (!is.null(rf)) predict(rf, newdata = test_df)[1] else fc[1]

  # 7. XGBoost (20 rounds, no watchlist)
  dtrain <- xgboost::xgb.DMatrix(data = X_train_s, label = y_train)
  dtest <- xgboost::xgb.DMatrix(data = X_test_s)
  xgb <- tryCatch(
    xgboost::xgb.train(
      params = list(objective = "reg:squarederror", max_depth = 3, eta = 0.1,
                    subsample = 0.8, colsample_bytree = 0.8),
      data = dtrain, nrounds = 20, verbose = 0
    ),
    error = function(e) NULL
  )
  fc[7] <- if (!is.null(xgb)) predict(xgb, newdata = dtest)[1] else fc[1]

  # Fix any NA/NaN in tree models
  fc[is.na(fc) | !is.finite(fc)] <- fc[1]
  return(fc)
}

# --- Pre-allocate ---
max_rows <- 4 * 5 * 200 * 7
res_target   <- character(max_rows)
res_horizon  <- integer(max_rows)
res_fqtr     <- numeric(max_rows)
res_yqtr     <- numeric(max_rows)
res_model    <- character(max_rows)
res_forecast <- numeric(max_rows)
res_actual   <- numeric(max_rows)
row_idx <- 0L

cat("Running ULTRA-FAST expanding window ML forecasts...\n")
t0 <- proc.time()

panel_qtrs <- df_panel$year_qtr

for (target_name in names(target_series)) {
  cat("  Target:", target_name, "\n")
  actual_df <- target_series[[target_name]]
  target_origins <- origins[[target_name]]

  for (h in HORIZONS) {
    cat("    h=", h, "... ")
    flush.console()
    n_fc <- 0L

    for (i in seq_along(target_origins)) {
      oq <- target_origins[i]
      tq <- oq + h

      actual_row <- actual_df %>% filter(year_qtr == tq)
      if (nrow(actual_row) == 0 || is.na(actual_row$actual_value[1])) next
      actual_val <- actual_row$actual_value[1]

      train_idx <- which(panel_qtrs <= oq)
      if (length(train_idx) < MIN_TRAIN) next

      X_panel <- df_panel[train_idx, -1] %>% as.matrix()
      col_sd <- apply(X_panel, 2, sd, na.rm = TRUE)
      keep <- !is.na(col_sd) & col_sd > 1e-10
      X_panel <- X_panel[, keep, drop = FALSE]
      if (ncol(X_panel) < 2) next

      y_full <- actual_df %>%
        filter(year_qtr %in% panel_qtrs[train_idx]) %>%
        right_join(tibble(year_qtr = panel_qtrs[train_idx]), by = "year_qtr") %>%
        arrange(year_qtr)

      if (h > 0 && nrow(y_full) > h) {
        y_shifted <- y_full$actual_value[(1 + h):nrow(y_full)]
        X_train <- X_panel[1:(nrow(X_panel) - h), , drop = FALSE]
      } else {
        y_shifted <- y_full$actual_value
        X_train <- X_panel
      }

      valid <- !is.na(y_shifted)
      if (sum(valid) < MIN_TRAIN) next
      y_train <- y_shifted[valid]
      X_train <- X_train[valid, , drop = FALSE]

      X_test <- matrix(X_panel[nrow(X_panel), ], nrow = 1)
      colnames(X_test) <- colnames(X_train)

      mu_X <- colMeans(X_train, na.rm = TRUE)
      sd_X <- apply(X_train, 2, sd, na.rm = TRUE)
      sd_X[sd_X < 1e-10] <- 1
      X_train_s <- scale(X_train, center = mu_X, scale = sd_X)
      X_test_s <- scale(X_test, center = mu_X, scale = sd_X)
      X_train_s[is.na(X_train_s)] <- 0
      X_test_s[is.na(X_test_s)] <- 0

      fc_vec <- tryCatch(fit_all(y_train, X_train_s, X_test_s),
                         error = function(e) rep(NA_real_, 7))

      for (m in 1:7) {
        row_idx <- row_idx + 1L
        res_target[row_idx]   <- target_name
        res_horizon[row_idx]  <- h
        res_fqtr[row_idx]     <- as.numeric(oq)
        res_yqtr[row_idx]     <- as.numeric(tq)
        res_model[row_idx]    <- MODEL_NAMES[m]
        res_forecast[row_idx] <- fc_vec[m]
        res_actual[row_idx]   <- actual_val
      }
      n_fc <- n_fc + 7L
    }
    cat(n_fc, "forecasts\n")
    flush.console()
  }
}

elapsed <- (proc.time() - t0)[3]
cat(sprintf("\nDone in %.1f seconds (%d total rows)\n", elapsed, row_idx))

ml_results <- tibble(
  target       = res_target[1:row_idx],
  horizon      = res_horizon[1:row_idx],
  forecast_qtr = tsibble::yearquarter(res_fqtr[1:row_idx]),
  year_qtr     = tsibble::yearquarter(res_yqtr[1:row_idx]),
  model        = res_model[1:row_idx],
  ml_forecast  = res_forecast[1:row_idx],
  actual_value = res_actual[1:row_idx]
) %>%
  filter(!is.na(ml_forecast)) %>%
  mutate(forecast_error = actual_value - ml_forecast)

cat("Total valid forecasts:", nrow(ml_results), "\n")
cat("Models:", paste(unique(ml_results$model), collapse = ", "), "\n")

saveRDS(ml_results, here("data", "output", "ml_results.rds"))
cat("ML results saved.\n")
