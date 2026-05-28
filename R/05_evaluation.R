# =============================================================================
# Step 5: Forecast Evaluation and Robustness Tests
# =============================================================================
# Compare RBA vs ML forecast accuracy.
# Diebold-Mariano tests, split-sample (pre/post COVID), relative RMSFE.
# =============================================================================

ml_results <- readRDS(here("data", "output", "ml_results.rds"))
rba_data   <- readRDS(here("data", "output", "rba_forecast_data.rds"))

# =============================================================================
# A. ML evaluation metrics by target, horizon, model
# =============================================================================
ml_eval <- ml_results %>%
  group_by(target, horizon, model) %>%
  summarise(
    rmsfe = sqrt(mean(forecast_error^2, na.rm = TRUE)),
    mafe  = mean(abs(forecast_error), na.rm = TRUE),
    bias  = mean(forecast_error, na.rm = TRUE),
    n     = n(),
    .groups = "drop"
  )

# =============================================================================
# B. Combine RBA and ML evaluations
# =============================================================================
rba_eval <- rba_data$eval %>%
  select(target, horizon, model, rmsfe, mafe, bias, n)

combined_eval <- bind_rows(rba_eval, ml_eval)

# =============================================================================
# C. Relative RMSFE (ML / RBA)
# =============================================================================
rba_rmsfe <- rba_eval %>%
  select(target, horizon, rba_rmsfe = rmsfe)

relative_eval <- ml_eval %>%
  left_join(rba_rmsfe, by = c("target", "horizon")) %>%
  mutate(relative_rmsfe = rmsfe / rba_rmsfe) %>%
  filter(!is.na(relative_rmsfe))

# =============================================================================
# D. Diebold-Mariano tests (each ML model vs RBA)
# =============================================================================
cat("Running Diebold-Mariano tests...\n")

rba_errors_list <- list(
  "CPI Inflation"        = rba_data$errors_cpi,
  "Underlying Inflation" = rba_data$errors_tm,
  "GDP Growth"           = rba_data$errors_gdp,
  "Unemployment Rate"    = rba_data$errors_ur
)

dm_results <- list()

for (target_name in names(rba_errors_list)) {
  rba_err <- rba_errors_list[[target_name]]

  for (h in c(0, 1, 2, 4, 8)) {
    rba_h <- rba_err %>%
      filter(horizon == h) %>%
      select(year_qtr, rba_error = forecast_error)

    ml_h <- ml_results %>%
      filter(target == target_name, horizon == h)

    for (mod in unique(ml_h$model)) {
      ml_mod <- ml_h %>%
        filter(model == mod) %>%
        select(year_qtr, ml_error = forecast_error)

      merged <- inner_join(rba_h, ml_mod, by = "year_qtr")
      if (nrow(merged) < 5) next

      dm <- dm_test(merged$rba_error, merged$ml_error, h = max(h, 1))

      dm_results[[length(dm_results) + 1]] <- tibble(
        target    = target_name,
        horizon   = h,
        model     = mod,
        dm_stat   = dm$statistic,
        dm_pvalue = dm$p.value,
        n_obs     = nrow(merged)
      )
    }
  }
}

dm_table <- bind_rows(dm_results)
cat("DM tests complete:", nrow(dm_table), "comparisons\n")

# =============================================================================
# E. Split-sample: Pre-COVID vs Post-COVID
# =============================================================================
COVID_BREAK <- yearquarter("2020 Q1")

split_sample_eval <- function(results_df, period_name, qtr_filter) {
  results_df %>%
    filter(qtr_filter(year_qtr)) %>%
    group_by(target, horizon, model) %>%
    summarise(
      rmsfe = sqrt(mean(forecast_error^2, na.rm = TRUE)),
      mafe  = mean(abs(forecast_error), na.rm = TRUE),
      n     = n(),
      .groups = "drop"
    ) %>%
    mutate(period = period_name)
}

ml_pre_covid  <- split_sample_eval(ml_results, "Pre-COVID",
                                    function(q) q < COVID_BREAK)
ml_post_covid <- split_sample_eval(ml_results, "Post-COVID",
                                    function(q) q >= COVID_BREAK)

# RBA split sample
rba_all_errors <- bind_rows(
  rba_data$errors_cpi %>% mutate(target = "CPI Inflation", model = "RBA"),
  rba_data$errors_tm  %>% mutate(target = "Underlying Inflation", model = "RBA"),
  rba_data$errors_gdp %>% mutate(target = "GDP Growth", model = "RBA"),
  rba_data$errors_ur  %>% mutate(target = "Unemployment Rate", model = "RBA")
)

rba_pre_covid  <- split_sample_eval(rba_all_errors, "Pre-COVID",
                                     function(q) q < COVID_BREAK)
rba_post_covid <- split_sample_eval(rba_all_errors, "Post-COVID",
                                     function(q) q >= COVID_BREAK)

split_eval <- bind_rows(ml_pre_covid, ml_post_covid, rba_pre_covid, rba_post_covid)

# =============================================================================
# F. Best ML model per target-horizon
# =============================================================================
best_ml <- ml_eval %>%
  group_by(target, horizon) %>%
  slice_min(rmsfe, n = 1) %>%
  ungroup() %>%
  rename(best_ml_model = model, best_ml_rmsfe = rmsfe)

# =============================================================================
# Save everything
# =============================================================================
eval_data <- list(
  combined_eval  = combined_eval,
  relative_eval  = relative_eval,
  dm_table       = dm_table,
  split_eval     = split_eval,
  best_ml        = best_ml,
  ml_eval        = ml_eval,
  rba_eval       = rba_eval
)
saveRDS(eval_data, here("data", "output", "evaluation_results.rds"))

cat("Evaluation complete.\n")
