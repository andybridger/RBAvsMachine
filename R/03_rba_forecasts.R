# =============================================================================
# Step 3: Extract RBA Forecasts and Compute Forecast Errors
# =============================================================================
# Following Tulip & Wallace (2012) methodology.
# Four targets: CPI, underlying inflation, GDP growth, unemployment rate.
# Horizons: 0 to 8 quarters ahead.
# =============================================================================

raw_forecasts <- readrba::read_forecasts()

# --- Actual values ---

# CPI inflation (year-ended)
raw_actual_cpi <- readrba::read_rba(series_id = "GCPIAGYP")
actual_cpi <- raw_actual_cpi %>%
  mutate(year_qtr = yearquarter(date)) %>%
  select(year_qtr, actual_value = value)

# Underlying inflation (trimmed mean, year-ended)
raw_actual_tm <- readrba::read_rba(series_id = "GCPIOCPMTMYP")
actual_tm <- raw_actual_tm %>%
  mutate(year_qtr = yearquarter(date)) %>%
  select(year_qtr, actual_value = value)

# GDP (chain volume, compute quarterly growth)
raw_actual_gdp <- readrba::read_rba(series_id = "GGDPCVGDP")
actual_gdp <- raw_actual_gdp %>%
  mutate(year_qtr = yearquarter(date)) %>%
  arrange(date) %>%
  mutate(actual_value = 100 * (value / lag(value) - 1)) %>%
  select(year_qtr, actual_value)

# Unemployment rate (monthly -> quarterly average)
raw_actual_ur <- readrba::read_rba(series_id = "GLFSURSA")
actual_ur <- raw_actual_ur %>%
  mutate(year_qtr = yearquarter(date)) %>%
  group_by(year_qtr) %>%
  summarise(actual_value = mean(value), .groups = "drop")

# --- Process each target ---
process_rba_target <- function(raw_fc, series_name, actual_df, start_year = 1993) {
  fc <- raw_fc %>%
    filter(series == series_name) %>%
    mutate(
      forecast_qtr = yearquarter(forecast_date),
      year_qtr = yearquarter(date),
      horizon = as.numeric(year_qtr - forecast_qtr)
    ) %>%
    filter(horizon >= 0, year(forecast_qtr) >= start_year) %>%
    select(year_qtr, forecast_qtr, forecast_value = value, horizon, date, forecast_date)

  errors <- fc %>%
    left_join(actual_df, by = "year_qtr") %>%
    filter(!is.na(actual_value)) %>%
    mutate(forecast_error = actual_value - forecast_value)

  errors
}

rba_errors_cpi <- process_rba_target(raw_forecasts, "cpi_annual_inflation", actual_cpi)
rba_errors_tm  <- process_rba_target(raw_forecasts, "underlying_annual_inflation", actual_tm)
rba_errors_gdp <- process_rba_target(raw_forecasts, "gdp_change", actual_gdp)
rba_errors_ur  <- process_rba_target(raw_forecasts, "unemp_rate", actual_ur)

# --- Compute RBA evaluation metrics by horizon ---
compute_rba_eval <- function(errors_df, target_name) {
  errors_df %>%
    filter(horizon <= 8) %>%
    group_by(horizon) %>%
    summarise(
      rmsfe = sqrt(mean(forecast_error^2, na.rm = TRUE)),
      mafe  = mean(abs(forecast_error), na.rm = TRUE),
      bias  = mean(forecast_error, na.rm = TRUE),
      n     = n(),
      .groups = "drop"
    ) %>%
    mutate(target = target_name, model = "RBA")
}

rba_eval_cpi <- compute_rba_eval(rba_errors_cpi, "CPI Inflation")
rba_eval_tm  <- compute_rba_eval(rba_errors_tm, "Underlying Inflation")
rba_eval_gdp <- compute_rba_eval(rba_errors_gdp, "GDP Growth")
rba_eval_ur  <- compute_rba_eval(rba_errors_ur, "Unemployment Rate")

rba_eval_all <- bind_rows(rba_eval_cpi, rba_eval_tm, rba_eval_gdp, rba_eval_ur)

# --- Save ---
rba_forecast_data <- list(
  errors_cpi = rba_errors_cpi,
  errors_tm  = rba_errors_tm,
  errors_gdp = rba_errors_gdp,
  errors_ur  = rba_errors_ur,
  actuals = list(cpi = actual_cpi, tm = actual_tm, gdp = actual_gdp, ur = actual_ur),
  eval = rba_eval_all,
  raw_forecasts = raw_forecasts
)
saveRDS(rba_forecast_data, here("data", "output", "rba_forecast_data.rds"))

cat("RBA forecast errors computed.\n")
cat("Targets processed:", paste(unique(rba_eval_all$target), collapse = ", "), "\n")
cat("Horizons: 0 to 8 quarters\n")
