# =============================================================================
# Step 1: Pull Data from ABS, RBA, FRED, Yahoo Finance
# =============================================================================
# End date: 2025-12-31
# Data saved as compressed RDS to minimise disk space.
# =============================================================================

END_DATE <- as.Date("2025-12-31")
START_DATE <- as.Date("1990-01-01")

output_file <- here("data", "output", "df_data_all.rds")

if (file.exists(output_file)) {
  cat("Data file already exists:", output_file, "\n")
  cat("Delete it to re-pull. Loading existing data.\n")
  df_data <- readRDS(output_file)
} else {

  # --- ABS ---
  cat("Pulling ABS data...\n")
  abs_series <- read_csv(here("data", "config", "abs_series.csv"), show_col_types = FALSE)
  df_abs_raw <- pull_abs_data(abs_series$series_id, chunk_size = 50)
  missing_abs <- get_missing_series(abs_series$series_id, df_abs_raw)
  if (length(missing_abs) > 0) cat("  Missing ABS series:", length(missing_abs), "\n")
  df_abs <- standardize_abs(df_abs_raw)
  cat("  ABS:", n_distinct(df_abs$series_id), "series,", nrow(df_abs), "obs\n")

  # --- RBA ---
  cat("Pulling RBA data...\n")
  rba_series <- read_csv(here("data", "config", "rba_series.csv"), show_col_types = FALSE)
  df_rba_raw <- pull_rba_data(rba_series$series_id)
  missing_rba <- get_missing_series(rba_series$series_id, df_rba_raw)
  if (length(missing_rba) > 0) cat("  Missing RBA series:", length(missing_rba), "\n")
  df_rba <- standardize_rba(df_rba_raw)
  cat("  RBA:", n_distinct(df_rba$series_id), "series,", nrow(df_rba), "obs\n")

  # --- Yahoo Finance ---
  cat("Pulling Yahoo Finance data...\n")
  yahoo_tickers <- read_csv(here("data", "config", "yahoo_tickers.csv"), show_col_types = FALSE)
  df_yahoo_raw <- pull_yahoo_data(yahoo_tickers,
                                   start_date = as.character(START_DATE),
                                   end_date = as.character(END_DATE))
  df_yahoo <- standardize_yahoo(df_yahoo_raw)
  cat("  Yahoo:", n_distinct(df_yahoo$series_id), "series,", nrow(df_yahoo), "obs\n")

  # --- FRED ---
  cat("Pulling FRED data...\n")
  fred_series <- read_csv(here("data", "config", "fred_series.csv"), show_col_types = FALSE)
  fred_api_key <- Sys.getenv("FRED_API_KEY")
  if (fred_api_key == "") fred_api_key <- "493222c4777291b6ef8631077cd167bb"
  fredr_set_key(fred_api_key)
  df_fred_raw <- pull_fred_data(fred_series,
                                 start_date = as.character(START_DATE),
                                 end_date = as.character(END_DATE))
  df_fred <- standardize_fred(df_fred_raw)
  cat("  FRED:", n_distinct(df_fred$series_id), "series,", nrow(df_fred), "obs\n")

  # --- Combine ---
  df_data <- bind_rows(df_abs, df_rba, df_yahoo, df_fred) %>%
    mutate(frequency = case_match(
      frequency,
      "Month" ~ "Monthly",
      "Quarter" ~ "Quarterly",
      .default = frequency
    )) %>%
    filter(frequency %in% c("Monthly", "Quarterly")) %>%
    filter(date <= END_DATE)

  saveRDS(df_data, output_file)
  cat("\nCombined data saved:", format(nrow(df_data), big.mark = ","), "observations,",
      n_distinct(df_data$series_id), "series\n")
}

cat("Data pull complete.\n")
