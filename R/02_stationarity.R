# =============================================================================
# Step 2: Stationarity Transformations
# =============================================================================
# Apply ADF-based transformation selection following FRED-MD conventions.
# Save transformation codes and transformed data.
# =============================================================================

df_data <- readRDS(here("data", "output", "df_data_all.rds"))

# --- Select optimal transformation for each series ---
cat("Selecting transformations via ADF tests...\n")

tcode_table <- df_data %>%
  group_by(series_id, series, package, frequency) %>%
  summarise(
    tcode = select_tcode(value),
    n_obs = n(),
    start_date = min(date),
    end_date = max(date),
    .groups = "drop"
  ) %>%
  mutate(transformation = tcode_label(tcode))

cat("Transformation distribution:\n")
print(table(tcode_table$tcode))

# --- Apply transformations ---
cat("Applying transformations...\n")

df_transformed <- df_data %>%
  left_join(tcode_table %>% select(series_id, tcode), by = "series_id") %>%
  group_by(series_id) %>%
  arrange(date) %>%
  mutate(value_transformed = apply_tcode(value, first(tcode))) %>%
  ungroup() %>%
  filter(!is.na(value_transformed))

# --- Create wide-format quarterly panel for ML ---
cat("Creating quarterly panel...\n")

df_quarterly <- df_transformed %>%
  filter(frequency == "Quarterly") %>%
  mutate(year_qtr = tsibble::yearquarter(date)) %>%
  select(year_qtr, series_id, value_transformed) %>%
  pivot_wider(names_from = series_id, values_from = value_transformed) %>%
  arrange(year_qtr)

# For monthly data aggregated to quarterly
df_monthly_to_qtr <- df_transformed %>%
  filter(frequency == "Monthly") %>%
  mutate(year_qtr = tsibble::yearquarter(date)) %>%
  group_by(year_qtr, series_id) %>%
  summarise(value_transformed = mean(value_transformed, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = series_id, values_from = value_transformed) %>%
  arrange(year_qtr)

# Merge monthly (aggregated) and quarterly into one panel
df_panel <- df_quarterly %>%
  left_join(df_monthly_to_qtr, by = "year_qtr", suffix = c("", ".mth"))

# Remove columns with >30% missing values
na_frac <- colMeans(is.na(df_panel))
keep_cols <- names(na_frac)[na_frac < 0.3]
df_panel <- df_panel[, keep_cols]

# Forward fill remaining NAs then drop any residual
df_panel <- df_panel %>%
  arrange(year_qtr) %>%
  mutate(across(-year_qtr, ~ zoo::na.locf(.x, na.rm = FALSE))) %>%
  drop_na()

cat("Panel dimensions:", nrow(df_panel), "quarters x", ncol(df_panel) - 1, "series\n")

# --- Save ---
saveRDS(df_transformed, here("data", "output", "df_transformed.rds"))
saveRDS(df_panel, here("data", "output", "df_panel_quarterly.rds"))
write_csv(tcode_table, here("data", "output", "transformation_codes.csv"))

cat("Stationarity transformations complete.\n")
