# =============================================================================
# Step 6: Publication-Quality Figures and Tables
# =============================================================================

eval_data <- readRDS(here("data", "output", "evaluation_results.rds"))
rba_data  <- readRDS(here("data", "output", "rba_forecast_data.rds"))
ml_results <- readRDS(here("data", "output", "ml_results.rds"))

# Colour palette: restrained, journal-quality
pal_models <- c(
  "RBA"             = "#1a1a1a",
  "Historical Mean" = "#bdbdbd",
  "AR(1)"           = "#969696",
  "Ridge"           = "#6baed6",
  "LASSO"           = "#3182bd",
  "Elastic Net"     = "#08519c",
  "Random Forest"   = "#e6550d",
  "XGBoost"         = "#a63603"
)

theme_paper <- theme_minimal(base_size = 11, base_family = "serif") +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0),
    plot.subtitle = element_text(size = 10, colour = "grey40", hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(face = "bold")
  )

TARGETS <- c("CPI Inflation", "Underlying Inflation", "GDP Growth", "Unemployment Rate")

# =============================================================================
# Figure 1: RBA Forecast Error Density Ridgeplots (one per target)
# =============================================================================
cat("Generating Figure 1: RBA forecast error densities...\n")

rba_all_errors <- bind_rows(
  rba_data$errors_cpi %>% mutate(target = "CPI Inflation"),
  rba_data$errors_tm  %>% mutate(target = "Underlying Inflation"),
  rba_data$errors_gdp %>% mutate(target = "GDP Growth"),
  rba_data$errors_ur  %>% mutate(target = "Unemployment Rate")
)

p1 <- rba_all_errors %>%
  filter(horizon <= 8) %>%
  ggplot(aes(x = forecast_error, y = factor(horizon), fill = factor(horizon))) +
  ggridges::geom_density_ridges(
    quantile_lines = TRUE, quantiles = 0.5,
    rel_min_height = 0.005, alpha = 0.7, scale = 1.2
  ) +
  facet_wrap(~ target, scales = "free_x", ncol = 2) +
  scale_fill_brewer(palette = "Blues") +
  labs(
    title = "Distribution of RBA Forecast Errors by Horizon",
    subtitle = "Vertical lines indicate medians. Sample: 1993 Q1 onwards.",
    x = "Forecast error (percentage points)",
    y = "Forecast horizon (quarters ahead)"
  ) +
  theme_paper +
  theme(legend.position = "none")

ggsave(here("figures", "fig1_rba_error_density.pdf"), p1, width = 9, height = 7)
ggsave(here("figures", "fig1_rba_error_density.png"), p1, width = 9, height = 7, dpi = 300)

# =============================================================================
# Figure 2: Spaghetti chart -- RBA forecasts vs actuals (post-2015)
# =============================================================================
cat("Generating Figure 2: Spaghetti forecast charts...\n")

make_spaghetti <- function(fc_df, actual_df, title, start = "2015-01-01") {
  plot_df <- fc_df %>%
    left_join(actual_df, by = "year_qtr") %>%
    filter(date >= as.Date(start))
  actual_line <- plot_df %>% distinct(date, actual_value)
  ggplot(plot_df, aes(x = date, y = forecast_value, group = forecast_date)) +
    geom_line(alpha = 0.25, colour = "#3182bd", linewidth = 0.4) +
    geom_line(data = actual_line, aes(x = date, y = actual_value),
              colour = "#1a1a1a", linewidth = 0.9, inherit.aes = FALSE) +
    labs(title = title, x = NULL, y = NULL) +
    theme_paper
}

# Extract forecast data frames
fc_cpi <- rba_data$raw_forecasts %>%
  filter(series == "cpi_annual_inflation") %>%
  mutate(forecast_qtr = yearquarter(forecast_date),
         year_qtr = yearquarter(date),
         horizon = as.numeric(year_qtr - forecast_qtr)) %>%
  filter(horizon >= 0, year(forecast_qtr) >= 1993) %>%
  select(year_qtr, forecast_qtr, forecast_value = value, horizon, date, forecast_date)

fc_ur <- rba_data$raw_forecasts %>%
  filter(series == "unemp_rate") %>%
  mutate(forecast_qtr = yearquarter(forecast_date),
         year_qtr = yearquarter(date),
         horizon = as.numeric(year_qtr - forecast_qtr)) %>%
  filter(horizon >= 0, year(forecast_qtr) >= 1993) %>%
  select(year_qtr, forecast_qtr, forecast_value = value, horizon, date, forecast_date)

p2a <- make_spaghetti(fc_cpi, rba_data$actuals$cpi, "CPI Inflation")
p2b <- make_spaghetti(fc_ur, rba_data$actuals$ur, "Unemployment Rate")

p2 <- cowplot::plot_grid(p2a, p2b, ncol = 1, labels = c("A", "B"))
ggsave(here("figures", "fig2_spaghetti.pdf"), p2, width = 8, height = 8)
ggsave(here("figures", "fig2_spaghetti.png"), p2, width = 8, height = 8, dpi = 300)

# =============================================================================
# Figure 3: Relative RMSFE bar charts (ML / RBA)
# =============================================================================
cat("Generating Figure 3: Relative RMSFE...\n")

p3 <- eval_data$relative_eval %>%
  filter(horizon %in% c(1, 2, 4, 8)) %>%
  mutate(
    horizon_label = paste0("h = ", horizon),
    model = factor(model, levels = c("Historical Mean", "AR(1)", "Ridge",
                                      "LASSO", "Elastic Net", "Random Forest", "XGBoost"))
  ) %>%
  ggplot(aes(x = model, y = relative_rmsfe, fill = model)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "#1a1a1a", linewidth = 0.5) +
  facet_grid(target ~ horizon_label) +
  scale_fill_manual(values = pal_models[-1]) +
  coord_flip() +
  labs(
    title = "Relative RMSFE: Machine Learning Models vs RBA",
    subtitle = "Values below 1 indicate the ML model outperforms the RBA. Dashed line = RBA benchmark.",
    x = NULL, y = "RMSFE relative to RBA"
  ) +
  theme_paper +
  theme(legend.position = "none",
        strip.text.y = element_text(angle = 0))

ggsave(here("figures", "fig3_relative_rmsfe.pdf"), p3, width = 11, height = 9)
ggsave(here("figures", "fig3_relative_rmsfe.png"), p3, width = 11, height = 9, dpi = 300)

# =============================================================================
# Figure 4: RMSFE comparison grouped bar chart
# =============================================================================
cat("Generating Figure 4: RMSFE comparison...\n")

p4 <- eval_data$combined_eval %>%
  filter(horizon %in% c(1, 4, 8)) %>%
  mutate(
    horizon_label = paste0("h = ", horizon),
    model = factor(model, levels = c("RBA", "Historical Mean", "AR(1)", "Ridge",
                                      "LASSO", "Elastic Net", "Random Forest", "XGBoost"))
  ) %>%
  ggplot(aes(x = model, y = rmsfe, fill = model)) +
  geom_col(width = 0.7) +
  facet_grid(target ~ horizon_label, scales = "free_y") +
  scale_fill_manual(values = pal_models) +
  coord_flip() +
  labs(
    title = "Root Mean Squared Forecast Error by Model and Horizon",
    x = NULL, y = "RMSFE (percentage points)"
  ) +
  theme_paper +
  theme(legend.position = "none",
        strip.text.y = element_text(angle = 0))

ggsave(here("figures", "fig4_rmsfe_comparison.pdf"), p4, width = 11, height = 9)
ggsave(here("figures", "fig4_rmsfe_comparison.png"), p4, width = 11, height = 9, dpi = 300)

# =============================================================================
# Figure 5: Pre-COVID vs Post-COVID comparison
# =============================================================================
cat("Generating Figure 5: Pre/post COVID...\n")

p5 <- eval_data$split_eval %>%
  filter(horizon %in% c(1, 4),
         model %in% c("RBA", "LASSO", "Random Forest", "XGBoost")) %>%
  mutate(
    horizon_label = paste0("h = ", horizon),
    model = factor(model, levels = c("RBA", "LASSO", "Random Forest", "XGBoost"))
  ) %>%
  ggplot(aes(x = model, y = rmsfe, fill = period)) +
  geom_col(position = "dodge", width = 0.7) +
  facet_grid(target ~ horizon_label, scales = "free_y") +
  scale_fill_manual(values = c("Pre-COVID" = "#6baed6", "Post-COVID" = "#e6550d")) +
  coord_flip() +
  labs(
    title = "Forecast Accuracy: Pre-COVID vs Post-COVID",
    subtitle = "Pre-COVID: before 2020 Q1. Post-COVID: 2020 Q1 onwards.",
    x = NULL, y = "RMSFE (percentage points)"
  ) +
  theme_paper +
  theme(strip.text.y = element_text(angle = 0))

ggsave(here("figures", "fig5_pre_post_covid.pdf"), p5, width = 10, height = 9)
ggsave(here("figures", "fig5_pre_post_covid.png"), p5, width = 10, height = 9, dpi = 300)

# =============================================================================
# Table 1: Main RMSFE results
# =============================================================================
cat("Generating Table 1: Main results...\n")

table1 <- eval_data$combined_eval %>%
  filter(horizon %in% c(0, 1, 2, 4, 8)) %>%
  select(target, horizon, model, rmsfe) %>%
  pivot_wider(names_from = model, values_from = rmsfe) %>%
  arrange(target, horizon) %>%
  mutate(across(where(is.numeric) & !matches("horizon"), ~ round(.x, 3)))

write_csv(table1, here("tables", "table1_rmsfe.csv"))

# =============================================================================
# Table 2: Relative RMSFE
# =============================================================================
table2 <- eval_data$relative_eval %>%
  filter(horizon %in% c(0, 1, 2, 4, 8)) %>%
  select(target, horizon, model, relative_rmsfe) %>%
  pivot_wider(names_from = model, values_from = relative_rmsfe) %>%
  arrange(target, horizon) %>%
  mutate(across(where(is.numeric) & !matches("horizon"), ~ round(.x, 3)))

write_csv(table2, here("tables", "table2_relative_rmsfe.csv"))

# =============================================================================
# Table 3: Diebold-Mariano test results
# =============================================================================
table3 <- eval_data$dm_table %>%
  filter(horizon %in% c(1, 4, 8)) %>%
  mutate(
    dm_stat = round(dm_stat, 2),
    dm_pvalue = round(dm_pvalue, 3),
    sig = case_when(
      dm_pvalue < 0.01 ~ "***",
      dm_pvalue < 0.05 ~ "**",
      dm_pvalue < 0.10 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  arrange(target, horizon, model)

write_csv(table3, here("tables", "table3_dm_tests.csv"))

# =============================================================================
# Table 4: Best ML model per target-horizon
# =============================================================================
table4 <- eval_data$best_ml %>%
  left_join(
    eval_data$rba_eval %>% select(target, horizon, rba_rmsfe = rmsfe),
    by = c("target", "horizon")
  ) %>%
  mutate(
    improvement_pct = round(100 * (1 - best_ml_rmsfe / rba_rmsfe), 1),
    best_ml_rmsfe = round(best_ml_rmsfe, 3),
    rba_rmsfe = round(rba_rmsfe, 3)
  ) %>%
  filter(horizon %in% c(0, 1, 2, 4, 8)) %>%
  arrange(target, horizon)

write_csv(table4, here("tables", "table4_best_ml.csv"))

cat("All exhibits generated.\n")
cat("Figures saved to:", here("figures"), "\n")
cat("Tables saved to:", here("tables"), "\n")
