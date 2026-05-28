# =============================================================================
# Step 0: Package Installation and Environment Setup
# =============================================================================

required_packages <- c(
  "data.table", "tidyverse", "lubridate", "here", "purrr",
  "readabs", "readrba", "fredr", "yahoofinancer",
  "tsibble", "tseries", "urca", "forecast", "vars",
  "glmnet", "randomForest", "xgboost",
  "ggplot2", "ggridges", "cowplot", "kableExtra", "scales",
  "sandwich", "lmtest"
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) {
  cat("Installing:", paste(new_packages, collapse = ", "), "\n")
  install.packages(new_packages, repos = "https://cloud.r-project.org", quiet = TRUE)
}

invisible(lapply(required_packages, library, character.only = TRUE))

here::i_am("run_all.R")

dir.create(here("data", "raw"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("data", "processed"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("data", "output"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("figures"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("tables"), showWarnings = FALSE, recursive = TRUE)

source(here("R", "functions", "data_pull.R"))
source(here("R", "functions", "transforms.R"))
source(here("R", "functions", "ml_helpers.R"))
source(here("R", "functions", "eval_helpers.R"))

cat("Setup complete.\n")
