#!/usr/bin/env Rscript
# =============================================================================
# RBA vs Machine: Master Pipeline Script
# =============================================================================
# Run this script to reproduce the entire analysis from data pull to results.
# Each step is modular and can be run independently.
# =============================================================================

cat("=== RBA vs Machine: Full Pipeline ===\n")
cat("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# Step 0: Setup and package installation
cat("--- Step 0: Setup ---\n")
source("R/00_setup.R")

# Step 1: Pull data from ABS, RBA, FRED, Yahoo Finance
cat("\n--- Step 1: Data Pull ---\n")
source("R/01_data_pull.R")

# Step 2: Apply stationarity transformations
cat("\n--- Step 2: Stationarity Transformations ---\n")
source("R/02_stationarity.R")

# Step 3: Extract RBA forecasts and compute errors
cat("\n--- Step 3: RBA Forecast Errors ---\n")
source("R/03_rba_forecasts.R")

# Step 4: Train ML models with expanding window
cat("\n--- Step 4: ML Model Training ---\n")
source("R/04_ml_models_fast.R")

# Step 5: Forecast evaluation and robustness tests
cat("\n--- Step 5: Evaluation ---\n")
source("R/05_evaluation.R")

# Step 6: Generate figures and tables
cat("\n--- Step 6: Exhibits ---\n")
source("R/06_exhibits.R")

cat("\n=== Pipeline Complete ===\n")
cat("Finished:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Render the manuscript with: quarto render documents/RBAvsMachine.qmd\n")
