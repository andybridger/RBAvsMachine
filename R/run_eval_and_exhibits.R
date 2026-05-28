# Quick runner for steps 5 + 6 (evaluation and exhibits)
library(here); here::i_am("run_all.R")
library(tidyverse); library(tsibble); library(ggridges); library(cowplot)
source(here("R", "functions", "eval_helpers.R"))

cat("--- Step 5: Evaluation ---\n")
source(here("R", "05_evaluation.R"))

cat("\n--- Step 6: Exhibits ---\n")
source(here("R", "06_exhibits.R"))

cat("\nAll done!\n")
