# Agent Guide

Scope: this file applies to the whole repository.

## Repository Map

This is an R/Quarto research project for the paper "RBA vs Machine", comparing Reserve Bank of Australia forecasts with machine-learning and benchmark forecasts.

- `run_all.R` is the master reproducibility script. It sources the numbered R pipeline in order.
- `R/00_setup.R` installs/loads required R packages, creates output directories, and sources shared helpers.
- `R/01_data_pull.R` pulls ABS, RBA, FRED, and Yahoo Finance data using series lists in `data/config/`.
- `R/02_stationarity.R` applies stationarity transformations and writes processed panels/tests.
- `R/03_rba_forecasts.R` builds the RBA historical forecast and forecast-error data.
- `R/04_ml_models.R` and `R/04_ml_models_fast.R` train the forecast models. The master pipeline uses the fast version.
- `R/05_evaluation.R` evaluates forecast accuracy and robustness.
- `R/06_exhibits.R` writes publication figures to `figures/` and tables to `tables/`.
- `R/functions/` contains shared helpers for data pulls, transformations, ML fitting, and evaluation.
- `documents/` contains the Quarto/R Markdown manuscripts, bibliography, CSL file, LaTeX header, and rendered PDFs.
- `data/config/` contains committed source-series definitions. Treat these CSVs as inputs.
- `data/processed/` contains committed processed summaries. `data/output/` and `data/raw/` are generated/ignored.
- `figures/` and `tables/` contain rendered exhibits used by the manuscript.
- `Python/` contains exploratory notebook/utilities for importing data and stationarity checks; the canonical pipeline is currently in R.
- `resources/` contains research notes and reference PDFs.

## Common Commands

- Run the full pipeline:
  ```sh
  Rscript run_all.R
  ```
- Render the main manuscript:
  ```sh
  quarto render documents/RBAvsMachine.qmd
  ```
- Re-run only evaluation and exhibits after model outputs exist:
  ```sh
  Rscript R/run_eval_and_exhibits.R
  ```
- Check R syntax without running the full pipeline:
  ```sh
  Rscript -e "invisible(lapply(list.files('R', pattern = '\\\\.R$', recursive = TRUE, full.names = TRUE), parse))"
  ```

## Data And Outputs

- The data pull hits external services and may need network access plus a valid `FRED_API_KEY`. `R/01_data_pull.R` falls back to a hard-coded key if the environment variable is missing; prefer setting `FRED_API_KEY` in the environment for new runs.
- Do not commit generated `data/output/`, `data/raw/*.csv`, `.RData`, `.Rhistory`, `.Rproj.user/`, LaTeX intermediates, or training logs.
- Preserve committed configuration files in `data/config/`; changes there alter the empirical information set.
- `figures/` and `tables/` are committed outputs. If code changes affect exhibits, regenerate and review them before committing.

## Editing Guidelines

- Keep the numbered R pipeline modular. Prefer editing the relevant step and shared helper rather than adding hidden side effects elsewhere.
- Use `here::here()`/`here::i_am()` patterns already present in the R code instead of hard-coded absolute paths.
- Be careful with long-running or networked steps. For small edits, prefer syntax checks or targeted scripts before running the full pipeline.
- Treat the Quarto manuscripts as source files; generated `.tex` and `*_files/` artifacts are ignored unless explicitly requested.
- `documents/DatabasePaper.qmd` is draft material and currently contains merge-marker text in the committed file. Do not clean or rewrite it unless the task asks for that document.
