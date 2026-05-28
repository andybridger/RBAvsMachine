# RBA Tabular Foundation Model Experiments

This directory is a uv-managed Python workspace for exploratory tabular
foundation model work. The canonical project pipeline remains the R pipeline in
the repository root.

The first notebook is:

```sh
uv run python -m ipykernel install --prefix .venv --name rba-tfm --display-name "Python (rba-tfm)"
uv run jupyter lab notebooks/tabpfn_cpi_exploration.ipynb
```

It reads the R-generated artifacts in `../data/output/`, forecasts CPI inflation
with TabPFN on the existing RBA forecast-origin grid from 2000Q1 onward, joins
the matching RBA CPI forecasts, and writes ignored experiment outputs under
`outputs/`.

The notebook uses TabPFN v2 by default so it can run headlessly without a
Prior Labs API token. Set `TABPFN_MODEL_VERSION = "v3"` in the notebook after
accepting the Prior Labs license and setting `TABPFN_TOKEN` if you want the
latest gated weights.

If `../data/output/` is missing, first run from the repository root:

```sh
Rscript run_all.R
```
