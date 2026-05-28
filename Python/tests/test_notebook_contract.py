import json
from pathlib import Path


def test_tabpfn_cpi_notebook_is_valid_and_documents_core_contract():
    notebook_path = Path("notebooks/tabpfn_cpi_exploration.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    cell_sources = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        cell_sources.append("".join(source) if isinstance(source, list) else source)
    sources = "\n".join(cell_sources)

    assert notebook["nbformat"] == 4
    assert "def find_python_project_root() -> Path:" in sources
    assert 'sys.path.insert(0, str(PYTHON_PROJECT_ROOT / "src"))' in sources
    assert "import polars as pl" in sources
    assert "from tabpfn import TabPFNRegressor" in sources
    assert "from tabpfn.constants import ModelVersion" in sources
    assert "export_rds_inputs_to_csv" in sources
    assert "subprocess.run" in sources
    assert "to_tabpfn_arrays" in sources
    assert "output_type=\"quantiles\"" in sources
    assert "TABPFN_DEVICE" in sources
    assert "TABPFN_MODEL_VERSION = \"v2\"" in sources
    assert "TABPFN_MODEL_CACHE_DIR" in sources
    assert "create_default_for_version" in sources
    assert "FORECAST_START = \"2000Q1\"" in sources
    assert "WINDOW_MODE = \"expanding\"" in sources
    assert "PLOT_START = pd.Timestamp(\"1980-01-01\")" in sources
    assert "rba_available_pairs" in sources
    assert "\"rba_comparable_grid\": True" in sources
    assert "rba_forecast" in sources
    assert "rba_rmsfe" in sources
    assert "tabpfn_cpi_forecasts.parquet" in sources
    assert "tabpfn_cpi_interval_coverage.png" in sources
