import math

import numpy as np
import polars as pl

from rba_tfm.cpi_notebook_helpers import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    REPO_ROOT,
    add_quarters,
    build_forecast_task,
    compute_interval_coverage,
    feature_columns,
    period_sort_key,
    select_top_correlated_features,
    standardize_train_test,
    to_tabpfn_arrays,
)


def test_project_paths_point_to_python_workspace_and_repo_root():
    assert PROJECT_ROOT.name == "Python"
    assert REPO_ROOT.name == "RBA"
    assert OUTPUT_DIR == PROJECT_ROOT / "outputs"


def test_add_quarters_rolls_year_boundary():
    assert add_quarters("1999Q4", 1) == "2000Q1"
    assert add_quarters("2000 Q1", 3) == "2000Q4"
    assert add_quarters("2000-Q2", 8) == "2002Q2"


def test_period_sort_key_orders_quarters_chronologically():
    quarters = ["2000Q1", "1999Q4", "2000Q3", "2000Q2"]
    assert sorted(quarters, key=period_sort_key) == ["1999Q4", "2000Q1", "2000Q2", "2000Q3"]


def test_feature_columns_excludes_metadata_and_target_columns():
    df = pl.DataFrame(
        {
            "year_qtr": ["2000Q1"],
            "actual_value": [1.0],
            "feature_a": [2.0],
            "feature_b": [3.0],
        }
    )
    assert feature_columns(df) == ["feature_a", "feature_b"]


def test_select_top_correlated_features_drops_constant_columns_and_tiebreaks_by_name():
    X = pl.DataFrame(
        {
            "b_feature": [1.0, 2.0, 3.0, 4.0],
            "a_feature": [4.0, 3.0, 2.0, 1.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    y = pl.Series("target", [1.0, 2.0, 3.0, 4.0])

    selected = select_top_correlated_features(X, y, max_features=2)

    assert selected == ["a_feature", "b_feature"]


def test_standardize_train_test_uses_train_statistics_only_and_fills_nan():
    X_train = pl.DataFrame({"x": [1.0, 2.0, 3.0], "constant": [4.0, 4.0, 4.0]})
    X_test = pl.DataFrame({"x": [4.0], "constant": [4.0]})
    y_train = pl.Series("target", [10.0, 12.0, 14.0])

    scaled = standardize_train_test(X_train, y_train, X_test)

    assert np.allclose(scaled.X_train.to_numpy()[:, 0], [-1.0, 0.0, 1.0])
    assert np.allclose(scaled.X_test.to_numpy()[:, 0], [2.0])
    assert np.allclose(scaled.X_train.to_numpy()[:, 1], [0.0, 0.0, 0.0])
    assert np.allclose(scaled.X_test.to_numpy()[:, 1], [0.0])
    assert np.allclose(scaled.y_train.to_numpy(), [-1.0, 0.0, 1.0])
    assert scaled.invert_y(np.array([0.0, 1.0])).tolist() == [12.0, 14.0]


def test_to_tabpfn_arrays_returns_float32_numpy_arrays():
    X_train = pl.DataFrame({"x": [1.0, 2.0]})
    y_train = pl.Series("target", [3.0, 4.0])
    X_test = pl.DataFrame({"x": [5.0]})

    X_train_np, y_train_np, X_test_np = to_tabpfn_arrays(X_train, y_train, X_test)

    assert X_train_np.dtype == np.float32
    assert y_train_np.dtype == np.float32
    assert X_test_np.dtype == np.float32
    assert X_train_np.shape == (2, 1)
    assert y_train_np.shape == (2,)
    assert X_test_np.shape == (1, 1)


def test_build_forecast_task_aligns_direct_horizon_and_test_row():
    panel = pl.DataFrame(
        {
            "year_qtr": ["2000Q1", "2000Q2", "2000Q3", "2000Q4", "2001Q1"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "year_qtr": ["2000Q1", "2000Q2", "2000Q3", "2000Q4", "2001Q1"],
            "actual_value": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    task = build_forecast_task(
        panel=panel,
        actuals=actuals,
        origin="2000Q3",
        horizon=1,
        min_train=2,
        max_features=10,
    )

    assert task is not None
    assert task.target_qtr == "2000Q4"
    assert task.actual_value == 40.0
    assert task.X_train.to_series().to_list() == [1.0, 2.0]
    assert task.y_train.to_list() == [20.0, 30.0]
    assert task.X_test.to_series().to_list() == [3.0]


def test_compute_interval_coverage_reports_rate_and_width():
    forecasts = pl.DataFrame(
        {
            "actual_value": [0.0, 2.0, 10.0],
            "q10": [-1.0, 1.0, 8.0],
            "q90": [1.0, 3.0, 9.0],
        }
    )

    result = compute_interval_coverage(forecasts, lower="q10", upper="q90", label="80")

    assert result["interval"] == "80"
    assert math.isclose(result["coverage"], 2 / 3)
    assert math.isclose(result["average_width"], 5 / 3)
