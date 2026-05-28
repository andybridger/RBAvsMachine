from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

HORIZONS = [0, 1, 2, 4, 8]
MIN_TRAIN = 40
MAX_FEATURES = 256
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

METADATA_COLUMNS = {
    "target",
    "horizon",
    "forecast_qtr",
    "year_qtr",
    "actual_value",
    "forecast_value",
    "forecast_error",
    "date",
    "forecast_date",
    "mean_forecast",
    "median_forecast",
    "mean_error",
    "median_error",
    "fit_status",
    "elapsed_sec",
    "n_train",
    "n_features_used",
}


@dataclass(frozen=True)
class ForecastTask:
    origin: str
    horizon: int
    target_qtr: str
    actual_value: float
    X_train: pl.DataFrame
    y_train: pl.Series
    X_test: pl.DataFrame
    selected_features: list[str]
    n_features_before: int


@dataclass(frozen=True)
class StandardizedData:
    X_train: pl.DataFrame
    y_train: pl.Series
    X_test: pl.DataFrame
    x_mean: np.ndarray
    x_sd: np.ndarray
    y_mean: float
    y_sd: float

    def invert_y(self, values: np.ndarray | list[float]) -> np.ndarray:
        values_np = np.asarray(values, dtype=np.float64)
        return values_np * self.y_sd + self.y_mean


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_quarter(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d{4})\s*[- ]?\s*Q([1-4])", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse quarter value: {value!r}")
    return f"{int(match.group(1)):04d}Q{int(match.group(2))}"


def period_sort_key(quarter: str) -> tuple[int, int]:
    normalized = normalize_quarter(quarter)
    return int(normalized[:4]), int(normalized[-1])


def add_quarters(quarter: str, n_quarters: int) -> str:
    year, quarter_num = period_sort_key(quarter)
    zero_based = (year * 4 + (quarter_num - 1)) + n_quarters
    new_year, new_quarter_zero = divmod(zero_based, 4)
    return f"{new_year:04d}Q{new_quarter_zero + 1}"


def sort_by_quarter(df: pl.DataFrame, column: str = "year_qtr") -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col(column)
            .map_elements(lambda q: period_sort_key(q)[0] * 4 + period_sort_key(q)[1], return_dtype=pl.Int64)
            .alias("_quarter_sort")
        )
        .sort("_quarter_sort")
        .drop("_quarter_sort")
    )


def normalize_quarter_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in df.columns:
        return df
    return df.with_columns(
        pl.col(column)
        .map_elements(normalize_quarter, return_dtype=pl.String)
        .alias(column)
    )


def normalize_quarter_columns(df: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    out = df
    for column in columns:
        out = normalize_quarter_column(out, column)
    return out


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and df.schema[column].is_numeric()
    ]


def _finite_float_array(values: pl.Series | np.ndarray) -> np.ndarray:
    arr = values.to_numpy() if isinstance(values, pl.Series) else np.asarray(values)
    return np.asarray(arr, dtype=np.float64)


def select_top_correlated_features(
    X: pl.DataFrame,
    y: pl.Series,
    max_features: int = MAX_FEATURES,
) -> list[str]:
    y_arr = _finite_float_array(y)
    scores: list[tuple[float, str]] = []
    for column in X.columns:
        x_arr = _finite_float_array(X[column])
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if valid.sum() < 2:
            continue
        x_valid = x_arr[valid]
        y_valid = y_arr[valid]
        x_sd = np.nanstd(x_valid, ddof=1)
        y_sd = np.nanstd(y_valid, ddof=1)
        if not np.isfinite(x_sd) or not np.isfinite(y_sd) or x_sd <= 1e-12 or y_sd <= 1e-12:
            continue
        corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
        if np.isfinite(corr):
            scores.append((abs(corr), column))

    ranked = sorted(scores, key=lambda item: (-item[0], item[1]))
    return [column for _, column in ranked[:max_features]]


def standardize_train_test(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
) -> StandardizedData:
    X_train_np = np.asarray(X_train.to_numpy(), dtype=np.float64)
    X_test_np = np.asarray(X_test.to_numpy(), dtype=np.float64)
    y_train_np = _finite_float_array(y_train)

    x_mean = np.nanmean(X_train_np, axis=0)
    x_sd = np.nanstd(X_train_np, axis=0, ddof=1)
    x_sd[~np.isfinite(x_sd) | (x_sd <= 1e-12)] = 1.0

    y_mean = float(np.nanmean(y_train_np))
    y_sd = float(np.nanstd(y_train_np, ddof=1))
    if not np.isfinite(y_sd) or y_sd <= 1e-12:
        y_sd = 1.0

    X_train_scaled = (X_train_np - x_mean) / x_sd
    X_test_scaled = (X_test_np - x_mean) / x_sd
    y_train_scaled = (y_train_np - y_mean) / y_sd

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return StandardizedData(
        X_train=pl.DataFrame(X_train_scaled, schema=X_train.columns, orient="row"),
        y_train=pl.Series(y_train.name, y_train_scaled),
        X_test=pl.DataFrame(X_test_scaled, schema=X_test.columns, orient="row"),
        x_mean=x_mean,
        x_sd=x_sd,
        y_mean=y_mean,
        y_sd=y_sd,
    )


def to_tabpfn_arrays(
    X_train_pl: pl.DataFrame,
    y_train_pl: pl.Series,
    X_test_pl: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_train_pl.to_numpy().astype("float32"),
        y_train_pl.to_numpy().ravel().astype("float32"),
        X_test_pl.to_numpy().astype("float32"),
    )


def build_forecast_task(
    panel: pl.DataFrame,
    actuals: pl.DataFrame,
    origin: str,
    horizon: int,
    min_train: int = MIN_TRAIN,
    max_features: int = MAX_FEATURES,
) -> ForecastTask | None:
    origin_norm = normalize_quarter(origin)
    target_qtr = add_quarters(origin_norm, horizon)

    panel_norm = sort_by_quarter(normalize_quarter_column(panel, "year_qtr"))
    actuals_norm = sort_by_quarter(normalize_quarter_column(actuals, "year_qtr"))

    actual_row = actuals_norm.filter(pl.col("year_qtr") == target_qtr)
    if actual_row.height == 0:
        return None
    actual_value = actual_row.get_column("actual_value")[0]
    if actual_value is None or not math.isfinite(float(actual_value)):
        return None

    origin_index = period_sort_key(origin_norm)[0] * 4 + period_sort_key(origin_norm)[1]
    panel_train = (
        panel_norm.with_columns(
            pl.col("year_qtr")
            .map_elements(lambda q: period_sort_key(q)[0] * 4 + period_sort_key(q)[1], return_dtype=pl.Int64)
            .alias("_quarter_index")
        )
        .filter(pl.col("_quarter_index") <= origin_index)
        .drop("_quarter_index")
    )
    if panel_train.height < min_train:
        return None

    candidate_features = feature_columns(panel_train)
    if not candidate_features:
        return None

    joined = panel_train.join(actuals_norm, on="year_qtr", how="left")
    if horizon > 0:
        y_values = joined.get_column("actual_value").slice(horizon)
        X_rows = joined.select(candidate_features).head(joined.height - horizon)
    else:
        y_values = joined.get_column("actual_value")
        X_rows = joined.select(candidate_features)

    valid_mask = y_values.is_not_null() & y_values.is_finite()
    X_train = X_rows.filter(valid_mask)
    y_train = y_values.filter(valid_mask).alias("actual_value")
    if y_train.len() < min_train:
        return None

    selected = select_top_correlated_features(X_train, y_train, max_features=max_features)
    if not selected:
        return None

    X_train_selected = X_train.select(selected)
    X_test_selected = panel_train.select(selected).tail(1)

    return ForecastTask(
        origin=origin_norm,
        horizon=horizon,
        target_qtr=target_qtr,
        actual_value=float(actual_value),
        X_train=X_train_selected,
        y_train=y_train,
        X_test=X_test_selected,
        selected_features=selected,
        n_features_before=len(candidate_features),
    )


def rmsfe(errors: pl.Series) -> float:
    arr = _finite_float_array(errors)
    valid = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(valid**2))) if valid.size else float("nan")


def mafe(errors: pl.Series) -> float:
    arr = _finite_float_array(errors)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(np.abs(valid))) if valid.size else float("nan")


def compute_interval_coverage(
    forecasts: pl.DataFrame,
    lower: str,
    upper: str,
    label: str,
) -> dict[str, float | str | int]:
    actual = _finite_float_array(forecasts["actual_value"])
    lower_arr = _finite_float_array(forecasts[lower])
    upper_arr = _finite_float_array(forecasts[upper])
    valid = np.isfinite(actual) & np.isfinite(lower_arr) & np.isfinite(upper_arr)
    if not valid.any():
        return {"interval": label, "coverage": float("nan"), "average_width": float("nan"), "n": 0}
    inside = (actual[valid] >= lower_arr[valid]) & (actual[valid] <= upper_arr[valid])
    width = upper_arr[valid] - lower_arr[valid]
    return {
        "interval": label,
        "coverage": float(inside.mean()),
        "average_width": float(width.mean()),
        "n": int(valid.sum()),
    }


def write_summary_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
