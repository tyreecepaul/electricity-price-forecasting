"""Framework-independent contracts for forecast values and temporal inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = ("mae", "rmse", "sse", "median_ae")


def forecast_vector(values, expected_rows: int, *, name="predictions") -> np.ndarray:
    """Normalize single-horizon outputs to (N,) and reject other dimensions.

    A neural head often returns (N, 1). Adding that directly to an (N,) price
    baseline would broadcast into an incorrect (N, N) matrix.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != expected_rows:
        raise ValueError(
            f"{name} must have shape ({expected_rows},) or ({expected_rows}, 1); "
            f"got {array.shape}"
        )
    return array


def score_forecast(actual, predicted, metric: str) -> float:
    """Score already aligned, finite vectors on their supplied target scale.

    Callers select eligible rows and validate shapes before reaching this helper.
    All supported metrics are errors, so lower values are better.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}")
    errors = np.asarray(actual) - np.asarray(predicted)
    if metric == "mae":
        return float(np.mean(np.abs(errors)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(errors**2)))
    if metric == "sse":
        return float(np.sum(errors**2))
    return float(np.median(np.abs(errors)))


def indexed_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Keep temporal metadata in the index, never among network features."""
    return frame.set_index(["region", "settlement_date"])[features].copy()


def temporal_metadata(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Read supported temporal indices; None means only positional data is available."""
    if isinstance(frame.index, pd.MultiIndex) and set(frame.index.names) == {
        "region",
        "settlement_date",
    }:
        return frame.index.to_frame(index=False)[["region", "settlement_date"]]
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame({"region": "single", "settlement_date": frame.index})
    return None


def validate_timeline(metadata: pd.DataFrame, interval_minutes: int = 5) -> None:
    """Reject gaps, mixed series, duplicate times and reversed order.

    This single-series contract deliberately fails on missing intervals instead
    of silently treating irregularly spaced observations as adjacent timesteps.
    """
    if metadata.empty:
        return
    if metadata["region"].isna().any() or metadata["region"].nunique() != 1:
        raise ValueError("Sequence inputs must contain exactly one region")
    times = pd.to_datetime(metadata["settlement_date"])
    if (
        times.isna().any()
        or times.duplicated().any()
        or not times.is_monotonic_increasing
    ):
        raise ValueError("Sequence timestamps must be unique and increasing")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if not times.diff().iloc[1:].eq(pd.Timedelta(minutes=interval_minutes)).all():
        raise ValueError(
            "Sequence timeline has gaps; supply contiguous observed history"
        )


def validate_context(history: pd.DataFrame, current: pd.DataFrame, interval_minutes=5):
    """Check that history and current rows form one continuous chronological series.

    Validating the concatenation also checks the split boundary, catching stale
    history, overlapping/future rows, and a change of region. Two unindexed
    frames retain positional behavior and cannot receive these checks.
    """
    previous = temporal_metadata(history)
    following = temporal_metadata(current)
    if (previous is None) != (following is None):
        raise ValueError("history and current inputs must both carry temporal metadata")
    if previous is not None:
        validate_timeline(
            pd.concat([previous, following], ignore_index=True), interval_minutes
        )
