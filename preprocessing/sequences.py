from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceBatch:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    regions: np.ndarray
    feature_columns: list[str]
    target_columns: list[str]


def make_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_columns: list[str] | None = None,
    sequence_length: int = 288,
    drop_nan: bool = True,
    dtype: np.dtype = np.float32,
) -> SequenceBatch:
    """Convert a feature frame into model-agnostic sequence arrays.

    The result can be wrapped by PyTorch without coupling this data
    package to a particular deep-learning framework.
    """
    target_columns = target_columns or ["target_rrp"]
    missing = [c for c in [*feature_columns, *target_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sequence columns: {missing}")

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    times: list[np.datetime64] = []
    regions: list[str] = []

    for region, part in df.sort_values(["region", "settlement_date"]).groupby("region", sort=False):
        part = part.reset_index(drop=True)
        x = part[feature_columns].to_numpy(dtype=dtype)
        y = part[target_columns].to_numpy(dtype=dtype)
        t = part["settlement_date"].to_numpy()

        for end in range(sequence_length - 1, len(part)):
            start = end - sequence_length + 1
            x_window = x[start : end + 1]
            y_value = y[end]
            if drop_nan and (np.isnan(x_window).any() or np.isnan(y_value).any()):
                continue
            Xs.append(x_window)
            ys.append(y_value)
            times.append(t[end])
            regions.append(str(region))

    n_features = len(feature_columns)
    n_targets = len(target_columns)
    X = np.stack(Xs) if Xs else np.empty((0, sequence_length, n_features), dtype=dtype)
    y = np.stack(ys) if ys else np.empty((0, n_targets), dtype=dtype)

    return SequenceBatch(
        X=X,
        y=y,
        timestamps=np.asarray(times),
        regions=np.asarray(regions),
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
