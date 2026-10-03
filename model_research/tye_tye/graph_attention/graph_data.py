"""Local, self-contained graph-tensor construction for the GAT experiment.

Builds a joint [time, num_nodes, features] tensor across all 5 NEM regions by
reusing preprocessing.adapters.prepare_model_data() once per region (read-only
reuse of the shared feature-engineering pipeline) and aligning the per-region
frames on settlement_date.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.adapters import prepare_model_data

REGION_ORDER: tuple[str, ...] = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")

# Fixed NEM interconnector topology (undirected), including self-loops.
# NSW1-QLD1 (QNI), NSW1-VIC1 (VNI), VIC1-SA1 (Heywood/Murraylink), VIC1-TAS1 (Basslink).
NEM_ADJACENCY = np.array(
    [
        # NSW  QLD  VIC  SA  TAS
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1],
    ],
    dtype=np.float32,
)

DEFAULT_FEATURES: tuple[str, ...] = (
    "rrp",
    "totaldemand",
    "availablegeneration",
    "reserve_margin_mw",
    "reserve_ratio",
    "time_sin",
    "time_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "rrp_lag_288",
    "rrp_delta_1",
    "rrp_delta_3",
    "rrp_delta_6",
    "rrp_roll_mean_12",
    "rrp_roll_std_12",
    "rrp_roll_mean_288",
)


@dataclass
class GraphSequenceBatch:
    """Windowed, node-aligned tensors for the graph model.

    X: [n_samples, sequence_length, num_nodes, n_features]
    y: [n_samples, num_nodes] (target for each region, at the horizon already
       encoded in the shared ``target`` column from prepare_model_data)
    """

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    regions: tuple[str, ...]
    feature_columns: tuple[str, ...]


def _region_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    columns = ["settlement_date", *feature_columns, "target"]
    return df[columns].set_index("settlement_date")


def _build_batch(
    per_region_frames: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    sequence_length: int,
    regions: Sequence[str],
) -> GraphSequenceBatch:
    # Align every region on the intersection of available timestamps.
    aligned_index = None
    for frame in per_region_frames.values():
        aligned_index = (
            frame.index if aligned_index is None else aligned_index.intersection(frame.index)
        )
    aligned_index = aligned_index.sort_values()

    feature_stack = []
    target_stack = []
    for region in regions:
        frame = per_region_frames[region].loc[aligned_index]
        feature_stack.append(frame[list(feature_columns)].to_numpy(dtype=np.float32))
        target_stack.append(frame["target"].to_numpy(dtype=np.float32))

    # [time, num_nodes, n_features]
    features_by_time = np.stack(feature_stack, axis=1)
    targets_by_time = np.stack(target_stack, axis=1)  # [time, num_nodes]

    n_time = features_by_time.shape[0]
    if n_time < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} aligned timestamps, got {n_time}."
        )

    n_samples = n_time - sequence_length + 1
    X = np.stack(
        [features_by_time[i : i + sequence_length] for i in range(n_samples)],
        axis=0,
    )
    y = targets_by_time[sequence_length - 1 :]
    timestamps = aligned_index.to_numpy()[sequence_length - 1 :]

    return GraphSequenceBatch(
        X=X,
        y=y,
        timestamps=timestamps,
        regions=tuple(regions),
        feature_columns=tuple(feature_columns),
    )


def load_graph_data(
    *,
    regions: Sequence[str] = REGION_ORDER,
    horizon: int = 30,
    target: str = "change",
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    sequence_length: int = 288,
) -> tuple[GraphSequenceBatch, GraphSequenceBatch]:
    """Load train/test graph-sequence batches across all requested regions."""

    from data_cleaning.data import getTrainTestData

    train_data, test_data = getTrainTestData()

    train_frames: dict[str, pd.DataFrame] = {}
    test_frames: dict[str, pd.DataFrame] = {}
    for region in regions:
        prepared = prepare_model_data(
            region=region,
            horizon=horizon,
            features=list(feature_columns),
            target=target,
            train_data=train_data,
            test_data=test_data,
        )
        train_frames[region] = _region_frame(prepared.train_df, feature_columns)
        test_frames[region] = _region_frame(prepared.test_df, feature_columns)

    train_batch = _build_batch(train_frames, feature_columns, sequence_length, regions)
    test_batch = _build_batch(test_frames, feature_columns, sequence_length, regions)
    return train_batch, test_batch
