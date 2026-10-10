"""Node-aligned tensors built from the all-region prepared feature frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.contracts import validate_timeline

from .config import DEFAULT_NODE_FEATURES, GraphEmbeddingConfig


@dataclass
class GraphPanel:
    """Features and targets on one complete time axis shared by every region.

    x: (time, num_nodes, num_features); y: (time, num_nodes) on the experiment's
    target scale. Intervals a region did not report are NaN, never dropped, so
    row offsets always correspond to elapsed time.
    """

    times: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    regions: tuple[str, ...]
    node_features: tuple[str, ...]


def resolve_node_features(
    frame: pd.DataFrame,
    config: GraphEmbeddingConfig,
    horizon: int,
) -> tuple[str, ...]:
    """Choose configured node inputs that exist and carry data in ``frame``."""
    if config.node_features is not None:
        missing = [name for name in config.node_features if name not in frame.columns]
        if missing:
            raise ValueError(f"Graph node features are missing: {missing}")
        return tuple(dict.fromkeys(config.node_features))
    wanted = [*DEFAULT_NODE_FEATURES, f"p5_h{horizon:02d}_rrp_vs_now"]
    selected = tuple(
        name for name in wanted if name in frame.columns and frame[name].notna().any()
    )
    if not selected:
        raise ValueError("No graph node features are available in the prepared data")
    return selected


def build_graph_panel(
    frame: pd.DataFrame,
    regions: Sequence[str],
    node_features: Sequence[str],
    *,
    interval_minutes: int = 5,
) -> GraphPanel:
    """Pivot a long (region, settlement_date) frame that has a ``target`` column."""
    present = set(frame["region"].unique())
    absent = [region for region in regions if region not in present]
    if absent:
        raise ValueError(f"Graph regions are missing from the prepared data: {absent}")

    moments = pd.to_datetime(frame["settlement_date"])
    times = pd.date_range(
        moments.min(), moments.max(), freq=pd.Timedelta(minutes=interval_minutes)
    )
    validate_timeline(
        pd.DataFrame({"region": "graph", "settlement_date": times}), interval_minutes
    )

    columns = [*node_features, "target"]
    x = np.full((len(times), len(regions), len(node_features)), np.nan, dtype=np.float32)
    y = np.full((len(times), len(regions)), np.nan, dtype=np.float32)
    for position, region in enumerate(regions):
        node = frame.loc[frame["region"] == region].set_index("settlement_date")
        if node.index.has_duplicates:
            raise ValueError(f"Region {region} has duplicate timestamps")
        values = (
            node[columns]
            .apply(pd.to_numeric, errors="coerce")
            .reindex(times)
            .to_numpy(dtype=np.float32)
        )
        x[:, position, :] = values[:, :-1]
        y[:, position] = values[:, -1]
    x[~np.isfinite(x)] = np.nan
    y[~np.isfinite(y)] = np.nan
    return GraphPanel(times, x, y, tuple(regions), tuple(node_features))
