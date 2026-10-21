"""Persist run configuration, scores, predictions and portable fitted models."""

from __future__ import annotations

import copy
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd


def create_run_directory(config, output_directory: Path, prepared=None) -> Path:
    """Create a unique run folder and record configuration/data/environment provenance."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_directory / f"{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment": asdict(config),
        "data_metadata": config.cache_metadata(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "torch", "scikit-learn", "xgboost")
        },
    }
    diagnostics = getattr(prepared, "embedding_diagnostics", None)
    if diagnostics:
        manifest["graph_embedding_diagnostics"] = diagnostics
    (directory / "experiment.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return directory


def save_model_run(
    directory,
    model,
    search_result,
    test_df,
    predictions,
    result,
    extra_columns=(),
    prepared=None,
    graph_embedding=None,
):
    """Save one selected model, its search results and aligned test-price forecasts.

    The bundle includes preprocessing and context for inference, not optimizer
    state for resuming training. Only load trusted joblib bundles.
    """
    path = directory / model.get_model_name().lower().replace(" ", "-")
    path.mkdir(exist_ok=False)
    search_result.trials.to_csv(path / "trials.csv", index=False)
    search_result.folds.to_csv(path / "folds.csv", index=False)
    metadata = {
        "model": model.get_model_name(),
        "best_config": search_result.best_config,
        "refit_config": search_result.refit_config,
        "selection_metric": search_result.metric,
        "validation_score": search_result.best_score,
        "test_metrics": result.metrics,
        "features": getattr(model, "feature_columns", []),
        # Models in one run may train on different graph-embedding features.
        "graph_embedding": asdict(graph_embedding) if graph_embedding else None,
        "graph_embedding_diagnostics": getattr(
            prepared, "embedding_diagnostics", None
        ),
        "training": asdict(model.training_result)
        if getattr(model, "training_result", None)
        else None,
    }
    (path / "model.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    rows = test_df[
        [
            c
            for c in (
                "region",
                "settlement_date",
                "rrp",
                "target_rrp",
                "target",
                *extra_columns,
            )
            if c in test_df
        ]
    ].copy()
    rows["prediction_rrp"] = predictions
    rows["valid"] = result.valid_mask
    rows.to_csv(path / "predictions.csv", index=False)
    # Move a copy to CPU so the bundle can load without a GPU while the caller's
    # live model retains its original device and state.
    portable = copy.deepcopy(model)
    if getattr(portable, "network", None) is not None:
        import torch

        portable.network.cpu()
        portable.device = torch.device("cpu")
    # Includes fitted imputers, scalers, target transform and temporal context.
    joblib.dump(portable, path / "model.joblib")


def save_run_summary(directory, results):
    """Save each model's test metrics; the console also reports paired comparisons."""
    pd.DataFrame([{"model": r.model_name, **r.metrics} for r in results]).to_csv(
        directory / "summary.csv", index=False
    )
