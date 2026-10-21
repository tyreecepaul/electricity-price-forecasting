"""Compare forecasts with and without the graph-embedding features.

The arms share the region, horizon, target, splits, grids and seeds:

    baseline        the pipeline without the stage
    gnn             embeddings from the encoder over the NEM interconnector graph
    no_graph        the same encoder with cross-region edges removed
    gnn_plain       gnn without edge attributes or spread targets
    no_graph_plain  no_graph without spread targets

``gnn`` against ``baseline`` measures the stage, ``gnn`` against ``no_graph``
what the graph itself contributes, and each arm against its ``_plain``
version what edge-aware attention and the spread objective add.

    python compare_graph_embedding.py run --regions VIC1 NSW1
    python compare_graph_embedding.py summarise
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.contracts import score_forecast
from pipeline.evaluation import PRICE_REGIMES
from pipeline.graph_embedding import GraphEmbeddingConfig

ROOT = Path(__file__).resolve().parent / "reports" / "ablation"
PLAIN = {"edge_features": False, "spread_loss_weight": 0.0}
ARMS: dict[str, GraphEmbeddingConfig | None] = {
    "baseline": None,
    "gnn": GraphEmbeddingConfig(),
    "no_graph": GraphEmbeddingConfig(adjacency_mode="identity"),
    "gnn_plain": GraphEmbeddingConfig(**PLAIN),
    "no_graph_plain": GraphEmbeddingConfig(adjacency_mode="identity", **PLAIN),
}
COMPARISONS = (
    ("gnn", "baseline"),
    ("gnn", "no_graph"),
    ("no_graph", "baseline"),
    ("gnn", "gnn_plain"),
    ("no_graph", "no_graph_plain"),
)
# Models that receive graph embeddings; others must be requested by name.
GRAPH_MODELS = ("xgboost", "lstm")
SEEDS = (42, 1, 2)
# Tree fitting barely varies with the seed, so repeat seeds only for networks.
SINGLE_SEED_MODELS = {"xgboost"}
METRICS = ("mae", "rmse", "median_ae")
AEMO_COLUMN = re.compile(r"^p5_h\d+_rrp$")


def cell_directory(root: Path, region: str, horizon: int, arm: str, seed: int) -> Path:
    return root / f"{region}_{horizon}min" / f"{arm}_seed{seed}"


def model_directory(cell: Path, model: str) -> Path | None:
    """Return the most recent finished run of ``model`` inside one cell."""
    finished = sorted(
        path.parent
        for path in cell.glob(f"*/{model}/predictions.csv")
        if (path.parent.parent / "summary.csv").exists()
    )
    return finished[-1] if finished else None


def run(
    regions: Sequence[str],
    models: Sequence[str] | None,
    seeds: Sequence[int],
    horizon: int | None,
    root: Path,
) -> None:
    """Train every missing region x seed x arm cell through ``main.main``."""
    import main

    names = list(models or GRAPH_MODELS)
    horizon = horizon or main.EXPERIMENT.horizon_minutes
    for region in regions:
        for seed in seeds:
            wanted = [
                name
                for name in names
                if seed == seeds[0] or name not in SINGLE_SEED_MODELS
            ]
            for arm, embedding in ARMS.items():
                cell = cell_directory(root, region, horizon, arm, seed)
                missing = [n for n in wanted if model_directory(cell, n) is None]
                if not missing:
                    continue
                print(f"\n##### {region} {horizon}min {arm} seed {seed}: {missing}")
                main.main(
                    missing,
                    config=replace(
                        main.EXPERIMENT,
                        region=region,
                        horizon_minutes=horizon,
                        random_seed=seed,
                        graph_embedding=embedding,
                        forecast_mode="single",
                    ),
                    output_directory=cell,
                    # The arm applies to every model in the cell.
                    graph_embeddings={},
                )


def load_runs(root: Path, horizon: int) -> pd.DataFrame:
    """Index every saved model run by region, model, arm and seed."""
    records = []
    for group in sorted(root.glob(f"*_{horizon}min")):
        region = group.name.rsplit("_", 1)[0]
        for cell in sorted(group.iterdir()):
            match = re.fullmatch(r"(\w+)_seed(\d+)", cell.name)
            if match is None or match.group(1) not in ARMS:
                continue
            models = {path.parent.name for path in cell.glob("*/*/predictions.csv")}
            for model in sorted(models):
                directory = model_directory(cell, model)
                if directory is not None:
                    records.append(
                        {
                            "region": region,
                            "model": model,
                            "arm": match.group(1),
                            "seed": int(match.group(2)),
                            "directory": directory,
                        }
                    )
    return pd.DataFrame.from_records(records)


def paired_predictions(runs: pd.DataFrame) -> pd.DataFrame:
    """Align one model's forecasts from every arm and seed on shared valid rows.

    Returns ``actual``, ``persistence``, optionally ``aemo``, and one
    ``<arm>@<seed>`` column per run. Arms that scored different test rows are rejected
    rather than silently intersected.
    """
    joined, origins = None, None
    for run_ in runs.itertuples():
        frame = pd.read_csv(
            run_.directory / "predictions.csv", parse_dates=["settlement_date"]
        ).set_index("settlement_date")
        if origins is None:
            origins = frame.index
            joined = frame[["target_rrp", "rrp"]].rename(
                columns={"target_rrp": "actual", "rrp": "persistence"}
            )
            aemo = [column for column in frame.columns if AEMO_COLUMN.match(column)]
            if aemo:
                joined["aemo"] = frame[aemo[0]]
            joined["valid"] = True
        elif not frame.index.equals(origins):
            raise ValueError(
                f"{run_.arm} seed {run_.seed} scored different test rows from the "
                f"other {run_.model} runs in {run_.region}"
            )
        joined[f"{run_.arm}@{run_.seed}"] = frame["prediction_rrp"]
        joined["valid"] &= frame["valid"].astype(bool)
    joined = joined[joined.pop("valid")]
    return joined.dropna()


def arm_errors(paired: pd.DataFrame) -> pd.DataFrame:
    """Absolute error per interval for each arm, averaged over its seeds."""
    runs = [column for column in paired.columns if "@" in column]
    errors = {}
    for arm in dict.fromkeys(column.split("@")[0] for column in runs):
        seeds = [column for column in runs if column.split("@")[0] == arm]
        errors[arm] = (
            paired[seeds].sub(paired["actual"], axis=0).abs().mean(axis=1)
        )
    return pd.DataFrame(errors)


def day_block_bootstrap(
    difference: pd.Series,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of an interval-level series with a 95% interval, resampling days.

    Five-minute errors are strongly autocorrelated, so intervals are not
    independent draws. Whole days are resampled instead.
    """
    days = difference.groupby(difference.index.normalize()).agg(["sum", "count"])
    totals, counts = days["sum"].to_numpy(), days["count"].to_numpy()
    estimate = float(totals.sum() / counts.sum())
    picks = np.random.default_rng(seed).integers(0, len(days), (samples, len(days)))
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return estimate, float(low), float(high)


def regime_rows(actual: pd.Series) -> dict[str, np.ndarray]:
    masks = {}
    for name, lower, upper in PRICE_REGIMES:
        mask = np.ones(len(actual), dtype=bool)
        if lower is not None:
            mask &= (actual >= lower).to_numpy()
        if upper is not None:
            mask &= (actual < upper).to_numpy()
        if mask.any():
            masks[name] = mask
    return masks


def _metadata(directory: Path) -> dict:
    return json.loads((directory / "model.json").read_text(encoding="utf-8"))


def _percent_change(value: float, reference: float) -> float:
    return (value - reference) / reference * 100.0 if reference else float("nan")


def summarise_model(runs: pd.DataFrame) -> dict[str, list[dict]]:
    """Build every comparison table for one region and model."""
    region, model = runs["region"].iloc[0], runs["model"].iloc[0]
    key = {"region": region, "model": model}
    paired = paired_predictions(runs)
    actual = paired["actual"]
    errors = arm_errors(paired)
    by_arm = {arm: group for arm, group in runs.groupby("arm")}

    overall, seed_mae = [], {}
    for arm, group in by_arm.items():
        scores = pd.DataFrame(
            [
                {m: score_forecast(actual, paired[f"{arm}@{seed}"], m) for m in METRICS}
                for seed in group["seed"]
            ]
        )
        seed_mae[arm] = scores["mae"]
        validation = [_metadata(d).get("validation_score") for d in group["directory"]]
        overall.append(
            {
                **key,
                "arm": arm,
                "seeds": len(group),
                **scores.mean().to_dict(),
                "mae_min": scores["mae"].min(),
                "mae_max": scores["mae"].max(),
                "validation_score": float(np.mean(validation)),
                "n_rows": len(paired),
            }
        )
    for reference in ("persistence", "aemo"):
        if reference in paired:
            overall.append(
                {
                    **key,
                    "arm": reference,
                    **{m: score_forecast(actual, paired[reference], m) for m in METRICS},
                    "n_rows": len(paired),
                }
            )
    mae = {row["arm"]: row["mae"] for row in overall}
    for row in overall:
        for reference in ("baseline", "no_graph", "gnn_plain"):
            if reference in mae and row["arm"] != reference:
                row[f"mae_vs_{reference}_pct"] = _percent_change(
                    row["mae"], mae[reference]
                )

    paired_rows, regimes = [], []
    for arm, reference in COMPARISONS:
        if arm not in errors or reference not in errors:
            continue
        estimate, low, high = day_block_bootstrap(errors[arm] - errors[reference])
        spread = float(seed_mae[reference].max() - seed_mae[reference].min())
        paired_rows.append(
            {
                **key,
                "comparison": f"{arm}_vs_{reference}",
                "mae_difference": estimate,
                "ci_low": low,
                "ci_high": high,
                "pct": _percent_change(mae[arm], mae[reference]),
                # Negative differences favour the first-named arm.
                "interval_excludes_zero": bool(low > 0 or high < 0),
                "reference_seed_spread": spread if len(seed_mae[reference]) > 1 else np.nan,
                "exceeds_seed_spread": (
                    bool(abs(estimate) > spread) if len(seed_mae[reference]) > 1 else None
                ),
                "days": int(errors.index.normalize().nunique()),
            }
        )
    for regime, mask in regime_rows(actual).items():
        regime_mae = errors[mask].mean()
        for arm, value in regime_mae.items():
            regimes.append(
                {
                    **key,
                    "regime": regime,
                    "arm": arm,
                    "mae": value,
                    "n_rows": int(mask.sum()),
                    "mae_vs_baseline_pct": (
                        _percent_change(value, regime_mae["baseline"])
                        if "baseline" in regime_mae and arm != "baseline"
                        else np.nan
                    ),
                }
            )

    encoder, importance = [], []
    for arm, group in by_arm.items():
        directory = group["directory"].iloc[0]
        manifest = json.loads(
            (directory.parent / "experiment.json").read_text(encoding="utf-8")
        )
        diagnostics = manifest.get("graph_embedding_diagnostics")
        if diagnostics:
            for node, value in diagnostics["validation_mae"].items():
                encoder.append(
                    {
                        "region": region,
                        "arm": arm,
                        "node": node,
                        "encoder_validation_mae": value,
                        "zero_forecast_mae": diagnostics["zero_forecast_mae"][node],
                        "best_epoch": diagnostics["best_epoch"],
                    }
                )
        if model == "xgboost" and arm != "baseline":
            # Only load bundles this project wrote; joblib files run code on load.
            booster = joblib.load(directory / "model.joblib").model.get_booster()
            gain = booster.get_score(importance_type="total_gain")
            embedded = sum(v for name, v in gain.items() if name.startswith("gnn_emb_"))
            importance.append(
                {**key, "arm": arm, "embedding_gain_share": embedded / sum(gain.values())}
            )
    return {
        "overall": overall,
        "paired": paired_rows,
        "regimes": regimes,
        "encoder": encoder,
        "importance": importance,
    }


def summarise(horizon: int, root: Path) -> dict[str, pd.DataFrame]:
    """Write and print the comparison tables for every finished model."""
    runs = load_runs(root, horizon)
    if runs.empty:
        print(f"No finished runs for a {horizon}-minute horizon under {root}.")
        return {}
    tables: dict[str, list[dict]] = {}
    for (region, model), group in runs.groupby(["region", "model"]):
        if "baseline" not in set(group["arm"]) or group["arm"].nunique() < 2:
            print(f"Skipping {region} {model}: needs baseline and one other arm.")
            continue
        for name, rows in summarise_model(group).items():
            tables.setdefault(name, []).extend(rows)
    frames = {}
    for name, rows in tables.items():
        frame = pd.DataFrame.from_records(rows)
        if name == "encoder":
            # Every model in a cell shares one encoder.
            frame = frame.drop_duplicates()
        if frame.empty:
            continue
        frame.to_csv(root / f"{name}_{horizon}min.csv", index=False, float_format="%.4f")
        print(f"\n{name.upper()}")
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        frames[name] = frame
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "summarise"):
        command = commands.add_parser(name)
        command.add_argument("--horizon", type=int)
        command.add_argument("--root", type=Path, default=ROOT)
        if name == "run":
            command.add_argument("--regions", nargs="+", default=["VIC1"])
            command.add_argument("--models", nargs="+", type=str.lower)
            command.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(
            arguments.regions,
            arguments.models,
            arguments.seeds,
            arguments.horizon,
            arguments.root,
        )
    else:
        import main

        summarise(arguments.horizon or main.EXPERIMENT.horizon_minutes, arguments.root)
