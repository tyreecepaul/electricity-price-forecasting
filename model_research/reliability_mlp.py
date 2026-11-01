"""Train the baseline MLP with and without the forecast reliability features.

The MLP class and its settings come unchanged from main.py; the only difference
between the two versions is the 18 extra inputs from reliability_features.py.
Each version follows main.py's steps: epochs chosen on the validation period,
refit on all pre-test data, then scored once on the test year. Three seeds are
used because the MLP's error moves by a few percent between seeds.

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reliability_features import add_reliability_features, baseline_experiment  # also makes the repo importable

from src.final_model.mlp import MLPModel  # noqa: E402
from main import MODEL_REGISTRY  # noqa: E402
from pipeline.evaluation import evaluate  # noqa: E402
from pipeline.experiment_pipeline import reconstruct_price_forecast  # noqa: E402
from pipeline.model_data import final_training_data, predict_test  # noqa: E402
from pipeline.tuning import search_hyperparameters  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "reliability_runs"
REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
SEEDS = (42, 1, 2)
VERSIONS = ("baseline", "reliability")
MLP_SETTINGS = next(grid for model, grid in MODEL_REGISTRY if model is MLPModel)
SPIKE_LEVEL = 300.0


def run_path(runs: Path, region: str, horizon: int, version: str, seed: int) -> Path:
    return runs / f"{region}_{horizon}min" / f"{version}_seed{seed}.csv"


def train(regions=REGIONS, horizon: int = 5, seeds=SEEDS, runs: Path = RUNS) -> None:
    for region in regions:
        base = baseline_experiment(region, horizon)
        for version in VERSIONS:
            experiment = base if version == "baseline" else add_reliability_features(base)
            for seed in seeds:
                path = run_path(runs, region, horizon, version, seed)
                if path.exists():
                    continue
                started = time.time()
                search = search_hyperparameters(MLPModel, {**MLP_SETTINGS, "random_seed": [seed]}, experiment)
                model = MLPModel(**search.best_config)
                model.train(*final_training_data(model, experiment.prepared))
                change = predict_test(model, experiment.prepared)
                test = experiment.test_df
                price = reconstruct_price_forecast(test, change, experiment.config)
                result = evaluate(model, test["target_rrp"].to_numpy(), price)
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "settlement_date": test["settlement_date"].to_numpy(),
                    "rrp": test["rrp"].to_numpy(),
                    "target_rrp": test["target_rrp"].to_numpy(),
                    "aemo": test[f"p5_h{horizon:02d}_rrp"].to_numpy(),
                    "prediction_rrp": price,
                    "valid": result.valid_mask,
                }).to_csv(path, index=False)
                print(f"{region} {horizon}min {version} seed {seed}: {len(experiment.features)} inputs, "
                      f"test MAE {result.metrics['mae']:.3f} ({(time.time() - started) / 60:.1f} min)", flush=True)


def summarise(horizon: int = 5, runs: Path = RUNS) -> pd.DataFrame:
    """Daily sums of absolute errors per state, version and seed: all the notebook needs."""
    tables = []
    for region in REGIONS:
        paths = {(version, seed): run_path(runs, region, horizon, version, seed)
                 for version in VERSIONS for seed in SEEDS}
        if not all(path.exists() for path in paths.values()):
            continue
        joined = None
        for (version, seed), path in paths.items():
            run = pd.read_csv(path, parse_dates=["settlement_date"]).set_index("settlement_date")
            run = run[run["valid"].astype(bool)]
            column = run[["prediction_rrp"]].rename(columns={"prediction_rrp": f"{version}_seed{seed}"})
            joined = (run[["target_rrp", "aemo"]].join(column) if joined is None
                      else joined.join(column, how="inner"))
        joined = joined.dropna()
        actual = joined["target_rrp"]
        errors = {name: (joined[name] - actual).abs() for name in joined.columns if name != "target_rrp"}
        spike = actual >= SPIKE_LEVEL
        daily = pd.DataFrame({"intervals": 1, "spike_intervals": spike.astype(int)}, index=joined.index)
        for name, error in errors.items():
            daily[f"{name}_abs"] = error
            daily[f"{name}_spike_abs"] = error.where(spike, 0.0)
        daily = daily.groupby(daily.index.normalize()).sum()
        daily.index.name = "day"
        tables.append(daily.reset_index().assign(region=region))
    if not tables:
        print("No state has all six runs yet, so the summary was not written.")
        return pd.DataFrame()
    summary = pd.concat(tables, ignore_index=True)
    summary = summary[["region", *[c for c in summary.columns if c != "region"]]]
    summary.to_csv(HERE / f"reliability_results_{horizon}min.csv", index=False, float_format="%.4f")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regions", nargs="+", default=list(REGIONS))
    parser.add_argument("--horizon", type=int, choices=[5, 30], default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--summary-only", action="store_true", help="only rebuild the summary from saved runs")
    arguments = parser.parse_args()
    if not arguments.summary_only:
        train(arguments.regions, arguments.horizon, arguments.seeds, arguments.runs_dir)
    summary = summarise(arguments.horizon, arguments.runs_dir)
    if not summary.empty:
        print(f"summary: {len(summary)} state-days -> reliability_results_{arguments.horizon}min.csv")
