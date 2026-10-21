"""Compare 12 horizon-specific XGBoost models with one horizon-conditioned model.

Run: python xgboost_hour_ahead_compare.py
Uses the first XGBoost configuration from main.py without hyperparameter search.
Both approaches use the same complete origins, raw price-change targets and
60-minute embargo. Validation is scored before refitting on all pre-test data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from electricity_price_forecasting_model.baseline_models.xgboost import XGBoostModel
from main import EXPERIMENT, MODEL_REGISTRY
from mlp_hour_ahead import HORIZONS, expand_horizons
from pipeline.contracts import score_forecast
from pipeline.experiment_pipeline import prepare_experiment
from pipeline.preprocessing import add_targets, essential_feature_columns, forecast_reliability_columns


def fit_and_predict(training, evaluation, features, options, stage):
    """Use identical rows for both strategies, keeping labels outside the inputs."""
    results, timings = {}, []
    metadata = expand_horizons(evaluation, [])[
        ["region", "settlement_date", "target_time", "horizon_minutes", "rrp", "actual_price"]
    ]
    train_x = training[features].astype(np.float32)
    eval_x = evaluation[features].astype(np.float32)
    changes, fit_seconds, prediction_seconds = [], 0.0, 0.0
    for horizon in HORIZONS:
        print(f"{stage}: fitting horizon-specific XGBoost for +{horizon} minutes...", flush=True)
        model = XGBoostModel(**options)
        target = training[f"target_rrp_h{horizon:02d}"] - training["rrp"]
        started = perf_counter()
        model.train(train_x, target)
        fit_seconds += perf_counter() - started
        started = perf_counter()
        changes.append(model.predict(eval_x))
        prediction_seconds += perf_counter() - started
    direct = metadata.copy()
    direct["predicted_price"] = direct["rrp"] + np.concatenate(changes)
    results["horizon_specific"] = direct
    timings.append({
        "stage": stage, "strategy": "horizon_specific", "models": 12,
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
    })
    del model, changes

    print(f"{stage}: fitting one horizon-conditioned XGBoost on 12 lead times...", flush=True)
    # Only input features are repeated. Target columns never enter this matrix.
    pooled_x = pd.concat(
        [train_x.assign(horizon_minutes=np.float32(h)) for h in HORIZONS], ignore_index=True,
    )
    pooled_y = pd.concat(
        [training[f"target_rrp_h{h:02d}"] - training["rrp"] for h in HORIZONS], ignore_index=True,
    )
    model = XGBoostModel(**options)
    started = perf_counter()
    model.train(pooled_x, pooled_y)
    fit_seconds = perf_counter() - started
    del pooled_x, pooled_y
    started = perf_counter()
    changes = [model.predict(eval_x.assign(horizon_minutes=np.float32(h))) for h in HORIZONS]
    prediction_seconds = perf_counter() - started
    pooled = metadata.copy()
    pooled["predicted_price"] = pooled["rrp"] + np.concatenate(changes)
    results["horizon_conditioned"] = pooled
    timings.append({
        "stage": stage, "strategy": "horizon_conditioned", "models": 1,
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
    })
    if any(not np.isfinite(frame["predicted_price"]).all() for frame in results.values()):
        raise ValueError("XGBoost produced non-finite forecasts")
    return results, timings


def comparison_metrics(results):
    rows = []
    for horizon in HORIZONS:
        row = {"horizon_minutes": horizon}
        for strategy, frame in results.items():
            group = frame.loc[frame["horizon_minutes"] == horizon]
            for metric in ("mae", "rmse"):
                row[f"{strategy}_{metric}"] = score_forecast(
                    group["actual_price"], group["predicted_price"], metric,
                )
        row["persistence_mae"] = score_forecast(group["actual_price"], group["rrp"], "mae")
        # Positive means the 12 horizon-specific models have lower MAE.
        row["conditioned_minus_specific_mae"] = row["horizon_conditioned_mae"] - row["horizon_specific_mae"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon_minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, help="Boosting rounds per model; defaults to main.py's setting.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU threads; -1 uses all available cores.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "reports" / "runs" / "xgboost-hour-ahead-compare",
    )
    args = parser.parse_args()
    if args.trees is not None and args.trees < 1:
        parser.error("--trees must be positive")
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")
    if EXPERIMENT.feature_config.interval_minutes != 5:
        parser.error("This experiment requires five-minute input data")

    print("XGBoost training/prediction device: cpu", flush=True)
    config = replace(EXPERIMENT, horizon_minutes=60, target_type="change", walk_forward_folds=1)
    experiment = prepare_experiment(config, force_refresh=args.refresh)
    prepared = experiment.prepared
    if prepared.train_timeline is None or prepared.test_timeline is None:
        raise ValueError("Complete price timelines are required; rerun with --refresh")
    for timeline in (prepared.train_timeline, prepared.test_timeline):
        if timeline.duplicated(["region", "settlement_date"]).any():
            raise ValueError("Duplicate region/timestamp rows make target alignment ambiguous")
    test_start = prepared.test_timeline["settlement_date"].min()
    if prepared.train_timeline["settlement_date"].max() >= test_start:
        raise ValueError("Training and held-out test timelines overlap")
    allowed = set(essential_feature_columns(prepared.train_timeline, config.feature_config))
    allowed.update(forecast_reliability_columns(prepared.train_timeline))
    unexpected = set(prepared.features) - allowed
    if unexpected:
        raise ValueError(f"Inputs outside the shared causal feature set: {sorted(unexpected)}")

    target_config = replace(config.feature_config, target_horizons_minutes=HORIZONS)
    target_columns = [f"target_rrp_h{h:02d}" for h in HORIZONS]
    training = add_targets(prepared.train_timeline, target_config)
    testing = add_targets(prepared.test_timeline, target_config)
    training = training.replace([np.inf, -np.inf], np.nan).dropna(subset=["rrp", *target_columns])
    testing = testing.replace([np.inf, -np.inf], np.nan).dropna(subset=["rrp", *target_columns])
    training = training.loc[training["settlement_date"] + pd.Timedelta(minutes=60) < test_start]
    validation_start = experiment.validation_df["settlement_date"].min()
    fit = training.loc[training["settlement_date"] + pd.Timedelta(minutes=60) < validation_start]
    validation = training.loc[training["settlement_date"] >= validation_start]
    if any(frame.empty for frame in (fit, validation, training, testing)):
        raise ValueError("Not enough complete one-hour targets for training, validation and test")

    grid = next(grid for factory, grid in MODEL_REGISTRY if factory is XGBoostModel)
    options = {name: values[0] for name, values in grid.items()}
    options.update(random_seed=config.random_seed, n_jobs=args.n_jobs)
    if args.trees is not None:
        options["max_estimators"] = args.trees
    print(f"Fixed settings for both strategies: {options}")
    print("No hyperparameter search; raw $/MWh changes, without scaling, for both strategies.")
    print(f"Origins: fit={len(fit):,}, validation={len(validation):,}, test={len(testing):,}")
    print(f"Fit targets end before {validation_start}; validation/refit targets end before {test_start}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_timings = []
    for stage, training_frame, evaluation_frame in (
        ("validation", fit, validation), ("test", training, testing),
    ):
        results, timings = fit_and_predict(
            training_frame, evaluation_frame, prepared.features, options, stage,
        )
        all_timings.extend(timings)
        metrics = comparison_metrics(results)
        print(f"\n{stage.upper()} price errors ($/MWh):")
        print(metrics.to_string(float_format=lambda value: f"{value:.3f}"))
        metrics.to_csv(args.output_dir / f"{stage}_metrics_by_horizon.csv")
        pd.DataFrame(all_timings).to_csv(args.output_dir / "timings.csv", index=False)
        if stage == "test":
            for strategy, forecasts in results.items():
                folder = args.output_dir / strategy
                folder.mkdir(exist_ok=True)
                forecasts.to_csv(folder / "predictions.csv", index=False)
                forecasts.pivot(
                    index=["region", "settlement_date"], columns="horizon_minutes", values="predicted_price",
                ).rename(columns=lambda h: f"price_in_{h:02d}min").to_csv(folder / "forecast_curves.csv")
            origin = results["horizon_specific"]["settlement_date"].min()
            example = results["horizon_specific"].loc[lambda f: f["settlement_date"] == origin].sort_values("horizon_minutes")
            minutes = np.r_[0, example["horizon_minutes"].to_numpy()]
            current = example["rrp"].iloc[0]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(minutes, np.r_[current, example["actual_price"]], marker="o", label="Actual")
            for strategy, forecasts in results.items():
                curve = forecasts.loc[forecasts["settlement_date"] == origin].sort_values("horizon_minutes")
                ax.plot(minutes, np.r_[current, curve["predicted_price"]], marker="o", label=strategy.replace("_", " "))
            ax.axhline(current, linestyle="--", color="gray", label="Persistence")
            ax.set(xlabel="Minutes ahead", ylabel="Price ($/MWh)", title=f"{config.region}: forecast issued {origin}", xticks=minutes)
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.output_dir / "forecast_example.png", dpi=150)
            plt.close(fig)
        del results
    print("\nFit/prediction timings (seconds; horizon-specific totals cover all 12 models):")
    print(pd.DataFrame(all_timings).to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(f"\nSaved comparison scores, timings and forecast curves to {args.output_dir}")


if __name__ == "__main__":
    main()
