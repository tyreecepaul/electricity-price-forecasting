"""Local horizon-conditioned MLP experiment: python mlp_hour_ahead.py.

One MLP predicts changes from the current price at 5, 10, ..., 60 minutes.
Uses main.py's region, features and first MLP configuration. It does not run
the other models or GradientSHAP. Results go to reports/runs/mlp-hour-ahead/.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from electricity_price_forecasting_model.final_model.mlp import MLPModel
from main import EXPERIMENT, MODEL_REGISTRY
from pipeline.contracts import score_forecast
from pipeline.experiment_pipeline import prepare_experiment
from pipeline.preprocessing import add_targets, essential_feature_columns, forecast_reliability_columns


HORIZONS = tuple(range(5, 61, 5))


def expand_horizons(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Reuse each origin's inputs with a different lead time and price-change label."""
    pieces = []
    for horizon in HORIZONS:
        columns = list(dict.fromkeys([*features, "region", "settlement_date", "rrp"]))
        rows = frame[columns].copy()
        rows["horizon_minutes"] = horizon
        rows["target_time"] = rows["settlement_date"] + pd.Timedelta(minutes=horizon)
        rows["actual_price"] = frame[f"target_rrp_h{horizon:02d}"]
        rows["target"] = rows["actual_price"] - rows["rrp"]
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--refresh", action="store_true", help="Rebuild the prepared-data cache.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "reports" / "runs" / "mlp-hour-ahead",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if EXPERIMENT.feature_config.interval_minutes != 5:
        parser.error("This experiment requires five-minute input data")

    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else
            "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        )
    print(f"MLP hour-ahead training/prediction device: {device}")
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
    allowed_features = set(essential_feature_columns(prepared.train_timeline, config.feature_config))
    allowed_features.update(forecast_reliability_columns(prepared.train_timeline))
    unexpected = set(prepared.features) - allowed_features
    if unexpected:
        raise ValueError(f"Inputs outside the shared causal feature set: {sorted(unexpected)}")

    # Build labels by exact timestamps on each split's full timeline, not on
    # filtered rows. No future actual is included in the input feature list.
    target_config = replace(config.feature_config, target_horizons_minutes=HORIZONS)
    target_columns = [f"target_rrp_h{h:02d}" for h in HORIZONS]
    training = add_targets(prepared.train_timeline, target_config)
    testing = add_targets(prepared.test_timeline, target_config)
    training = training.replace([np.inf, -np.inf], np.nan).dropna(subset=["rrp", *target_columns])
    testing = testing.replace([np.inf, -np.inf], np.nan).dropna(subset=["rrp", *target_columns])

    # Split by forecast origin before repeating it for each horizon. The
    # existing 60-minute experiment determines the embargo and cutoff.
    training = training.loc[training["settlement_date"] + pd.Timedelta(minutes=60) < test_start]
    validation_start = experiment.validation_df["settlement_date"].min()
    fit = training.loc[training["settlement_date"] + pd.Timedelta(minutes=60) < validation_start]
    validation = training.loc[training["settlement_date"] >= validation_start]
    if any(frame.empty for frame in (fit, validation, training, testing)):
        raise ValueError("Not enough complete one-hour targets for training, validation and test")

    features = [*prepared.features, "horizon_minutes"]
    grid = next(grid for factory, grid in MODEL_REGISTRY if factory is MLPModel)
    options = {name: values[0] for name, values in grid.items()}
    options.update(
        device=device, max_epochs=args.epochs, random_seed=config.random_seed,
        selection_metric=config.selection_metric,
    )
    print(f"Horizons: {HORIZONS} minutes")
    print(f"Origins: fit={len(fit):,}, validation={len(validation):,}, test={len(testing):,}")
    print(f"Fit targets end before {validation_start}; validation/refit targets end before {test_start}.")
    print("Each origin supplies 12 training examples; all changes are relative to its current price.")

    fit_rows = expand_horizons(fit, prepared.features)
    validation_rows = expand_horizons(validation, prepared.features)
    selected = MLPModel(**options)
    selected.train_with_validation(
        fit_rows[features], fit_rows["target"],
        validation_rows[features], validation_rows["target"],
    )
    options["max_epochs"] = selected.best_epoch or args.epochs
    print(f"Refitting on all pre-test origins for {options['max_epochs']} epoch(s).")
    del fit_rows, validation_rows, selected

    train_rows = expand_horizons(training, prepared.features)
    model = MLPModel(**options)
    model.train(train_rows[features], train_rows["target"])
    del train_rows

    # Predict all twelve lead times at each test origin without feeding any
    # intermediate prediction or future observation back into the model.
    rows = expand_horizons(testing, prepared.features)
    changes = model.predict(rows[features])
    forecasts = rows[["region", "settlement_date", "target_time", "horizon_minutes", "rrp", "actual_price"]].copy()
    forecasts["predicted_price"] = forecasts["rrp"] + changes
    if not np.isfinite(forecasts["predicted_price"]).all():
        raise ValueError("MLP produced non-finite forecasts")

    metrics = pd.DataFrame([
        {
            "horizon_minutes": horizon,
            "mae": score_forecast(group["actual_price"], group["predicted_price"], "mae"),
            "rmse": score_forecast(group["actual_price"], group["predicted_price"], "rmse"),
            "persistence_mae": score_forecast(group["actual_price"], group["rrp"], "mae"),
        }
        for horizon, group in forecasts.groupby("horizon_minutes", sort=True)
    ]).set_index("horizon_minutes")
    print("\nHeld-out price errors ($/MWh; persistence predicts the current price):")
    print(metrics.to_string(float_format=lambda value: f"{value:.3f}"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_horizon.csv")
    forecasts.pivot(
        index=["region", "settlement_date"], columns="horizon_minutes", values="predicted_price",
    ).rename(columns=lambda horizon: f"price_in_{horizon:02d}min").to_csv(args.output_dir / "forecast_curves.csv")

    origin = forecasts["settlement_date"].min()
    example = forecasts.loc[forecasts["settlement_date"] == origin].sort_values("horizon_minutes")
    minutes = np.r_[0, example["horizon_minutes"].to_numpy()]
    current = example["rrp"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(minutes, np.r_[current, example["actual_price"]], marker="o", label="Actual")
    ax.plot(minutes, np.r_[current, example["predicted_price"]], marker="o", label="MLP forecast")
    ax.axhline(current, linestyle="--", color="gray", label="Persistence")
    ax.set(xlabel="Minutes ahead", ylabel="Price ($/MWh)", title=f"{config.region}: forecast issued {origin}", xticks=minutes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "forecast_example.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved forecasts, per-horizon scores and example curve to {args.output_dir}")


if __name__ == "__main__":
    main()
