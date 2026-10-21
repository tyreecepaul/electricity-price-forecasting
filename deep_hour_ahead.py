"""Run shared 12-output MLP/GRU/LSTM forecasting: python deep_hour_ahead.py.

Uses main.py's first configuration per model, with validation epoch selection
and a fresh pre-test refit. GradientSHAP explains a selected price-change output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from main import EXPERIMENT, MODEL_REGISTRY
from pipeline.contracts import score_forecast
from pipeline.experiment_pipeline import prepare_experiment
from pipeline.preprocessing import add_targets, essential_feature_columns, forecast_reliability_columns


HORIZONS = tuple(range(5, 61, 5))


def prepare_hour(config, refresh=False):
    """Keep full feature timelines; mask unavailable or embargoed 12-value labels."""
    experiment = prepare_experiment(config, force_refresh=refresh)
    prepared = experiment.prepared
    if prepared.train_timeline is None or prepared.test_timeline is None:
        raise ValueError("Complete timelines are required; rerun with --refresh")
    for frame in (prepared.train_timeline, prepared.test_timeline):
        if frame.duplicated(["region", "settlement_date"]).any():
            raise ValueError("Duplicate region/timestamp rows make target alignment ambiguous")
    test_start = prepared.test_timeline["settlement_date"].min()
    if prepared.train_timeline["settlement_date"].max() >= test_start:
        raise ValueError("Training and held-out test timelines overlap")
    allowed = set(essential_feature_columns(prepared.train_timeline, config.feature_config))
    allowed.update(forecast_reliability_columns(prepared.train_timeline))
    if unexpected := set(prepared.features) - allowed:
        raise ValueError(f"Inputs outside the shared causal feature set: {sorted(unexpected)}")

    target_config = replace(config.feature_config, target_horizons_minutes=HORIZONS)
    columns = [f"target_rrp_h{h:02d}" for h in HORIZONS]
    train = add_targets(prepared.train_timeline, target_config).set_index(["region", "settlement_date"])
    test = add_targets(prepared.test_timeline, target_config).set_index(["region", "settlement_date"])
    y_train = train[columns].sub(train["rrp"], axis=0).replace([np.inf, -np.inf], np.nan)
    y_test = test[columns].sub(test["rrp"], axis=0).replace([np.inf, -np.inf], np.nan)
    y_train.columns = y_test.columns = [f"change_h{h:02d}" for h in HORIZONS]
    train_times = train.index.get_level_values("settlement_date")
    y_train.loc[train_times + pd.Timedelta(minutes=60) >= test_start, :] = np.nan
    validation_start = experiment.validation_df["settlement_date"].min()
    fit_rows = train_times < validation_start
    fit_y = y_train.loc[fit_rows].copy()
    fit_times = fit_y.index.get_level_values("settlement_date")
    fit_y.loc[fit_times + pd.Timedelta(minutes=60) >= validation_start, :] = np.nan
    val_y = y_train.loc[~fit_rows].copy()
    if any(not labels.notna().all(axis=1).any() for labels in (fit_y, val_y, y_train, y_test)):
        raise ValueError("Not enough complete one-hour targets for all splits")
    x_train, x_test = train[prepared.features], test[prepared.features]
    print(f"Fit targets end before {validation_start}; validation/refit targets end before {test_start}.")
    return x_train, y_train, x_test, y_test, train, test, fit_rows, fit_y, val_y


def save_forecasts(folder, frame, labels, changes, region):
    valid = labels.notna().all(axis=1).to_numpy()
    current = frame.loc[valid, "rrp"].to_numpy()
    predicted = current[:, None] + changes[valid]
    actual = current[:, None] + labels.loc[valid].to_numpy()
    if not np.isfinite(predicted).all():
        raise ValueError("Model produced non-finite forecasts")
    origins = frame.loc[valid].index
    rows, scores = [], []
    for i, horizon in enumerate(HORIZONS):
        part = origins.to_frame(index=False)
        part["target_time"] = part["settlement_date"] + pd.Timedelta(minutes=horizon)
        part["horizon_minutes"] = horizon
        part["rrp"], part["actual_price"], part["predicted_price"] = current, actual[:, i], predicted[:, i]
        rows.append(part)
        scores.append({
            "horizon_minutes": horizon,
            "mae": score_forecast(actual[:, i], predicted[:, i], "mae"),
            "rmse": score_forecast(actual[:, i], predicted[:, i], "rmse"),
            "persistence_mae": score_forecast(actual[:, i], current, "mae"),
        })
    forecasts = pd.concat(rows, ignore_index=True)
    forecasts.to_csv(folder / "predictions.csv", index=False)
    forecasts.pivot(
        index=["region", "settlement_date"], columns="horizon_minutes", values="predicted_price",
    ).rename(columns=lambda h: f"price_in_{h:02d}min").to_csv(folder / "forecast_curves.csv")
    metrics = pd.DataFrame(scores).set_index("horizon_minutes")
    metrics.to_csv(folder / "metrics_by_horizon.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot((0, *HORIZONS), np.r_[current[0], actual[0]], marker="o", label="Actual")
    ax.plot((0, *HORIZONS), np.r_[current[0], predicted[0]], marker="o", label="12-output forecast")
    ax.axhline(current[0], linestyle="--", color="gray", label="Persistence")
    ax.set(xlabel="Minutes ahead", ylabel="Price ($/MWh)", title=f"{region}: forecast issued {origins[0][1]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "forecast_example.png", dpi=150)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["mlp", "gru", "lstm"], default=["mlp", "gru", "lstm"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--epochs", type=int, help="Override each model's configured maximum epochs.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--shap-horizons", nargs="+", type=int, choices=HORIZONS, default=[5, 30, 60])
    parser.add_argument("--shap-samples", type=int, default=128)
    parser.add_argument("--shap-draws", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reports" / "runs" / "deep-hour-ahead")
    args = parser.parse_args()
    if (args.epochs is not None and args.epochs < 1) or min(args.shap_samples, args.shap_draws) < 1:
        parser.error("Epoch and attribution counts must be positive")
    if EXPERIMENT.feature_config.interval_minutes != 5:
        parser.error("This experiment requires five-minute input data")
    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else
            "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        )
    config = replace(EXPERIMENT, horizon_minutes=60, target_type="change", walk_forward_folds=1)
    x_train, y_train, x_test, y_test, train, test, fit_rows, fit_y, val_y = prepare_hour(config, args.refresh)
    print(f"Training/prediction device: {device}; outputs: {HORIZONS} minutes")
    print("One shared training-only target scaler; complete timelines retained for sequence history.")
    summaries = []
    for factory, grid in MODEL_REGISTRY:
        name = factory.model_name.lower()
        if name not in args.models:
            continue
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        options = {key: values[0] for key, values in grid.items()}
        options.update(device=device, random_seed=config.random_seed, selection_metric=config.selection_metric)
        if args.epochs is not None:
            options["max_epochs"] = args.epochs
        print(f"\n=== {factory.model_name}: 12 outputs ===", flush=True)
        started = perf_counter()
        selected = factory(**options)
        result = selected.train_with_validation(x_train.loc[fit_rows], fit_y, x_train.loc[~fit_rows], val_y)
        options["max_epochs"] = result.best_epoch
        del selected
        print(f"Refitting on pre-test data for {result.best_epoch} epoch(s).", flush=True)
        model = factory(**options)
        model.train(x_train, y_train)
        fit_seconds = perf_counter() - started
        changes = model.predict(x_test, history=x_train) if model.requires_sequence else model.predict(x_test)
        metrics = save_forecasts(folder, test, y_test, changes, config.region)
        print("Held-out price errors ($/MWh):")
        print(metrics.to_string(float_format=lambda value: f"{value:.3f}"))
        summaries.append(metrics.reset_index().assign(model=name))
        (folder / "model.json").write_text(json.dumps({
            "experiment": asdict(config), "options": options,
            "features": model.feature_columns, "target_columns": model.target_columns,
            "validation_score": result.best_validation_score, "fit_seconds": fit_seconds,
        }, indent=2, default=str), encoding="utf-8")

        if not args.skip_shap:
            positions = np.flatnonzero(y_test.notna().all(axis=1).to_numpy())
            for horizon in dict.fromkeys(args.shap_horizons):
                print(f"GradientSHAP: +{horizon}min price change, samples<={args.shap_samples}, draws={args.shap_draws}", flush=True)
                attribution = model.gradient_shap(
                    x_test, x_train, history=x_train if model.requires_sequence else None,
                    positions=positions, target=HORIZONS.index(horizon), device=device,
                    sample_size=args.shap_samples, baseline_size=64, n_samples=args.shap_draws,
                )
                attribution.assign(residual=attribution.attrs["completeness_residual"]).to_csv(folder / f"gradient_shap_h{horizon:02d}.csv")
                attribution.attrs["absolute_contributions"].mean().sort_values(ascending=False).rename("mean_absolute_contribution").to_csv(folder / f"gradient_shap_importance_h{horizon:02d}.csv")
                if model.requires_sequence:
                    attribution.attrs["lag_attributions"].to_csv(folder / f"gradient_shap_by_lag_h{horizon:02d}.csv")
        pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "summary.csv", index=False)
        print(f"Saved {factory.model_name} results to {folder}")


if __name__ == "__main__":
    main()
