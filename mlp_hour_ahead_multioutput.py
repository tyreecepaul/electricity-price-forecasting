"""Compare a 12-output MLP: python mlp_hour_ahead_multioutput.py --device xpu.

Uses the horizon-conditioned experiment's origins, features, pooled target
scaling and MLP settings, without its horizon feature or repeated inputs.
The original experiment and shared single-output pipeline are unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from electricity_price_forecasting_model.final_model.mlp import MLPModel
from main import EXPERIMENT, MODEL_REGISTRY
from mlp_hour_ahead import HORIZONS, expand_horizons
from pipeline.contracts import score_forecast
from pipeline.experiment_pipeline import prepare_experiment
from pipeline.preprocessing import (
    NumericPreprocessor, TargetTransformer, add_targets, essential_feature_columns,
    forecast_reliability_columns, suggested_scaling_groups,
)
from pipeline.training import TorchEarlyStoppingTrainer


def predict_changes(model, inputs: torch.Tensor) -> np.ndarray:
    """Return (origins, 12) price changes in original $/MWh units."""
    model.network.eval()
    with torch.inference_mode():
        values = np.concatenate([
            model.network(batch.to(model.device)).cpu().numpy()
            for batch in inputs.split(model.prediction_batch_size)
        ])
    if values.shape != (len(inputs), len(HORIZONS)):
        raise ValueError("Expected twelve outputs per forecast origin")
    predictions = model.target_transformer.inverse_transform(values)
    if not np.isfinite(predictions).all():
        raise ValueError("MLP produced non-finite forecasts")
    return predictions


def fit_mlp(frame, features, target_columns, options, validation=None):
    """Experiment-local vector-target fitting with the shared trainer and scalers."""
    model = MLPModel(**options)
    model.feature_columns = list(features)
    fitting = model._numeric_frame(frame[features])
    model.feature_medians = fitting.median().fillna(0.0)
    groups = suggested_scaling_groups(features)
    model.preprocessor = NumericPreprocessor(
        asinh_columns=tuple(groups["asinh_then_standardise"]),
        standardise_columns=tuple(groups["standardise_only"]),
    ).fit(fitting.fillna(model.feature_medians))
    changes = frame[target_columns].to_numpy(dtype=float) - frame["rrp"].to_numpy()[:, None]
    # One common target scale across all horizons, just like the pooled rows
    # in the horizon-conditioned run. Fit it on training labels only.
    model.target_transformer = TargetTransformer().fit(changes.ravel())
    inputs = model._inputs(frame[features])
    targets = torch.tensor(model.target_transformer.transform(changes), dtype=torch.float32)
    model._set_random_seed()
    model.network = model._build_network(len(features))
    head = model.network.network[-1]
    model.network.network[-1] = nn.Linear(head.in_features, len(HORIZONS))
    model.network.to(model.device)
    loader = DataLoader(
        TensorDataset(inputs, targets), batch_size=model.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(model.random_seed),
    )
    optimizer = torch.optim.Adam(
        model.network.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay,
    )
    criterion = model._build_criterion()
    epoch = 0

    def train_epoch():
        nonlocal epoch
        epoch += 1
        model.network.train()
        total = 0.0
        with tqdm(loader, desc=f"12-output MLP epoch {epoch}/{model.max_epochs}", leave=False) as batches:
            for xb, yb in batches:
                optimizer.zero_grad(set_to_none=True)
                prediction = model.network(xb.to(model.device))
                loss = criterion(prediction, yb.to(model.device))
                loss.backward()
                if model.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.network.parameters(), model.gradient_clip_norm)
                optimizer.step()
                total += loss.item() * len(xb)
        return total / len(inputs)

    callback = None
    if validation is not None:
        validation_inputs = model._inputs(validation[features])
        actual = validation[target_columns].to_numpy(dtype=float) - validation["rrp"].to_numpy()[:, None]

        def callback():
            predicted = predict_changes(model, validation_inputs)
            # Pool all origin/horizon pairs exactly as the scalar experiment does.
            return score_forecast(actual.ravel(), predicted.ravel(), model.selection_metric)

    result = TorchEarlyStoppingTrainer(
        max_epochs=model.max_epochs, patience=model.patience, min_delta=model.min_delta,
    ).fit(network=model.network, train_epoch=train_epoch, validation_score=callback)
    model.best_epoch = result.best_epoch
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "reports" / "runs" / "mlp-hour-ahead-multioutput",
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
    print(f"12-output MLP training/prediction device: {device}")
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

    features = prepared.features
    grid = next(grid for factory, grid in MODEL_REGISTRY if factory is MLPModel)
    options = {name: values[0] for name, values in grid.items()}
    options.update(
        device=device, max_epochs=args.epochs, random_seed=config.random_seed,
        selection_metric=config.selection_metric,
    )
    print(f"Horizons: {HORIZONS} minutes; one input row and twelve outputs per origin.")
    print(f"Origins: fit={len(fit):,}, validation={len(validation):,}, test={len(testing):,}")
    print(f"Fit targets end before {validation_start}; validation/refit targets end before {test_start}.")
    started = perf_counter()
    selected = fit_mlp(fit, features, target_columns, options, validation=validation)
    options["max_epochs"] = selected.best_epoch or args.epochs
    del selected
    print(f"Refitting on all pre-test origins for {options['max_epochs']} epoch(s).")
    model = fit_mlp(training, features, target_columns, options)
    print(f"Selection and refit took {perf_counter() - started:.1f} seconds.")
    changes = predict_changes(model, model._inputs(testing[features]))

    # Only output metadata is expanded. The network sees each origin once.
    # expand_horizons orders rows by horizon, then origin, hence transpose here.
    rows = expand_horizons(testing, [])
    forecasts = rows[["region", "settlement_date", "target_time", "horizon_minutes", "rrp", "actual_price"]].copy()
    forecasts["predicted_price"] = forecasts["rrp"] + changes.T.reshape(-1)
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
    ax.plot(minutes, np.r_[current, example["predicted_price"]], marker="o", label="12-output MLP forecast")
    ax.axhline(current, linestyle="--", color="gray", label="Persistence")
    ax.set(xlabel="Minutes ahead", ylabel="Price ($/MWh)", title=f"{config.region}: forecast issued {origin}", xticks=minutes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "forecast_example.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved forecasts, per-horizon scores and example curve to {args.output_dir}")


if __name__ == "__main__":
    main()
