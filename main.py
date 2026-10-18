import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from electricity_price_forecasting_model.final_model.transformer import TransformerModel
from electricity_price_forecasting_model.final_model.gru import GRUModel
from electricity_price_forecasting_model.final_model.lstm import LSTMModel
from electricity_price_forecasting_model.final_model.mlp import MLPModel
from electricity_price_forecasting_model.final_model.tcn import TCNModel
from electricity_price_forecasting_model.baseline_models.arima import ARIMAModel
from electricity_price_forecasting_model.baseline_models.xgboost import XGBoostModel
from pipeline.artifacts import create_run_directory, save_model_run, save_run_summary
from pipeline.evaluation import ModelResult, evaluate, print_evaluation_summary
from pipeline.experiment_pipeline import (
    ExperimentConfig,
    ExperimentData,
    prepare_experiment,
    print_experiment_summary,
    reconstruct_price_forecast,
)
from pipeline.model import Model
from pipeline.model_data import final_training_data, predict_test
from pipeline.contracts import indexed_features
from pipeline.neural import TorchForecastModel
from pipeline.tuning import SearchResult, search_hyperparameters

EXPERIMENT: ExperimentConfig = ExperimentConfig(
    region="VIC1",
    horizon_minutes=30,
    target_type="change",
    model_features="all",
    validation_fraction=0.20,
)

# Each entry pairs a model constructor with candidate hyperparameter values.
# Search evaluates their Cartesian product; one-element lists fix a setting.
MODEL_REGISTRY: list[tuple[type[Model], dict[str, Sequence]]] = [
    (
        XGBoostModel,
        {
            "max_depth": [6],
            "learning_rate": [0.1],
            "min_child_weight": [4],
            "max_estimators": [1000],
        },
    ),
    (
        ARIMAModel,
        {
            # ARIMA forecasts prices itself, so it must know the experiment's target.
            "horizon_minutes": [EXPERIMENT.horizon_minutes],
            "target_type": [EXPERIMENT.target_type],
            # Chosen on validation MAE for VIC1 at 30 minutes (test MAE 13.40).
            "p": [2],
            "d": [1],
            "q": [0],
            "transform": ["asinh_mad"],
            "daily_harmonics": [0],
            "fit_window_days": [7],
        },
    ),
    (
        MLPModel,
        {
            "hidden_sizes": [(128, 64, 32)],
            "dropout": [(0.2, 0.15, 0.0)],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [256],
            "max_epochs": [30],
            "patience": [3],
            "min_delta": [0.001],
            "loss": ["huber"],
            "huber_delta": [1.0],
        },
    ),
    (
        GRUModel,
        {
            "sequence_length": [12],
            "hidden_size": [16],
            "num_layers": [1],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [10],
            "patience": [2],
            "min_delta": [0.001],
            "loss": ["huber"],
            "huber_delta": [1.0],
        },
    ),
    (
        TransformerModel,
        {
            "sequence_length": [72], #72
            "d_model": [64],# 32
            "nhead": [4], #8
            "num_layers": [3], #6
            "dim_feedforward": [128],#64
            "dropout": [0.0], #0.1
            "learning_rate": [0.001], #test weight decay next
            "weight_decay": [0.0001],
            "batch_size": [128], #64
            "prediction_batch_size": [256],
            "max_epochs": [50],
            "patience": [10],
            "min_delta": [0.001],
            "loss": ["huber"],
            "huber_delta": [0.5],
        },
    ),
    (
        LSTMModel,
        {
            "sequence_length": [12],
            "hidden_size": [64],
            "num_layers": [2],
            "dropout": [0.25],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [20],
            "patience": [3],
            "min_delta": [0.001],
            "loss": ["huber"],
            "huber_delta": [1.0],
            "gradient_clip_norm": [1.0],
        },
    ),
    (
        TCNModel,
        {
            "sequence_length": [12],
            "channels": [16],
            "kernel_size": [3],
            "num_blocks": [2],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [10],
            "patience": [2],
            "min_delta": [0.001],
            "loss": ["huber"],
            "huber_delta": [1.0],
            "device": [None],
        },    
    )
]

def model_cli_name(model_factory: type[Model]) -> str:
    """Return the stable command-line name for a registered model."""
    return model_factory.model_name.lower().replace(" ", "-")


def select_model_registry(
    selected_models: Sequence[str] | None,
) -> list[tuple[type[Model], dict[str, Sequence]]]:
    """Select registry entries while preserving their configured order."""
    if selected_models is None:
        return MODEL_REGISTRY

    requested = {name.lower() for name in selected_models}
    available = {model_cli_name(factory) for factory, _ in MODEL_REGISTRY}
    unknown = requested.difference(available)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown models: {names}")
    return [
        entry
        for entry in MODEL_REGISTRY
        if model_cli_name(entry[0]) in requested
    ]


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate electricity-price forecasting models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str.lower,
        choices=sorted(model_cli_name(factory) for factory, _ in MODEL_REGISTRY),
        help="Models to run. Defaults to all registered models.",
    )
    parser.add_argument("--folds", type=int, default=EXPERIMENT.walk_forward_folds)
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "sse", "median_ae"],
        default=EXPERIMENT.selection_metric,
    )
    parser.add_argument("--seed", type=int, default=EXPERIMENT.random_seed)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "runs",
    )
    parser.add_argument("--save", action="store_true", help="Save models and result artifacts; disabled by default.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild prepared data from ingestion/cache",
    )
    return parser.parse_args(arguments)


def sequence_attribution_summary(
    lag_attributions: pd.DataFrame, interval_minutes: float,
) -> pd.DataFrame:
    """Summarise contributions by sequence position, not engineered feature lag."""
    lags = sorted({int(column.split("_", 2)[1]) for column in lag_attributions.columns})
    rows = []
    for lag in lags:
        values = lag_attributions.loc[:, lag_attributions.columns.str.startswith(f"lag_{lag}_")]
        rows.append({
            "sequence_lag": lag,
            "minutes_before_prediction": lag * interval_minutes,
            "mean_absolute_contribution": values.abs().sum(axis=1).mean(),
            "mean_signed_contribution": values.sum(axis=1).mean(),
        })
    summary = pd.DataFrame(rows).set_index("sequence_lag")
    total = summary["mean_absolute_contribution"].sum()
    summary["absolute_share_percent"] = (
        100 * summary["mean_absolute_contribution"] / total if total > 0 else 0.0
    )
    return summary


def main(
    selected_models: Sequence[str] | None = None,
    *,
    config=None,
    output_directory=None,
    save_results=False,
    force_refresh=False,
) -> None:
    """Prepare shared data, select/refit each model, then score the held-out test.

    Temporal representation choices live in model_data; model-specific network
    and fitting details stay behind the common model interface.
    """
    config = config or EXPERIMENT
    experiment: ExperimentData = (
        prepare_experiment(config, force_refresh=True)
        if force_refresh
        else prepare_experiment(config)
    )
    print_experiment_summary(experiment)

    prepared = experiment.prepared
    run_directory = (
        create_run_directory(
            config,
            Path(output_directory)
            if output_directory is not None
            else Path(__file__).resolve().parent / "reports" / "runs",
        )
        if save_results
        else None
    )

    y_test_price: np.ndarray = experiment.test_df["target_rrp"].to_numpy()

    # Fit each model and find optimal hyperparameters
    results: list[ModelResult] = []

    for model_factory, hyperparam_grid in select_model_registry(selected_models):
        print(f"=== {model_factory.model_name} ===")

        search_result: SearchResult = search_hyperparameters(
            model_factory, hyperparam_grid, experiment
        )

        # Start a fresh fit on all pre-test data, using the selected epoch budget.
        # Reusing the search model would leave out its validation-period training data.
        model: Model = model_factory(**search_result.best_config)
        x_train, y_train = final_training_data(model, prepared)
        model.train(x_train, y_train)

        y_pred: np.ndarray = predict_test(model, prepared)
        # Model outputs may be price changes or AEMO residuals. Convert them to
        # actual RRP units before comparing with prices and common baselines.
        predicted_price: np.ndarray = reconstruct_price_forecast(
            experiment.test_df, y_pred, experiment.config
        )

        result = evaluate(model, y_test_price, predicted_price)

        results.append(result)
        if run_directory is not None:
            save_model_run(
                run_directory,
                model,
                search_result,
                experiment.test_df,
                predicted_price,
                result,
            )
        if isinstance(model, TorchForecastModel):
            sample_size, baseline_size, n_samples = 1024, 256, 2048
            print(f"GradientShap: target={config.target_type}, samples<={sample_size}, "
                  f"references<={baseline_size}, draws={n_samples}, seed={model.random_seed}")
            frame = prepared.test_timeline if model.requires_sequence else prepared.test_df
            inputs = indexed_features(frame, prepared.features)
            history = indexed_features(prepared.train_timeline, prepared.features) if model.requires_sequence else None
            positions = inputs.index.get_indexer(indexed_features(prepared.test_df, prepared.features).index)
            attribution = model.gradient_shap(
                inputs, x_train, history=history, positions=positions,
                sample_size=sample_size, baseline_size=baseline_size, n_samples=n_samples,
            )
            print(f"Observations={len(attribution)}, references={attribution.attrs['baseline_count']}, "
                  f"reference mean={attribution.attrs['reference_prediction']:.3f} $/MWh")
            importance = attribution.attrs["absolute_contributions"].mean().sort_values(ascending=False).rename(
                "mean_absolute_contribution"
            )
            feature_scope = "all timesteps" if model.requires_sequence else "single input row"
            print(f"Top 10 features ($/MWh; {feature_scope}):")
            print(importance.to_frame().join(attribution.mean().rename("mean_signed_contribution")).head(10).to_string(float_format=lambda x: f"{x:.3f}"))
            residual = np.abs(attribution.attrs["completeness_residual"])
            print(f"Attribution residual ($/MWh): mean absolute={residual.mean():.4f}, "
                  f"max absolute={residual.max():.4f}")
            if model.requires_sequence:
                timestep_summary = sequence_attribution_summary(
                    attribution.attrs["lag_attributions"], model.interval_minutes,
                )
                print("Timestep importance ($/MWh; all features; lag 0 = latest input):")
                print(timestep_summary.to_string(float_format=lambda x: f"{x:.3f}"))
            if run_directory is not None:
                path = run_directory / model_cli_name(model_factory)
                attribution.assign(
                    completeness_residual=attribution.attrs["completeness_residual"],
                ).to_csv(path / "gradient_shap.csv")
                importance.to_csv(path / "feature_importance.csv", index_label="feature")
                if model.requires_sequence:
                    attribution.attrs["lag_attributions"].to_csv(path / "gradient_shap_by_lag.csv")
                    timestep_summary.to_csv(path / "gradient_shap_timestep_summary.csv")

    print_evaluation_summary(results, prepared)
    if run_directory is not None:
        save_run_summary(run_directory, results)
        print(f"Run artifacts saved to {run_directory}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        cli_args.models,
        config=replace(
            EXPERIMENT,
            walk_forward_folds=cli_args.folds,
            selection_metric=cli_args.metric,
            random_seed=cli_args.seed,
        ),
        output_directory=cli_args.output_dir,
        save_results=cli_args.save,
        force_refresh=cli_args.refresh,
    )
