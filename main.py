import argparse
from electricity_price_forecasting_model.xgboost import XGBoostModel
from electricity_price_forecasting_model.mlp import MLPModel
from pipeline.evaluation import ModelResult, evaluate, print_evaluation_summary
from pipeline.experiment_pipeline import (
    ExperimentConfig,
    prepare_experiment,
    print_experiment_summary,
    ExperimentData,
    reconstruct_price_forecast,
)
from pipeline.tuning import search_hyperparameters, SearchResult
from pipeline.model import Model

from typing import Sequence

import pandas as pd
import numpy as np

EXPERIMENT: ExperimentConfig = ExperimentConfig(
    region="NSW1",
    horizon_minutes=30,
    target_type="change",
    model_features="all",
    validation_fraction=0.20
)

MODEL_REGISTRY: list[tuple[type[Model], dict[str, Sequence]]] = [
    (
        XGBoostModel,
        {
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [4, 8],
            "max_estimators": [500, 1000],
            "random_seed": [42],
        }
    ),
    (
        MLPModel,
        {
            "hidden_sizes": [(128, 64, 32)],
            "dropout": [(0.2, 0.15, 0.0)],
            "learning_rate": [3e-4, 1e-3],
            "weight_decay": [1e-4],
            "batch_size": [256],
            "max_epochs": [5],
            "loss": ["huber", "mse", "mae"],
            "huber_delta": [1.0],
            "random_seed": [42],
        },
    ),
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

    requested = set(selected_models)
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
        choices=sorted(model_cli_name(factory) for factory, _ in MODEL_REGISTRY),
        help="Models to run. Omit this option to run every registered model.",
    )
    return parser.parse_args(arguments)


def main(selected_models: Sequence[str] | None = None) -> None:
    # Prepare experiment
    experiment: ExperimentData = prepare_experiment(EXPERIMENT)
    print_experiment_summary(experiment)

    prepared = experiment.prepared

    x_train: pd.DataFrame = prepared.X_train
    y_train: pd.Series = prepared.y_train
    x_test: pd.DataFrame = prepared.X_test

    y_test_price: np.ndarray = experiment.test_df["target_rrp"].to_numpy()

    # Fit each model and find optimal hyperparameters
    results: list[ModelResult] = []

    for model_factory, hyperparam_grid in select_model_registry(selected_models):
        print(f"=== {model_factory.model_name} ===")

        search_result: SearchResult = search_hyperparameters(
            model_factory,
            hyperparam_grid,
            experiment
        )

        model: Model = model_factory(**search_result.best_config)
        model.train(x_train, y_train)

        y_pred: np.ndarray = model.predict(x_test)
        predicted_price: np.ndarray = reconstruct_price_forecast(
            experiment.test_df,
            y_pred,
            experiment.config
        )

        result = evaluate(model, y_test_price, predicted_price)

        results.append(result)

    print_evaluation_summary(results, prepared)

if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args.models)
