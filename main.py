from electricity_price_forecasting_model.xgboost import XGBoostModel
from pipeline.evaluation import ModelResult, evaluate
from pipeline.experiment_pipeline import (
    ExperimentConfig,
    prepare_experiment,
    print_experiment_summary,
    ExperimentData,
    reconstruct_price_forecast
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

MODEL_REGISTRY: list[tuple[Model, dict[str, Sequence]]] = [
    (
        XGBoostModel,
        {
            "max_depth": [3,4,6],
            "learning_rate": [0.03, 0.1], 
            "min_child_weight": [4,8]
        }
    )
]

def main() -> None:
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

    for model_factory, hyperparam_grid in MODEL_REGISTRY:
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

    print(results)

if __name__ == "__main__":
    main()