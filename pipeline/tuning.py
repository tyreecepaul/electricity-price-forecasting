from dataclasses import dataclass
from typing import Callable, Sequence, Any
import itertools

from pipeline.experiment_pipeline import ExperimentData 
from pipeline.model import Model
from pipeline.evaluation import METRIC_NAMES, ModelResult, evaluate

import numpy as np
import pandas as pd


@dataclass
class SearchResult:
    best_config: dict
    best_score: float 
    metric: str 
    trials: pd.DataFrame

def search_hyperparameters(
    model_factory: Callable[..., Model],
    hyperparam_grid: dict[str, Sequence],
    experiment: ExperimentData,
    *,
    metric: str = "mae",
    seed: int = 42,
    verbose: bool = True
) -> SearchResult:
    if metric not in METRIC_NAMES:
        raise ValueError(f"Metric must be one of {METRIC_NAMES}")

    features = experiment.features

    fit_df: pd.DataFrame = experiment.fit_df
    validation_df: pd.DataFrame = experiment.validation_df

    x_train: pd.DataFrame = fit_df[features]
    y_train: pd.Series = fit_df["target"]

    x_val: pd.DataFrame = validation_df[features]
    y_val_series: pd.Series = validation_df["target"]
    y_val: np.ndarray = validation_df["target"].to_numpy()

    candidates: list[dict[str, Any]] = [
        dict(zip(hyperparam_grid, values))
        for values in itertools.product(*hyperparam_grid.values())
    ]

    if verbose:
        print(f"Searching {len(candidates)} configurations")
        print(f"Fit rows: {len(fit_df)} | validation rows: {len(validation_df)}")

    records: list[dict[str, object]] = []
    for position, config in enumerate(candidates, start=1):
        try:
            model: Model = model_factory(**config)
            validation_trainer = getattr(model, "train_with_validation", None)
            if validation_trainer is None:
                model.train(x_train, y_train)
                training_result = None
            else:
                training_result = validation_trainer(
                    x_train,
                    y_train,
                    x_val,
                    y_val_series,
                )
            y_pred: np.ndarray = model.predict(x_val)
            result: ModelResult = evaluate(model, y_val, y_pred)

            if not result.valid_mask.all():
                invalid_count = int((~result.valid_mask).sum())
                raise ValueError(
                    f"Model produced {invalid_count} non-finite validation results"
                )

            metrics = result.metrics
            n_rows: int = int(result.valid_mask.sum())
            selected_epoch = getattr(training_result, "best_epoch", None)
            error = ""
        except Exception as exception:
            metrics: dict[str, float] = { name: float("nan") for name in METRIC_NAMES }
            n_rows = 0
            selected_epoch = None
            error = f"{ type(exception).__name__ }: {exception}"

        records.append({
            "trial": position,
            **config,
            **metrics,
            "n_rows": n_rows,
            "selected_epoch": selected_epoch,
            "error": error
        })

        if verbose:
            outcome: str = error if error else f"{metric}={metrics[metric]:.3f}"
            if not error and selected_epoch is not None:
                outcome += f" (best epoch {selected_epoch})"
            print(f"[{position}/{len(candidates)}] : {outcome}")

    trials: pd.DataFrame = pd.DataFrame.from_records(records)
    scored: pd.DataFrame = trials[np.isfinite(trials[metric])]
    if scored.empty:
        raise RuntimeError("No configuration produced a usable score")

    best_row = scored.loc[scored[metric].idxmin()]
    best_config: dict[str, Any] = candidates[int(best_row["trial"]) - 1].copy()
    selected_epoch = best_row.get("selected_epoch")
    if "max_epochs" in best_config and pd.notna(selected_epoch):
        best_config["max_epochs"] = int(selected_epoch)
    best_score: float = float(best_row[metric])

    if verbose:
        print(f"Best {metric}: {best_score:.3f} with {best_config}")

    return SearchResult(
        best_config=best_config,
        best_score=best_score,
        metric=metric,
        trials=trials
    )
