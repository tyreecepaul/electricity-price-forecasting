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

    fit_df: pd.DataFrame = experiment.fit_df
    validation_df: pd.DataFrame = experiment.validation_df

    x_train: pd.DataFrame = fit_df.drop(["target_rrp"])
    y_train: pd.Series = fit_df["target_rrp"]

    x_val: pd.DataFrame = validation_df.drop(["target_rrp"])
    y_val: np.ndarray = validation_df["target_rrp"].to_numpy()

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
            model.train(x_train, y_train)
            y_pred: np.ndarray = model.predict(x_val)
            result: ModelResult = evaluate(model, y_val, y_pred)
            n_rows: int = int(result.valid_mask.sum())
            error = ""
        except Exception as exception:
            metrics: dict[str, float] = { name: float("nan") for name in METRIC_NAMES }
            n_rows = 0
            error = f"{ type(exception).__name__ }: {exception}"

        records.append({
            "trial": position,
            **config,
            **result.metrics,
            "n_rows": n_rows,
            "error": error
        })

        if verbose:
            outcome: str = error if error else f"{metric}={result.metrics[metric]:.3f}"
            print(f"[{position}/{len(candidates)}] : {outcome}")

    trials: pd.DataFrame = pd.DataFrame.from_records(records)
    scored: pd.DataFrame = trials[np.isfinite(trials[metric])]
    if scored.empty:
        raise RuntimeError("No configuration produced a usable score")

    best_row = scored.loc[scored[metric].idxmin()]
    best_config: dict[str, Any] = candidates[int(best_row["trial"]) - 1]
    best_score: float = float(best_row[metric])

    if verbose:
        print(f"Best {metric}: {best_score:.3f} with {best_config}")

    return SearchResult(
        best_config=best_config,
        best_score=best_score,
        metric=metric,
        trials=trials
    )
