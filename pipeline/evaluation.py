from dataclasses import dataclass
import numpy as np
import pandas as pd

from pipeline.model import Model
from pipeline.preprocessing import PreparedModelData

METRIC_NAMES: list[str] = ["mae", "rmse", "sse", "median_ae"]

@dataclass 
class ModelResult:
    model_name: str 
    predictions: np.ndarray
    valid_mask: np.ndarray
    metrics: dict[str, float]

def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if actual.size == 0:
        return { name: float("nan") for name in METRIC_NAMES }

    errors: np.ndarray = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "sse": float(np.sum(errors ** 2)),
        "median_ae": float(np.median(np.abs(errors))),
    }

def evaluate(
    model: Model,
    test_data: PreparedModelData,
    *,
    mask: np.ndarray | None = None,
) -> ModelResult:
    actual: np.ndarray = test_data.test_df["target_rrp"].to_numpy(dtype=float)
    predicted: np.ndarray = np.asarray(model.predict(test_data), dtype=float).reshape(-1)

    if (predicted.size != actual.size)
        raise ValueError(
            f"{model.get_model_name()} returned {predicted.size} predictions"
            f"for {actual.size} test rows"
        )

    valid_mask: np.ndarray = np.isfinite(actual) & np.isfinite(predicted)
    if mask is not None:
        valid_mask &= np.asarray(mask, dtype=bool)

    return ModelResult(
        model_name=model.get_model_name(),
        predictions=predicted,
        valid_mask=valid_mask,
        metrics=_score(actual[valid_mask], predicted[valid_mask])
    )

def compare_models(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    if not results:
        raise ValueError("needs at least one ModelResult")

    actual: np.ndarray = test_data.test_df["target_rrp"].to_numpy(dtype=float)

    shared_mask = np.isfinite(actual)
    for result in results:
        shared_mask &= result.valid_mask
    
    records: list[dict[str, object]] = []

    def add(name: str, predicted: np.ndarray, rows: np.ndarray) -> None:
        records.append(
            {
                "model": name,
                **_score(actual[rows], predicted[rows]),
                "n_rows": int (rows.sum())
            }
        )

    for result in results:
        add(result.model_name, result.predictions, shared_mask)

    baselines: dict[str, str] = {
        "Persistence": "rrp",
        "AEMO P5MIN": f"p5_h{test_data.horizon:02d}_rrp"
    }

    for name, column in baselines.items():
        if column not in test_data.test_df.columns:
            continue

        values: np.ndarray = test_data.test_df[column].to_numpy(dtype=float)
        baseline_mask: np.ndarray = shared_mask & np.isfinite(values)
        if not baseline_mask.any():
            continue

        add(name, values, baseline_mask)
    
    return pd.DataFrame.from_records(records).set_index("model").sort_values("mae")
