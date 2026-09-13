from dataclasses import dataclass
import numpy as np

from evaluation.Model import Model
from preprocessing import PreparedModelData

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
    actual = test_data.test_df["target_rrp"].to_numpy(dtype=float)
    predicted = np.asarray(model.predict(test_data), dtype=float).reshape(-1)

    if (predicted.size != actual.size)
        raise ValueError(
            f"{model.get_model_name()} returned {predicted.size} predictions"
            f"for {actual.size} test rows"
        )

    valid_mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask is not None:
        valid_mask &= np.asarray(mask, dtype=bool)

    return ModelResult(
        model_name=model.get_model_name(),
        predictions=predicted,
        valid_mask=valid_mask,
        metrics=_score(actual[valid_mask], predicted[valid_mask])
    )