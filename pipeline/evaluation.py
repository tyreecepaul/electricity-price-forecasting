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


def _validate_result_lengths(
    results: list[ModelResult],
    expected_rows: int,
) -> None:
    for result in results:
        if result.predictions.size != expected_rows:
            raise ValueError(
                f"{result.model_name} has {result.predictions.size} predictions "
                f"for {expected_rows} test rows"
            )
        if result.valid_mask.size != expected_rows:
            raise ValueError(
                f"{result.model_name} has a validity mask of length "
                f"{result.valid_mask.size} for {expected_rows} test rows"
            )

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
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> ModelResult:
    if (predicted.size != actual.size):
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
    """Compare models and baselines on exactly the same finite test rows.

    Persistence is the current RRP at the forecast origin. When aligned AEMO
    P5MIN forecasts are available, their finite-row mask is also included in
    the shared comparison mask so every value in the table is directly
    comparable.
    """
    if not results:
        raise ValueError("needs at least one ModelResult")

    test_df = test_data.test_df
    actual = test_df["target_rrp"].to_numpy(dtype=float)
    persistence = test_df["rrp"].to_numpy(dtype=float)
    _validate_result_lengths(results, actual.size)

    shared_mask = np.isfinite(actual) & np.isfinite(persistence)
    for result in results:
        shared_mask &= np.asarray(result.valid_mask, dtype=bool)

    forecasts: list[tuple[str, np.ndarray]] = [
        (result.model_name, np.asarray(result.predictions, dtype=float))
        for result in results
    ]
    forecasts.append(("Persistence", persistence))

    aemo_column = f"p5_h{test_data.horizon:02d}_rrp"
    if aemo_column in test_df.columns:
        aemo = test_df[aemo_column].to_numpy(dtype=float)
        aemo_mask = np.isfinite(aemo)
        if aemo_mask.any():
            shared_mask &= aemo_mask
            forecasts.append(("AEMO P5MIN", aemo))

    if not shared_mask.any():
        raise RuntimeError("No test rows are shared by all models and baselines")

    records = []
    for name, predicted in forecasts:
        records.append(
            {
                "model": name,
                **_score(actual[shared_mask], predicted[shared_mask]),
                "n_rows": int(shared_mask.sum()),
            }
        )

    comparison = pd.DataFrame.from_records(records).set_index("model")
    persistence_mae = comparison.loc["Persistence", "mae"]
    comparison["mae_vs_persistence_pct"] = np.where(
        persistence_mae == 0,
        np.nan,
        (persistence_mae - comparison["mae"]) / persistence_mae * 100.0,
    )

    if "AEMO P5MIN" in comparison.index:
        aemo_mae = comparison.loc["AEMO P5MIN", "mae"]
        comparison["mae_vs_aemo_pct"] = np.where(
            aemo_mae == 0,
            np.nan,
            (aemo_mae - comparison["mae"]) / aemo_mae * 100.0,
        )

    return comparison.sort_values("mae")


def print_evaluation_summary(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Print a compact final-test comparison and return its numeric table."""
    comparison = compare_models(results, test_data)
    display = comparison.rename(
        columns={
            "mae": "MAE",
            "rmse": "RMSE",
            "sse": "SSE",
            "median_ae": "Median AE",
            "n_rows": "Rows",
            "mae_vs_persistence_pct": "vs Persistence",
            "mae_vs_aemo_pct": "vs AEMO",
        }
    )

    formatters = {
        "MAE": lambda value: f"{value:.3f}",
        "RMSE": lambda value: f"{value:.3f}",
        "SSE": lambda value: f"{value:.3e}",
        "Median AE": lambda value: f"{value:.3f}",
        "Rows": lambda value: f"{int(value):,}",
        "vs Persistence": lambda value: f"{value:+.2f}%",
    }
    if "vs AEMO" in display.columns:
        formatters["vs AEMO"] = lambda value: f"{value:+.2f}%"

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print("All forecasts are scored on the same finite test rows.")
    print(display.to_string(formatters=formatters, na_rep="n/a"))
    print("\nPositive percentages mean lower MAE than the named baseline.")
    return comparison
