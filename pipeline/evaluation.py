from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.contracts import forecast_vector
from pipeline.model import Model
from pipeline.preprocessing import PreparedModelData

METRIC_NAMES: list[str] = ["mae", "rmse", "sse", "median_ae"]
PRICE_REGIMES: tuple[tuple[str, float | None, float | None], ...] = (
    ("Negative (< $0/MWh)", None, 0.0),
    ("Normal ($0-$300/MWh)", 0.0, 300.0),
    ("High ($300-$1,000/MWh)", 300.0, 1000.0),
    ("Extreme (>= $1,000/MWh)", 1000.0, None),
)


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
        return {name: float("nan") for name in METRIC_NAMES}

    errors: np.ndarray = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "sse": float(np.sum(errors**2)),
        "median_ae": float(np.median(np.abs(errors))),
    }


def evaluate(
    model: Model,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> ModelResult:
    actual = forecast_vector(actual, len(actual), name="actual")
    predicted = forecast_vector(predicted, len(actual))

    valid_mask: np.ndarray = np.isfinite(actual) & np.isfinite(predicted)
    if mask is not None:
        valid_mask &= forecast_vector(mask, len(actual), name="mask").astype(bool)

    return ModelResult(
        model_name=model.get_model_name(),
        predictions=predicted,
        valid_mask=valid_mask,
        metrics=_score(actual[valid_mask], predicted[valid_mask]),
    )


def _comparison_inputs(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
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

    return actual, shared_mask, forecasts


def _comparison_table(
    actual: np.ndarray,
    forecasts: list[tuple[str, np.ndarray]],
    rows: np.ndarray,
) -> pd.DataFrame:
    records = []
    for name, predicted in forecasts:
        records.append(
            {
                "model": name,
                **_score(actual[rows], predicted[rows]),
                "n_rows": int(rows.sum()),
            }
        )

    comparison = pd.DataFrame.from_records(records).set_index("model")
    persistence_mae = comparison.loc["Persistence", "mae"]
    if persistence_mae == 0:
        comparison["mae_vs_persistence_pct"] = np.nan
    else:
        comparison["mae_vs_persistence_pct"] = (
            (persistence_mae - comparison["mae"]) / persistence_mae * 100.0
        )

    if "AEMO P5MIN" in comparison.index:
        aemo_mae = comparison.loc["AEMO P5MIN", "mae"]
        if aemo_mae == 0:
            comparison["mae_vs_aemo_pct"] = np.nan
        else:
            comparison["mae_vs_aemo_pct"] = (
                (aemo_mae - comparison["mae"]) / aemo_mae * 100.0
            )

    return comparison.sort_values("mae")


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
    actual, shared_mask, forecasts = _comparison_inputs(results, test_data)
    return _comparison_table(actual, forecasts, shared_mask)


def compare_models_by_price_regime(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Compare forecasts within regimes defined by the actual future RRP."""
    actual, shared_mask, forecasts = _comparison_inputs(results, test_data)
    comparisons: dict[str, pd.DataFrame] = {}

    for name, lower, upper in PRICE_REGIMES:
        regime_mask = shared_mask.copy()
        if lower is not None:
            regime_mask &= actual >= lower
        if upper is not None:
            regime_mask &= actual < upper
        if regime_mask.any():
            comparisons[name] = _comparison_table(
                actual,
                forecasts,
                regime_mask,
            )

    if not comparisons:
        raise RuntimeError("No shared test rows fall within a price regime")
    return pd.concat(comparisons, names=["regime", "model"])


def _display_table(
    comparison: pd.DataFrame,
    *,
    include_sse: bool,
    include_rows: bool,
) -> str:
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
    if not include_sse:
        display = display.drop(columns="SSE")
    if not include_rows:
        display = display.drop(columns="Rows")

    formatters = {
        "MAE": lambda value: f"{value:.3f}",
        "RMSE": lambda value: f"{value:.3f}",
        "Median AE": lambda value: f"{value:.3f}",
        "vs Persistence": lambda value: f"{value:+.2f}%",
    }
    if include_sse:
        formatters["SSE"] = lambda value: f"{value:.3e}"
    if include_rows:
        formatters["Rows"] = lambda value: f"{int(value):,}"
    if "vs AEMO" in display.columns:
        formatters["vs AEMO"] = lambda value: f"{value:+.2f}%"

    return display.to_string(formatters=formatters, na_rep="n/a")


def print_price_regime_summary(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Print model and baseline performance for each actual-price regime."""
    comparisons = compare_models_by_price_regime(results, test_data)

    print("\nRESULTS BY ACTUAL PRICE REGIME")
    print("------------------------------")
    for regime in comparisons.index.get_level_values("regime").unique():
        comparison = comparisons.xs(regime, level="regime")
        row_count = int(comparison["n_rows"].iloc[0])
        print(f"\n{regime} ({row_count:,} rows)")
        print(
            _display_table(
                comparison,
                include_sse=False,
                include_rows=False,
            )
        )
    return comparisons


def print_evaluation_summary(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Print compact overall and price-regime final-test comparisons."""
    comparison = compare_models(results, test_data)

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print("All forecasts are scored on the same finite test rows.")
    print(_display_table(comparison, include_sse=True, include_rows=True))
    print("\nPositive percentages mean lower MAE than the named baseline.")
    print_price_regime_summary(results, test_data)
    return comparison
