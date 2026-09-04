"""Reusable data preparation and evaluation for dispatch-price models.

Model scripts should define only their estimator, fitting procedure, and any
model-specific diagnostics. This module owns experiment configuration, prepared
data caching, chronological validation, target reconstruction, and common
forecast evaluation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import PreparedModelData, prepare_model_data


TargetType = Literal["change", "aemo_error"]
FeatureSelection = Literal["all"] | Sequence[str]
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by all models in a comparable experiment."""

    region: str = "NSW1"
    horizon_minutes: int = 60
    target_type: TargetType = "change"
    model_features: FeatureSelection = "all"
    validation_fraction: float = 0.15
    cache_directory: Path = PROJECT_ROOT / "model_cache"

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.target_type not in {"change", "aemo_error"}:
            raise ValueError("target_type must be 'change' or 'aemo_error'")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")

    @property
    def aemo_forecast_column(self) -> str:
        return f"p5_h{self.horizon_minutes:02d}_rrp"

    @property
    def predicted_target_column(self) -> str:
        if self.target_type == "change":
            return "predicted_change"
        return "predicted_aemo_error"

    @property
    def feature_cache_key(self) -> str:
        if self.model_features == "all":
            return "all"
        serialised = json.dumps(list(self.model_features), separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]

    @property
    def cache_file(self) -> Path:
        filename = (
            f"prepared_{self.region}_{self.horizon_minutes}min_"
            f"{self.target_type}_{self.feature_cache_key}_"
            f"v{CACHE_SCHEMA_VERSION}.joblib"
        )
        return self.cache_directory / filename

    def cache_metadata(self) -> dict[str, object]:
        features: str | list[str]
        if self.model_features == "all":
            features = "all"
        else:
            features = list(self.model_features)
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "region": self.region,
            "horizon_minutes": self.horizon_minutes,
            "target_type": self.target_type,
            "model_features": features,
        }


@dataclass(frozen=True)
class ExperimentData:
    """Prepared data and the leakage-safe model-selection split."""

    config: ExperimentConfig
    prepared: PreparedModelData
    fit_df: pd.DataFrame
    validation_df: pd.DataFrame

    @property
    def features(self) -> list[str]:
        return self.prepared.features


def _cache_payload_is_valid(
    payload: object,
    config: ExperimentConfig,
) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("metadata") == config.cache_metadata()
        and isinstance(payload.get("data"), PreparedModelData)
    )


def load_prepared_data(
    config: ExperimentConfig,
    *,
    force_refresh: bool = False,
) -> PreparedModelData:
    """Load a compatible prepared-data cache or construct and store one."""
    config.cache_directory.mkdir(parents=True, exist_ok=True)

    if config.cache_file.exists() and not force_refresh:
        payload = joblib.load(config.cache_file)
        if _cache_payload_is_valid(payload, config):
            print(f"Loading prepared data from {config.cache_file.name}...")
            return payload["data"]
        print("Prepared-data cache metadata is incompatible; rebuilding cache...")
    else:
        print("Preparing model data...")

    data = prepare_model_data(
        region=config.region,
        horizon=config.horizon_minutes,
        features=config.model_features,
        target=config.target_type,
    )
    payload = {
        "metadata": config.cache_metadata(),
        "data": data,
    }
    joblib.dump(payload, config.cache_file)
    print(f"Prepared data cached as {config.cache_file.name}.")
    return data


def chronological_validation_split(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological split with an embargo equal to the horizon."""
    split_index = int(len(train_df) * (1 - config.validation_fraction))
    if split_index <= 0 or split_index >= len(train_df):
        raise RuntimeError("Validation split produced an empty subset")

    validation_start = train_df.iloc[split_index]["settlement_date"]
    target_delta = pd.Timedelta(minutes=config.horizon_minutes)

    fit_df = train_df[
        (train_df["settlement_date"] + target_delta) < validation_start
    ].copy()
    validation_df = train_df[
        train_df["settlement_date"] >= validation_start
    ].copy()

    if fit_df.empty or validation_df.empty:
        raise RuntimeError("Chronological validation split produced an empty subset")
    return fit_df, validation_df


def prepare_experiment(
    config: ExperimentConfig,
    *,
    force_refresh: bool = False,
) -> ExperimentData:
    """Load prepared data and construct the model-selection split."""
    prepared = load_prepared_data(config, force_refresh=force_refresh)
    fit_df, validation_df = chronological_validation_split(
        prepared.train_df,
        config,
    )
    return ExperimentData(
        config=config,
        prepared=prepared,
        fit_df=fit_df,
        validation_df=validation_df,
    )


def reconstruct_price_forecast(
    frame: pd.DataFrame,
    predicted_target: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Convert target-scale model output into an RRP forecast."""
    if config.target_type == "change":
        base = frame["rrp"].to_numpy()
    else:
        base = frame[config.aemo_forecast_column].to_numpy()
    return base + np.asarray(predicted_target)


def aemo_forecast_values(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> np.ndarray:
    """Return aligned AEMO forecasts, using NaN when the column is unavailable."""
    if config.aemo_forecast_column not in frame.columns:
        return np.full(len(frame), np.nan)
    return frame[config.aemo_forecast_column].to_numpy()


def forecast_metrics(actual, predicted) -> dict[str, float]:
    """Calculate standard point-forecast error metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "median_ae": float(median_absolute_error(actual, predicted)),
    }


def print_metrics(
    title: str,
    actual,
    forecasts: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Print and return metrics for forecasts evaluated on identical rows."""
    print(f"\n{title}")
    print("-" * len(title))
    values = {
        name: forecast_metrics(actual, prediction)
        for name, prediction in forecasts.items()
    }
    for metric_name, label in (
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("median_ae", "Median AE"),
    ):
        print(f"\n{label}")
        for name, metrics in values.items():
            print(f"{name:<16} {metrics[metric_name]:.2f}")
    return values


def print_improvement(
    model_name: str,
    model_mae: float,
    baseline_name: str,
    baseline_mae: float,
) -> None:
    """Print the percentage MAE improvement relative to a baseline."""
    if baseline_mae == 0:
        return
    improvement = (baseline_mae - model_mae) / baseline_mae * 100.0
    print(
        f"{model_name} MAE improvement over {baseline_name}: "
        f"{improvement:.2f}%"
    )


def print_experiment_summary(experiment: ExperimentData) -> None:
    """Report the prepared sample, features, and chronological split."""
    data = experiment.prepared
    config = experiment.config
    train_df = data.train_df
    test_df = data.test_df

    print("\nPREPARED MODEL DATA")
    print("-------------------")
    print("Target region:", data.region)
    print("Forecast horizon:", data.horizon, "minutes")
    print("Target type:", data.target_type)
    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Features:", len(data.features))
    print("P5MIN features:", sum(c.startswith("p5_h") for c in data.features))
    print("Interconnector features:", sum(c.startswith("ic_") for c in data.features))
    print(
        "Training period:",
        train_df["settlement_date"].min(),
        "->",
        train_df["settlement_date"].max(),
    )
    print(
        "Test period:",
        test_df["settlement_date"].min(),
        "->",
        test_df["settlement_date"].max(),
    )

    column = config.aemo_forecast_column
    print("\nAEMO FORECAST AVAILABILITY")
    print("--------------------------")
    for label, frame in (("training", train_df), ("test", test_df)):
        count = frame[column].notna().sum() if column in frame.columns else 0
        print(f"Valid {label} {column}:", count)

    print("\nCHRONOLOGICAL VALIDATION")
    print("------------------------")
    print(
        "Fit:",
        experiment.fit_df["settlement_date"].min(),
        "->",
        experiment.fit_df["settlement_date"].max(),
    )
    print(
        "Validation:",
        experiment.validation_df["settlement_date"].min(),
        "->",
        experiment.validation_df["settlement_date"].max(),
    )
    print("Fit rows:", len(experiment.fit_df))
    print("Validation rows:", len(experiment.validation_df))


def evaluate_aemo_rows(
    title: str,
    frame: pd.DataFrame,
    model_predictions: np.ndarray,
    config: ExperimentConfig,
    *,
    model_name: str,
    include_persistence: bool,
) -> dict[str, dict[str, float]] | None:
    """Compare forecasts on the subset containing aligned AEMO predictions."""
    column = config.aemo_forecast_column
    if column not in frame.columns:
        print(f"\nNo {column} forecasts are available for comparison.")
        return None

    mask = frame[column].notna().to_numpy()
    if not mask.any():
        print(f"\nNo usable {column} forecasts are available for comparison.")
        return None

    forecasts = {model_name: np.asarray(model_predictions)[mask]}
    if include_persistence:
        forecasts["Persistence"] = frame.loc[mask, "rrp"].to_numpy()
    forecasts["AEMO P5MIN"] = frame.loc[mask, column].to_numpy()

    metrics = print_metrics(
        title,
        frame.loc[mask, "target_rrp"].to_numpy(),
        forecasts,
    )
    print_improvement(
        model_name,
        metrics[model_name]["mae"],
        "AEMO P5MIN",
        metrics["AEMO P5MIN"]["mae"],
    )
    return metrics


def build_results_frame(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    predicted_target: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Assemble interval-level forecasts, targets, baselines, and errors."""
    actual = test_df["target_rrp"].to_numpy()
    current = test_df["rrp"].to_numpy()
    aemo = aemo_forecast_values(test_df, config)

    results = pd.DataFrame(
        {
            "settlement_date": test_df["settlement_date"].values,
            "region": test_df["region"].values,
            "current": current,
            "actual": actual,
            "actual_change": actual - current,
            "actual_aemo_error": actual - aemo,
            config.predicted_target_column: predicted_target,
            "predicted": predictions,
            "baseline": current,
            "aemo_p5min": aemo,
        }
    )
    results["model_error"] = np.abs(results["actual"] - results["predicted"])
    results["baseline_error"] = np.abs(results["actual"] - results["baseline"])
    results["aemo_error"] = np.abs(results["actual"] - results["aemo_p5min"])
    return results


def print_price_regime_results(
    results: pd.DataFrame,
    *,
    model_name: str,
) -> None:
    """Report MAE for normal, high, and spike price intervals."""
    bands = (
        ("Normal (< $300)", results["actual"] < 300),
        (
            "High ($300-$1000)",
            (results["actual"] >= 300) & (results["actual"] < 1000),
        ),
        ("Spike (> $1000)", results["actual"] >= 1000),
    )

    print("\nRESULTS BY PRICE REGIME")
    print("-----------------------")
    for name, mask in bands:
        subset = results[mask]
        if subset.empty:
            continue
        print(f"\n{name}")
        print("Observations:", len(subset))
        model_mae = mean_absolute_error(subset["actual"], subset["predicted"])
        print(f"{model_name:<12} ${model_mae:.2f}")
        print(f"{'Persistence':<12} ${mean_absolute_error(subset['actual'], subset['baseline']):.2f}")
        aemo_subset = subset.dropna(subset=["aemo_p5min"])
        if not aemo_subset.empty:
            aemo_mae = mean_absolute_error(
                aemo_subset["actual"],
                aemo_subset["aemo_p5min"],
            )
            print(f"{'AEMO P5MIN':<12} ${aemo_mae:.2f}")


def print_target_diagnostics(
    actual_target: np.ndarray,
    predicted_target: np.ndarray,
    config: ExperimentConfig,
) -> None:
    """Print distributional diagnostics on the model's target scale."""
    title = f"{config.target_type.upper()} TARGET DIAGNOSTICS"
    print(f"\n{title}")
    print("-" * len(title))
    diagnostics = pd.DataFrame(
        {
            f"actual_{config.target_type}": actual_target,
            config.predicted_target_column: predicted_target,
        }
    )
    print(
        diagnostics.describe(
            percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
    )
