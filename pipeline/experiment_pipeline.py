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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.contracts import forecast_array, target_columns
from pipeline.graph_embedding import GraphEmbeddingConfig
from pipeline.preprocessing import (
    PreparedModelData,
    forecast_reliability_columns,
    prepare_model_data,
    add_targets,
)
from pipeline.preprocessing.adapters import LearnedFeatures
from pipeline.preprocessing.config import FeatureConfig

TargetType = Literal["change", "price", "aemo_error"]
FeatureSelection = Literal["all"] | Sequence[str]
CACHE_SCHEMA_VERSION = 3
HOUR_HORIZONS = tuple(range(5, 61, 5))


def source_fingerprint() -> str:
    """Fingerprint preparation code and local raw-cache file identities.

    Raw files use path, size and nanosecond mtime (not a full content hash).
    Upstream revisions not downloaded locally still require force_refresh.
    """
    digest = hashlib.sha256()
    sources = sorted((PROJECT_ROOT / "pipeline" / "preprocessing").glob("*.py"))
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    ingestion = PROJECT_ROOT / "pipeline" / "data_cleaning" / "data.py"
    digest.update(ingestion.read_bytes())
    for name in ("nemosis_cache", "nemseer_cache"):
        for path in sorted((ingestion.parent / name).rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(
                    f"{path.relative_to(PROJECT_ROOT)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
                )
    return digest.hexdigest()


def graph_source_fingerprint() -> str:
    """Fingerprint the graph-encoder code, used only when the stage is enabled.

    Kept apart from ``source_fingerprint`` so editing the encoder does not
    invalidate prepared data for experiments that never use it.
    """
    digest = hashlib.sha256()
    layers = PROJECT_ROOT / "electricity_price_forecasting_model" / "layers" / "gnn_layers"
    for directory in (PROJECT_ROOT / "pipeline" / "graph_embedding", layers):
        for source in sorted(directory.glob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by all models in a comparable experiment."""

    region: str = "NSW1"
    horizon_minutes: int = 60
    target_type: TargetType = "change"
    model_features: FeatureSelection = "all"
    validation_fraction: float = 0.15
    cache_directory: Path = PROJECT_ROOT / "model_cache"
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    walk_forward_folds: int = 1
    selection_metric: str = "mae"
    random_seed: int = 42
    # Reliability features are appended after loading the shared prepared cache.
    forecast_reliability: bool = False
    forecast_mode: Literal["single", "multioutput"] = "single"
    graph_embedding: GraphEmbeddingConfig | None = None

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.forecast_mode not in {"single", "multioutput"}:
            raise ValueError("forecast_mode must be 'single' or 'multioutput'")
        if self.forecast_mode == "multioutput" and self.feature_config.interval_minutes != 5:
            raise ValueError("The 12-output experiment requires five-minute data")
        if self.target_type not in {"change", "price", "aemo_error"}:
            raise ValueError("target_type must be 'change', 'price', or 'aemo_error'")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if (
            self.walk_forward_folds < 1
            or int(self.walk_forward_folds) != self.walk_forward_folds
        ):
            raise ValueError("walk_forward_folds must be a positive integer")
        if self.selection_metric not in {"mae", "rmse", "sse", "median_ae"}:
            raise ValueError("Unsupported selection_metric")

    @property
    def forecast_horizons(self) -> tuple[int, ...]:
        """Derive the output layout without a second configurable mode setting."""
        return HOUR_HORIZONS if self.forecast_mode == "multioutput" else ()

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.forecast_horizons or (self.horizon_minutes,)

    @property
    def aemo_forecast_column(self) -> str:
        return f"p5_h{self.horizon_minutes:02d}_rrp"

    @property
    def predicted_target_column(self) -> str:
        if self.target_type == "change":
            return "predicted_change"
        if self.target_type == "aemo_error":
            return "predicted_aemo_error"
        return "predicted_price"

    @property
    def feature_cache_key(self) -> str:
        if self.model_features == "all":
            return "all"
        serialised = json.dumps(list(self.model_features), separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]

    @property
    def graph_embedding_cache_key(self) -> str:
        """Empty without the stage, so plain experiments keep their filenames."""
        if self.graph_embedding is None:
            return ""
        serialised = json.dumps(asdict(self.graph_embedding), sort_keys=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]

    @property
    def cache_file(self) -> Path:
        embedding = self.graph_embedding_cache_key
        filename = (
            f"prepared_{self.region}_{self.horizon_minutes}min_"
            f"{self.target_type}_{self.feature_cache_key}_"
            f"{f'gnn{embedding}_' if embedding else ''}"
            f"v{CACHE_SCHEMA_VERSION}.joblib"
        )
        return self.cache_directory / filename

    @property
    def graph_embedding_cache_file(self) -> Path:
        """Encoder output is independent of ``region`` and shared across them."""
        filename = (
            f"graph_embedding_{self.horizon_minutes}min_{self.target_type}_"
            f"{self.graph_embedding_cache_key}_v{CACHE_SCHEMA_VERSION}.joblib"
        )
        return self.cache_directory / filename

    def cache_metadata(self) -> dict[str, object]:
        features: str | list[str]
        if self.model_features == "all":
            features = "all"
        else:
            features = list(self.model_features)
        metadata: dict[str, object] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "region": self.region,
            "horizon_minutes": self.horizon_minutes,
            "target_type": self.target_type,
            "model_features": features,
            "feature_config": asdict(self.feature_config),
            "source_fingerprint": source_fingerprint(),
        }
        if self.graph_embedding is not None:
            metadata["graph_embedding"] = asdict(self.graph_embedding)
            metadata["graph_source_fingerprint"] = graph_source_fingerprint()
        return metadata

    def graph_embedding_cache_metadata(self) -> dict[str, object]:
        metadata = self.cache_metadata()
        del metadata["region"], metadata["model_features"]
        return metadata


@dataclass(frozen=True)
class ExperimentData:
    """Prepared data and the leakage-safe model-selection split."""

    config: ExperimentConfig
    prepared: PreparedModelData
    fit_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame

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


def graph_embedding_stage(config: ExperimentConfig, *, force_refresh: bool = False):
    """Return the learned-feature stage for ``prepare_model_data``, or None.

    The encoder is fitted once per horizon/target and cached, so experiments
    on different regions reuse the same embeddings.
    """
    if config.graph_embedding is None:
        return None

    def stage(train_features: pd.DataFrame, test_features: pd.DataFrame):
        path = config.graph_embedding_cache_file
        metadata = config.graph_embedding_cache_metadata()
        if path.exists() and not force_refresh:
            payload = joblib.load(path)
            if (
                isinstance(payload, dict)
                and payload.get("metadata") == metadata
                and isinstance(payload.get("data"), LearnedFeatures)
            ):
                print(f"Loading graph embeddings from {path.name}...")
                return payload["data"]
        print("Fitting graph encoder...")
        # Imported here so experiments without the stage never load the encoder.
        from pipeline.graph_embedding.embedder import compute_graph_embeddings

        learned = compute_graph_embeddings(
            train_features,
            test_features,
            config=config.graph_embedding,
            horizon=config.horizon_minutes,
            target=config.target_type,
            interval_minutes=config.feature_config.interval_minutes,
        )
        joblib.dump({"metadata": metadata, "data": learned}, path)
        return learned

    return stage


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
        config=config.feature_config,
        learned_features=graph_embedding_stage(config, force_refresh=force_refresh),
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
    validation_df = train_df[train_df["settlement_date"] >= validation_start].copy()

    if fit_df.empty or validation_df.empty:
        raise RuntimeError("Chronological validation split produced an empty subset")
    return fit_df, validation_df


def with_forecast_reliability(prepared: PreparedModelData) -> PreparedModelData:
    """Append the forecast reliability features to the model inputs."""
    extra = [
        column
        for column in forecast_reliability_columns(prepared.train_df)
        if column not in prepared.features
        and column in prepared.test_df.columns
        and prepared.train_df[column].notna().any()
    ]
    if not extra:
        raise ValueError(
            "Prepared data has no forecast reliability features; they need "
            "aligned AEMO P5MIN price forecasts"
        )
    features = [*prepared.features, *extra]
    return replace(
        prepared,
        X_train=prepared.train_df[features].copy(),
        X_test=prepared.test_df[features].copy(),
        features=features,
    )


def prepare_experiment(
    config: ExperimentConfig,
    *,
    force_refresh: bool = False,
) -> ExperimentData:
    """Load prepared data and construct the model-selection split."""
    if config.forecast_horizons:
        # Use the longest lead for caching and embargoes without changing the
        # user's configured single-target horizon or the CLI override defaults.
        features = config.feature_config
        if config.target_type == "aemo_error":
            features = replace(features, p5min_horizons_minutes=config.forecast_horizons)
        config = replace(config, horizon_minutes=max(config.forecast_horizons), feature_config=features)
    prepared = load_prepared_data(config, force_refresh=force_refresh)
    if config.forecast_reliability:
        prepared = with_forecast_reliability(prepared)
    # Explicit feature selections must not turn future outcomes into inputs.
    # Learned graph embeddings remain valid ordinary feature columns.
    leaked = [c for c in prepared.features if c == "target" or c.startswith("target_")]
    if leaked:
        raise ValueError(f"Target columns cannot be model inputs: {leaked}")
    if config.forecast_horizons:
        prepared = with_forecast_horizons(prepared, config)
    fit_df, validation_df = chronological_validation_split(
        prepared.train_df,
        config,
    )
    experiment = ExperimentData(
        config=config,
        prepared=prepared,
        fit_df=fit_df,
        validation_df=validation_df,
        test_df=prepared.test_df,
    )
    check_embedding_precedes_validation(experiment)
    return experiment


def check_embedding_precedes_validation(experiment: ExperimentData) -> None:
    """Reject splits whose validation periods the graph encoder learned from.

    Validating a downstream model on labels the encoder was fitted to would
    make every candidate using the embedding look better than it is.
    """
    fit_end = getattr(experiment.prepared, "embedding_fit_end", None)
    if fit_end is None:
        return
    earliest = min(
        validation["settlement_date"].min()
        for _, validation in walk_forward_splits(experiment)
    )
    # fit_end bounds realised outcomes used for both fitting and checkpoint
    # selection. Subtracting the horizon would allow information from validation.
    if fit_end > earliest:
        raise ValueError(
            f"The graph encoder's information cutoff ({fit_end}) is "
            f"after the earliest validation block starts ({earliest}). Lower "
            "GraphEmbeddingConfig.fit_fraction or use fewer walk-forward folds."
        )


def with_forecast_horizons(prepared: PreparedModelData, config: ExperimentConfig) -> PreparedModelData:
    """Add exact-time labels after feature preparation, preserving learned features.

    Full timelines retain unlabeled rows as sequence context. Only supervised
    frames require every horizon, so all models are compared on the same origins.
    No fitted transforms or graph-encoder metadata are replaced here.
    """
    if prepared.train_timeline is None or prepared.test_timeline is None:
        raise ValueError("Multi-output experiments require full timelines; rerun with --refresh")
    test_start = prepared.test_timeline["settlement_date"].min()
    if prepared.train_timeline["settlement_date"].max() >= test_start:
        raise ValueError("Training and test timelines overlap")
    feature_config = replace(config.feature_config, target_horizons_minutes=config.horizons)
    targets = target_columns(config.horizons)
    timelines, supervised = [], []
    for is_training, timeline in ((True, prepared.train_timeline), (False, prepared.test_timeline)):
        if timeline.duplicated(["region", "settlement_date"]).any():
            raise ValueError("Duplicate region/timestamp rows make target alignment ambiguous")
        frame = add_targets(timeline, feature_config)
        for horizon, column in zip(config.horizons, targets):
            price = frame[f"target_rrp_h{horizon:02d}"]
            if config.target_type == "price":
                frame[column] = price
            else:
                base = "rrp" if config.target_type == "change" else f"p5_h{horizon:02d}_rrp"
                if base not in frame:
                    raise ValueError(f"Missing forecast input {base}; prepare all requested P5 horizons")
                frame[column] = price - frame[base]
        frame[targets] = frame[targets].replace([np.inf, -np.inf], np.nan)
        frame.loc[~np.isfinite(frame["rrp"]), targets] = np.nan
        if is_training:
            # The entire vector is embargoed at the latest forecast outcome.
            unresolved = frame["settlement_date"] + pd.Timedelta(minutes=config.horizon_minutes) >= test_start
            frame.loc[unresolved, targets] = np.nan
        frame["target"] = frame[targets[-1]]
        frame["target_rrp"] = frame[f"target_rrp_h{config.horizon_minutes:02d}"]
        rows = frame.dropna(subset=targets).reset_index(drop=True)
        if rows.empty:
            raise ValueError("No complete multi-horizon targets remain in a split")
        timelines.append(frame)
        supervised.append(rows)
    train, test = supervised
    return replace(
        prepared, forecast_horizons=config.forecast_horizons,
        train_timeline=timelines[0], test_timeline=timelines[1], train_df=train, test_df=test,
        X_train=train[prepared.features].copy(), X_test=test[prepared.features].copy(),
        y_train=train[targets].copy(), y_test=test[targets].copy(),
    )


def walk_forward_splits(experiment: ExperimentData):
    """Return expanding fit periods with disjoint validation blocks and label embargoes.

    Blocks are sized by eligible row count, not calendar duration. Later folds
    may train on earlier validation periods, as in periodic real-world retraining;
    the held-out test period is never included in these folds.
    """
    count = int(experiment.config.walk_forward_folds)
    if count == 1:
        return [(experiment.fit_df, experiment.validation_df)]
    frame = experiment.prepared.train_df
    width = len(frame) - int(len(frame) * (1 - experiment.config.validation_fraction))
    initial = len(frame) - count * width
    if initial <= 0:
        raise ValueError("Too many walk-forward folds for validation_fraction")
    folds = []
    for position in range(count):
        start = initial + position * width
        validation = frame.iloc[start : start + width].copy()
        cutoff = validation["settlement_date"].min()
        fit = frame.loc[
            frame["settlement_date"]
            + pd.Timedelta(minutes=experiment.config.horizon_minutes)
            < cutoff
        ].copy()
        if fit.empty:
            raise ValueError("Walk-forward fold has no eligible training rows")
        folds.append((fit, validation))
    return folds


def complete_case_rows(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return rows usable by models that cannot accept missing numeric inputs."""
    return frame.dropna(subset=[*feature_columns, "target"]).copy()


def reconstruct_price_forecast(
    frame: pd.DataFrame,
    predicted_target: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Convert target-scale model output into an RRP forecast."""
    predicted_target = forecast_array(predicted_target, len(frame), len(config.horizons))
    if config.target_type == "price":
        return predicted_target
    if config.target_type == "change":
        base = frame["rrp"].to_numpy()
        if config.forecast_horizons:
            base = base[:, None]
    else:
        columns = [f"p5_h{h:02d}_rrp" for h in config.horizons]
        base = frame[columns].to_numpy() if config.forecast_horizons else frame[columns[0]].to_numpy()
    return base + predicted_target


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
        "mse": float(mean_squared_error(actual, predicted)),
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
        ("mse", "MSE"),
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
    print(f"{model_name} MAE improvement over {baseline_name}: {improvement:.2f}%")


def print_experiment_summary(experiment: ExperimentData) -> None:
    """Report the prepared sample, features, and chronological split."""
    data = experiment.prepared
    config = experiment.config
    train_df = data.train_df
    test_df = data.test_df

    print("\nPREPARED MODEL DATA")
    print("-------------------")
    print("Target region:", data.region)
    print("Forecast horizons:", config.horizons, "minutes")
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

    if data.embedding_columns:
        print("Graph embedding features:", len(data.embedding_columns))
        print("Graph encoder fitted before:", data.embedding_fit_end)

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
        print(
            f"{'Persistence':<12} ${mean_absolute_error(subset['actual'], subset['baseline']):.2f}"
        )
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
    print(diagnostics.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))
