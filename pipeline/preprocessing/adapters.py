from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from .config import FeatureConfig
from .features import (
    build_features,
    canonicalise_columns,
    essential_feature_columns,
    join_actual_tables,
)


@dataclass
class PreparedModelData:
    """Standard train/test output shared by classical and deep-learning models.

    Attributes
    ----------
    X_train, X_test:
        Selected model input columns.

    y_train, y_test:
        Selected target. By default this is future price change in $/MWh;
        ``price`` targets contain the direct future RRP.
        ``aemo_error`` targets contain actual future RRP minus the aligned AEMO
        P5MIN dispatch-price forecast at the requested horizon.

    train_df, test_df:
        Full prepared rows, including ``settlement_date``, ``region``, ``rrp``,
        ``target_rrp`` and a standard ``target`` column. These are useful for
        diagnostics and sequence models.

    features:
        Exact feature names used in X_train/X_test.

    train_timeline, test_timeline:
        Complete feature timelines, including observations with unavailable
        targets. Sequence models use these frames as history and apply label
        masks; train_df/test_df remain the eligible supervised/scoring rows.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    features: list[str]
    region: str
    horizon: int
    target_type: str
    train_timeline: pd.DataFrame | None = None
    test_timeline: pd.DataFrame | None = None


def extract_actuals_and_p5min(
    data,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Adapt one ingestion-layer Data object to preprocessing inputs."""

    p5min = getattr(data, "p5min", None)

    if hasattr(data, "prices") and hasattr(data, "region_summary"):
        actuals = join_actual_tables(
            data.prices,
            data.region_summary,
            getattr(data, "interconnectors", None),
        )
    else:
        actuals = canonicalise_columns(data.df.copy())

    required = {
        "settlement_date",
        "region",
        "rrp",
        "totaldemand",
        "availablegeneration",
    }

    missing = sorted(required - set(actuals.columns))
    if missing:
        raise ValueError(
            "The ingestion layer does not provide the minimum actuals required "
            f"by preprocessing. Missing: {missing}"
        )

    return actuals, p5min


def prepare_model_data(
    *,
    region: str = "NSW1",
    horizon: int = 30,
    features: Literal["all"] | Sequence[str] = "all",
    target: Literal["change", "price", "aemo_error"] = "change",
    config: FeatureConfig | None = None,
    train_data=None,
    test_data=None,
) -> PreparedModelData:
    """Build the common leakage-safe train/test data used by any model.

    This is the main convenience function most modelling scripts should use.

    Parameters
    ----------
    region:
        NEM region to model, e.g. ``"NSW1"``.

    horizon:
        Forecast horizon in minutes. The future target is matched by exact
        timestamp, not by row offset.

    features:
        ``"all"`` uses the standard feature set from
        :func:`essential_feature_columns`.

        A list/tuple selects an explicit experiment-specific subset, e.g.::

            features=[
                "rrp",
                "rrp_delta_1",
                "reserve_ratio",
                "p5_h60_rrp_vs_now",
            ]

    target:
        ``"change"`` (default): ``future RRP - current RRP``.
        ``"price"``: direct future RRP.
        ``"aemo_error"``: ``future RRP - AEMO P5MIN forecast RRP``. Rows
        without an aligned AEMO forecast at ``horizon`` are removed.

    config:
        Optional FeatureConfig override. ``horizon`` remains authoritative and
        is inserted into ``target_horizons_minutes`` automatically.

    train_data, test_data:
        Optional already-loaded ingestion objects. If omitted, the function
        calls ``data_cleaning.data.getTrainTestData()`` itself.

    Returns
    -------
    PreparedModelData
        X/y matrices, full prepared train/test DataFrames and selected feature
        names. The full frames include a standard ``target`` column so they can
        also be passed to sequence-model helpers.
    """

    if horizon <= 0:
        raise ValueError("horizon must be a positive number of minutes")

    if target not in {"change", "price", "aemo_error"}:
        raise ValueError("target must be one of 'change', 'price', or 'aemo_error'")

    if config is None:
        config = FeatureConfig(
            target_horizons_minutes=(horizon,),
        )
    else:
        config = replace(
            config,
            target_horizons_minutes=(horizon,),
        )

    if target == "aemo_error" and horizon not in config.p5min_horizons_minutes:
        config = replace(
            config,
            p5min_horizons_minutes=tuple(
                sorted({*config.p5min_horizons_minutes, horizon})
            ),
        )

    if train_data is None or test_data is None:
        if train_data is not None or test_data is not None:
            raise ValueError(
                "Provide both train_data and test_data, or provide neither."
            )

        # Lazy import keeps the lower-level preprocessing modules independent of
        # the ingestion implementation until this convenience function is used.
        from pipeline.data_cleaning.data import getTrainTestData

        train_data, test_data = getTrainTestData()

    train_actuals, train_p5min = extract_actuals_and_p5min(train_data)
    test_actuals, test_p5min = extract_actuals_and_p5min(test_data)

    train_features = build_features(
        train_actuals,
        p5min=train_p5min,
        config=config,
    )

    test_features = build_features(
        test_actuals,
        p5min=test_p5min,
        p5min_history=train_p5min,
        history=train_actuals,
        config=config,
    )

    train_features = (
        train_features.loc[train_features["region"] == region]
        .sort_values("settlement_date")
        .reset_index(drop=True)
    )
    test_features = (
        test_features.loc[test_features["region"] == region]
        .sort_values("settlement_date")
        .reset_index(drop=True)
    )

    if train_features.empty:
        raise ValueError(f"No training rows found for region {region!r}")
    if test_features.empty:
        raise ValueError(f"No test rows found for region {region!r}")

    if features == "all":
        selected_features = essential_feature_columns(train_features, config)

        # Standard mode is forgiving: keep features available in both splits and
        # automatically remove columns that are entirely missing in training.
        selected_features = [
            column
            for column in selected_features
            if (
                column in test_features.columns and train_features[column].notna().any()
            )
        ]
    elif isinstance(features, Sequence) and not isinstance(features, str):
        selected_features = list(dict.fromkeys(features))

        missing_train = [
            column
            for column in selected_features
            if column not in train_features.columns
        ]
        missing_test = [
            column
            for column in selected_features
            if column not in test_features.columns
        ]
        empty_train = [
            column
            for column in selected_features
            if (
                column in train_features.columns
                and not train_features[column].notna().any()
            )
        ]

        problems: list[str] = []
        if missing_train:
            problems.append(f"missing from training: {missing_train}")
        if missing_test:
            problems.append(f"missing from test: {missing_test}")
        if empty_train:
            problems.append(f"entirely empty in training: {empty_train}")

        if problems:
            raise ValueError(
                "Requested feature set is not usable: " + "; ".join(problems)
            )
    else:
        raise TypeError("features must be 'all' or a sequence of column names")

    if not selected_features:
        raise ValueError("No usable model features were selected")

    # Do not require every feature to be non-null. XGBoost can use missing values,
    # while linear/deep models can apply their own imputation/scaling policy.
    required_rows = ["target_rrp", "rrp"]
    aemo_forecast_column = f"p5_h{horizon:02d}_rrp"
    if target == "aemo_error":
        if aemo_forecast_column not in train_features.columns:
            raise ValueError(
                "AEMO-error target requires the aligned P5MIN forecast column "
                f"{aemo_forecast_column!r} in training data"
            )
        if aemo_forecast_column not in test_features.columns:
            raise ValueError(
                "AEMO-error target requires the aligned P5MIN forecast column "
                f"{aemo_forecast_column!r} in test data"
            )
        required_rows.append(aemo_forecast_column)

    # Keep unlabeled observations for sequence context, including the final
    # horizon of training. Eligibility is separate from the feature timeline.
    train_df = train_features.copy()
    test_df = test_features.copy()

    if train_df.empty:
        raise ValueError("No training rows remain after target filtering")
    if test_df.empty:
        raise ValueError("No test rows remain after target filtering")

    if target == "change":
        train_df["target"] = train_df["target_rrp"] - train_df["rrp"]
        test_df["target"] = test_df["target_rrp"] - test_df["rrp"]
    elif target == "aemo_error":
        train_df["target"] = train_df["target_rrp"] - train_df[aemo_forecast_column]
        test_df["target"] = test_df["target_rrp"] - test_df[aemo_forecast_column]
    else:
        train_df["target"] = train_df["target_rrp"]
        test_df["target"] = test_df["target_rrp"]

    train_df.loc[train_df[required_rows].isna().any(axis=1), "target"] = float("nan")
    test_df.loc[test_df[required_rows].isna().any(axis=1), "target"] = float("nan")
    train_timeline, test_timeline = train_df.copy(), test_df.copy()
    train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["target"]).reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("No rows remain after target filtering")

    X_train = train_df[selected_features].copy()
    X_test = test_df[selected_features].copy()
    y_train = train_df["target"].copy()
    y_test = test_df["target"].copy()

    y_train.name = "target"
    y_test.name = "target"

    return PreparedModelData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
        features=selected_features,
        region=region,
        horizon=horizon,
        target_type=target,
        train_timeline=train_timeline,
        test_timeline=test_timeline,
    )
