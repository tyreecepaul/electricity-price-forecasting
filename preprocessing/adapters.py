from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

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
        Selected target. By default this is future price change in $/MWh.

    train_df, test_df:
        Full prepared rows, including ``settlement_date``, ``region``, ``rrp``,
        ``target_rrp`` and a standard ``target`` column. These are useful for
        diagnostics and sequence models.

    features:
        Exact feature names used in X_train/X_test.
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
    target: Literal["change", "price"] = "change",
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

    if target not in {"change", "price"}:
        raise ValueError("target must be either 'change' or 'price'")

    if config is None:
        config = FeatureConfig(
            target_horizons_minutes=(horizon,),
        )
    else:
        config = replace(
            config,
            target_horizons_minutes=(horizon,),
        )

    if train_data is None or test_data is None:
        if train_data is not None or test_data is not None:
            raise ValueError(
                "Provide both train_data and test_data, or provide neither."
            )

        # Lazy import keeps the lower-level preprocessing modules independent of
        # the ingestion implementation until this convenience function is used.
        from data_cleaning.data import getTrainTestData

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
                column in test_features.columns
                and train_features[column].notna().any()
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

    train_df = (
        train_features.dropna(subset=required_rows)
        .sort_values("settlement_date")
        .reset_index(drop=True)
        .copy()
    )
    test_df = (
        test_features.dropna(subset=required_rows)
        .sort_values("settlement_date")
        .reset_index(drop=True)
        .copy()
    )

    if train_df.empty:
        raise ValueError("No training rows remain after target filtering")
    if test_df.empty:
        raise ValueError("No test rows remain after target filtering")

    if target == "change":
        train_df["target"] = train_df["target_rrp"] - train_df["rrp"]
        test_df["target"] = test_df["target_rrp"] - test_df["rrp"]
    else:
        train_df["target"] = train_df["target_rrp"]
        test_df["target"] = test_df["target_rrp"]

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
    )
