"""
XGBoost NEM price forecasting model using the shared preprocessing API.

The preprocessing package handles:
    - loading the train/test split
    - adapting ingestion objects
    - joining actual AEMO tables
    - leakage-safe P5MIN alignment
    - feature engineering
    - region filtering
    - target construction
    - default feature selection
    - removal of completely unavailable default features

This script is therefore responsible only for:
    - chronological validation
    - XGBoost fitting / early stopping
    - final refit
    - evaluation and diagnostics

Target:
    future_RRP - current_RRP

Forecast:
    current_RRP + predicted_change
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from xgboost import XGBRegressor


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import prepare_model_data


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_REGION = "NSW1"
TARGET_HORIZON_MINUTES = 30
VALIDATION_FRACTION = 0.15

# "all" uses the standard feature set from preprocessing.
# You can instead provide an explicit list, e.g.
#
# MODEL_FEATURES = [
#     "rrp",
#     "rrp_delta_1",
#     "reserve_ratio",
#     "p5_h60_rrp_vs_now",
# ]
MODEL_FEATURES = "all"

AEMO_FORECAST_COLUMN = (
    f"p5_h{TARGET_HORIZON_MINUTES:02d}_rrp"
)


# ============================================================
# METRICS
# ============================================================

def _rmse(actual, predicted) -> float:
    return float(
        mean_squared_error(
            actual,
            predicted,
        ) ** 0.5
    )


def _metrics(actual, predicted) -> dict[str, float]:
    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "rmse": _rmse(
            actual,
            predicted,
        ),
        "median_ae": float(
            median_absolute_error(
                actual,
                predicted,
            )
        ),
    }


def _print_metrics(
    title: str,
    actual,
    forecasts: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    print(f"\n{title}")
    print("-" * len(title))

    values = {
        name: _metrics(actual, pred)
        for name, pred in forecasts.items()
    }

    print("\nMAE")
    for name, metric in values.items():
        print(f"{name:<16} {metric['mae']:.2f}")

    print("\nRMSE")
    for name, metric in values.items():
        print(f"{name:<16} {metric['rmse']:.2f}")

    print("\nMedian AE")
    for name, metric in values.items():
        print(f"{name:<16} {metric['median_ae']:.2f}")

    return values


def _print_improvement(
    model_name: str,
    model_mae: float,
    baseline_name: str,
    baseline_mae: float,
) -> None:
    if baseline_mae == 0:
        return

    improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
        * 100.0
    )

    print(
        f"{model_name} MAE improvement over "
        f"{baseline_name}: {improvement:.2f}%"
    )


# ============================================================
# PREPARE SHARED TRAIN / TEST DATA
# ============================================================

data = prepare_model_data(
    region=TARGET_REGION,
    horizon=TARGET_HORIZON_MINUTES,
    features=MODEL_FEATURES,
    target="change",
)

for feature in data.features:
    print(feature)

features = data.features
train_model_df = data.train_df
test_model_df = data.test_df

print("\nPREPARED MODEL DATA")
print("-------------------")
print("Target region:", data.region)
print(
    "Forecast horizon:",
    data.horizon,
    "minutes",
)
print("Target type:", data.target_type)
print("Training rows:", len(train_model_df))
print("Test rows:", len(test_model_df))
print("Features:", len(features))
print(
    "P5MIN features:",
    sum(
        column.startswith("p5_h")
        for column in features
    ),
)
print(
    "Interconnector features:",
    sum(
        column.startswith("ic_")
        for column in features
    ),
)

print(
    "Training period:",
    train_model_df["settlement_date"].min(),
    "->",
    train_model_df["settlement_date"].max(),
)

print(
    "Test period:",
    test_model_df["settlement_date"].min(),
    "->",
    test_model_df["settlement_date"].max(),
)


# ============================================================
# AEMO FORECAST AVAILABILITY
# ============================================================

print("\nAEMO FORECAST AVAILABILITY")
print("--------------------------")

if AEMO_FORECAST_COLUMN in train_model_df.columns:
    print(
        f"Valid training {AEMO_FORECAST_COLUMN}:",
        train_model_df[
            AEMO_FORECAST_COLUMN
        ].notna().sum(),
    )
else:
    print(
        f"{AEMO_FORECAST_COLUMN} is not present "
        "in the training frame."
    )

if AEMO_FORECAST_COLUMN in test_model_df.columns:
    print(
        f"Valid test {AEMO_FORECAST_COLUMN}:",
        test_model_df[
            AEMO_FORECAST_COLUMN
        ].notna().sum(),
    )
else:
    print(
        f"{AEMO_FORECAST_COLUMN} is not present "
        "in the test frame."
    )


# ============================================================
# CHRONOLOGICAL VALIDATION SPLIT
# ============================================================

if not 0 < VALIDATION_FRACTION < 1:
    raise ValueError(
        "VALIDATION_FRACTION must be between 0 and 1."
    )

split_index = int(
    len(train_model_df)
    * (1 - VALIDATION_FRACTION)
)

if split_index <= 0 or split_index >= len(train_model_df):
    raise RuntimeError(
        "Validation split produced an empty subset."
    )

validation_start = train_model_df.iloc[
    split_index
]["settlement_date"]

target_delta = pd.Timedelta(
    minutes=TARGET_HORIZON_MINUTES
)

# Purge rows whose prediction target lands inside the
# validation period.
fit_model_df = train_model_df[
    (
        train_model_df["settlement_date"]
        + target_delta
    )
    < validation_start
].copy()

validation_model_df = train_model_df[
    train_model_df["settlement_date"]
    >= validation_start
].copy()

if fit_model_df.empty or validation_model_df.empty:
    raise RuntimeError(
        "Chronological validation split produced "
        "an empty subset."
    )

X_fit = fit_model_df[features]
y_fit = fit_model_df["target"]

X_val = validation_model_df[features]
y_val = validation_model_df["target"]

actual_val = validation_model_df[
    "target_rrp"
].to_numpy()

current_val = validation_model_df[
    "rrp"
].to_numpy()

print("\nCHRONOLOGICAL VALIDATION")
print("------------------------")
print(
    "Fit:",
    fit_model_df["settlement_date"].min(),
    "->",
    fit_model_df["settlement_date"].max(),
)
print(
    "Validation:",
    validation_model_df["settlement_date"].min(),
    "->",
    validation_model_df["settlement_date"].max(),
)
print("Fit rows:", len(fit_model_df))
print("Validation rows:", len(validation_model_df))


# ============================================================
# MODEL SELECTION WITH EARLY STOPPING
# ============================================================

selection_model = XGBRegressor(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=8,
    reg_alpha=0.1,
    reg_lambda=8.0,
    objective="reg:absoluteerror",
    eval_metric="mae",
    tree_method="hist",
    max_bin=256,
    n_jobs=-1,
    early_stopping_rounds=100,
    random_state=42,
)

print(
    "\nStarting model-selection XGBoost training..."
)

start_time = time.perf_counter()

selection_model.fit(
    X_fit,
    y_fit,
    eval_set=[
        (
            X_val,
            y_val,
        )
    ],
    verbose=50,
)

selection_seconds = (
    time.perf_counter()
    - start_time
)

best_iteration = int(
    selection_model.best_iteration
)

best_n_estimators = (
    best_iteration
    + 1
)

print(
    "\nModel-selection training finished in "
    f"{selection_seconds:.1f} seconds"
)
print(
    "Best boosting iteration:",
    best_iteration,
)
print(
    "Trees used in final model:",
    best_n_estimators,
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

validation_predicted_change = (
    selection_model.predict(
        X_val
    )
)

validation_predictions = (
    current_val
    + validation_predicted_change
)

persistence_val = current_val

validation_metrics = _print_metrics(
    title="VALIDATION RESULTS",
    actual=actual_val,
    forecasts={
        "XGBoost":
            validation_predictions,
        "Persistence":
            persistence_val,
    },
)

_print_improvement(
    "XGBoost",
    validation_metrics["XGBoost"]["mae"],
    "Persistence",
    validation_metrics["Persistence"]["mae"],
)


# ============================================================
# OPTIONAL VALIDATION AEMO BASELINE
# ============================================================

if AEMO_FORECAST_COLUMN in validation_model_df.columns:
    validation_aemo_mask = (
        validation_model_df[
            AEMO_FORECAST_COLUMN
        ].notna()
    )

    if validation_aemo_mask.any():
        validation_aemo_actual = (
            validation_model_df.loc[
                validation_aemo_mask,
                "target_rrp",
            ].to_numpy()
        )

        validation_aemo_predictions = (
            validation_model_df.loc[
                validation_aemo_mask,
                AEMO_FORECAST_COLUMN,
            ].to_numpy()
        )

        validation_xgb_on_aemo_rows = (
            validation_predictions[
                validation_aemo_mask.to_numpy()
            ]
        )

        validation_aemo_metrics = _print_metrics(
            title="VALIDATION: PAIRED XGBOOST VS AEMO",
            actual=validation_aemo_actual,
            forecasts={
                "XGBoost":
                    validation_xgb_on_aemo_rows,
                "AEMO P5MIN":
                    validation_aemo_predictions,
            },
        )

        _print_improvement(
            "XGBoost",
            validation_aemo_metrics[
                "XGBoost"
            ]["mae"],
            "AEMO P5MIN",
            validation_aemo_metrics[
                "AEMO P5MIN"
            ]["mae"],
        )
    else:
        print(
            f"\nNo usable validation "
            f"{AEMO_FORECAST_COLUMN} forecasts available."
        )


# ============================================================
# REFIT ON ALL PRE-TEST TRAINING DATA
# ============================================================

final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=8,
    reg_alpha=0.1,
    reg_lambda=8.0,
    objective="reg:absoluteerror",
    eval_metric="mae",
    tree_method="hist",
    max_bin=256,
    n_jobs=-1,
    random_state=42,
)

print(
    "\nRefitting final XGBoost on all pre-test "
    "training data..."
)

start_time = time.perf_counter()

final_model.fit(
    data.X_train,
    data.y_train,
    verbose=False,
)

refit_seconds = (
    time.perf_counter()
    - start_time
)

print(
    "Final refit finished in "
    f"{refit_seconds:.1f} seconds"
)


# ============================================================
# FINAL TEST FORECASTS
# ============================================================

actual_test = test_model_df[
    "target_rrp"
].to_numpy()

current_test = test_model_df[
    "rrp"
].to_numpy()

predicted_change = final_model.predict(
    data.X_test
)

predictions = (
    current_test
    + predicted_change
)

baseline_predictions = current_test

actual_change = (
    actual_test
    - current_test
)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

test_metrics = _print_metrics(
    title="FINAL TEST RESULTS",
    actual=actual_test,
    forecasts={
        "XGBoost":
            predictions,
        "Persistence":
            baseline_predictions,
    },
)

_print_improvement(
    "XGBoost",
    test_metrics["XGBoost"]["mae"],
    "Persistence",
    test_metrics["Persistence"]["mae"],
)


# ============================================================
# OPTIONAL PAIRED AEMO TEST COMPARISON
# ============================================================

aemo_predictions_full = np.full(
    len(test_model_df),
    np.nan,
)

if AEMO_FORECAST_COLUMN in test_model_df.columns:
    aemo_mask = (
        test_model_df[
            AEMO_FORECAST_COLUMN
        ].notna()
    )

    if aemo_mask.any():
        aemo_actual = (
            test_model_df.loc[
                aemo_mask,
                "target_rrp",
            ].to_numpy()
        )

        aemo_predictions = (
            test_model_df.loc[
                aemo_mask,
                AEMO_FORECAST_COLUMN,
            ].to_numpy()
        )

        xgb_on_aemo_rows = (
            predictions[
                aemo_mask.to_numpy()
            ]
        )

        persistence_on_aemo_rows = (
            baseline_predictions[
                aemo_mask.to_numpy()
            ]
        )

        aemo_predictions_full[
            aemo_mask.to_numpy()
        ] = aemo_predictions

        paired_metrics = _print_metrics(
            title="FINAL TEST: PAIRED XGBOOST VS AEMO",
            actual=aemo_actual,
            forecasts={
                "XGBoost":
                    xgb_on_aemo_rows,
                "Persistence":
                    persistence_on_aemo_rows,
                "AEMO P5MIN":
                    aemo_predictions,
            },
        )

        _print_improvement(
            "XGBoost",
            paired_metrics["XGBoost"]["mae"],
            "AEMO P5MIN",
            paired_metrics["AEMO P5MIN"]["mae"],
        )

    else:
        print(
            f"\nNo usable test "
            f"{AEMO_FORECAST_COLUMN} forecasts available."
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    {
        "settlement_date":
            test_model_df[
                "settlement_date"
            ].values,

        "region":
            test_model_df[
                "region"
            ].values,

        "current":
            current_test,

        "actual":
            actual_test,

        "actual_change":
            actual_change,

        "predicted_change":
            predicted_change,

        "predicted":
            predictions,

        "baseline":
            baseline_predictions,

        "aemo_p5min":
            aemo_predictions_full,
    }
)

results["xgb_error"] = np.abs(
    results["actual"]
    - results["predicted"]
)

results["baseline_error"] = np.abs(
    results["actual"]
    - results["baseline"]
)

results["aemo_error"] = np.where(
    results["aemo_p5min"].notna(),
    np.abs(
        results["actual"]
        - results["aemo_p5min"]
    ),
    np.nan,
)


# ============================================================
# FIRST 20 PREDICTIONS
# ============================================================

print("\nFIRST 20 PREDICTIONS")
print("--------------------")

pd.set_option(
    "display.max_columns",
    None,
)

print(
    results.head(20)
)


# ============================================================
# LARGEST XGBOOST ERRORS
# ============================================================

print("\nLARGEST XGBOOST ERRORS")
print("----------------------")

print(
    results
    .sort_values(
        "xgb_error",
        ascending=False,
    )
    .head(20)
)


# ============================================================
# RESULTS BY PRICE REGIME
# ============================================================

bands = [
    (
        "Normal (< $300)",
        results["actual"] < 300,
    ),
    (
        "High ($300-$1000)",
        (
            (results["actual"] >= 300)
            & (results["actual"] < 1000)
        ),
    ),
    (
        "Spike (> $1000)",
        results["actual"] >= 1000,
    ),
]

print("\nRESULTS BY PRICE REGIME")
print("-----------------------")

for name, mask in bands:
    subset = results[
        mask
    ]

    if subset.empty:
        continue

    print(f"\n{name}")
    print(
        "Observations:",
        len(subset),
    )

    xgb_mae = mean_absolute_error(
        subset["actual"],
        subset["predicted"],
    )

    baseline_mae = mean_absolute_error(
        subset["actual"],
        subset["baseline"],
    )

    print(
        f"{'XGBoost':<12} "
        f"${xgb_mae:.2f}"
    )

    print(
        f"{'Persistence':<12} "
        f"${baseline_mae:.2f}"
    )

    aemo_subset = subset.dropna(
        subset=["aemo_p5min"]
    )

    if not aemo_subset.empty:
        aemo_mae = mean_absolute_error(
            aemo_subset["actual"],
            aemo_subset["aemo_p5min"],
        )

        print(
            f"{'AEMO P5MIN':<12} "
            f"${aemo_mae:.2f}"
        )


# ============================================================
# PRICE CHANGE DIAGNOSTICS
# ============================================================

print("\nPRICE CHANGE DIAGNOSTICS")
print("------------------------")

diagnostics = pd.DataFrame(
    {
        "actual_change":
            actual_change,

        "predicted_change":
            predicted_change,
    }
)

print(
    diagnostics.describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = (
    pd.DataFrame(
        {
            "feature":
                features,

            "importance":
                final_model.feature_importances_,
        }
    )
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

print("\nFEATURE IMPORTANCE")
print("------------------")

print(
    importance.to_string(
        index=False
    )
)
