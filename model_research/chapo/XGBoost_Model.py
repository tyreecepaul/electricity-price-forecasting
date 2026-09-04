"""Train and evaluate XGBoost on the shared dispatch-price experiment data."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_research.common.experiment_pipeline import (
    ExperimentConfig,
    build_results_frame,
    evaluate_aemo_rows,
    print_experiment_summary,
    print_improvement,
    print_metrics,
    print_price_regime_results,
    print_target_diagnostics,
    prepare_experiment,
    reconstruct_price_forecast,
)


# Experiment configuration shared with any future model implementation

EXPERIMENT = ExperimentConfig(
    region="NSW1",
    horizon_minutes=60,
    target_type="change",
    model_features="all",
    validation_fraction=0.15,
)


# XGBoost specification

XGBOOST_PARAMETERS = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 8,
    "reg_alpha": 0.1,
    "reg_lambda": 8.0,
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "max_bin": 256,
    "n_jobs": -1,
    "random_state": 42,
}


# Shared data preparation and chronological validation

experiment = prepare_experiment(EXPERIMENT)
data = experiment.prepared
features = experiment.features
fit_df = experiment.fit_df
validation_df = experiment.validation_df
test_df = data.test_df

print_experiment_summary(experiment)

X_fit = fit_df[features]
y_fit = fit_df["target"]
X_validation = validation_df[features]
y_validation = validation_df["target"]


# Select the number of boosting rounds by early stopping

selection_model = XGBRegressor(
    n_estimators=2000,
    early_stopping_rounds=100,
    **XGBOOST_PARAMETERS,
)

print("\nStarting model-selection XGBoost training...")
start_time = time.perf_counter()
selection_model.fit(
    X_fit,
    y_fit,
    eval_set=[(X_validation, y_validation)],
    verbose=50,
)
selection_seconds = time.perf_counter() - start_time

best_iteration = int(selection_model.best_iteration)
best_n_estimators = best_iteration + 1

print(f"\nModel-selection training finished in {selection_seconds:.1f} seconds")
print("Best boosting iteration:", best_iteration)
print("Trees used in final model:", best_n_estimators)


# Evaluate model-selection forecasts

validation_predicted_target = selection_model.predict(X_validation)
validation_predictions = reconstruct_price_forecast(
    validation_df,
    validation_predicted_target,
    EXPERIMENT,
)
actual_validation = validation_df["target_rrp"].to_numpy()

validation_metrics = print_metrics(
    "VALIDATION RESULTS",
    actual_validation,
    {
        "XGBoost": validation_predictions,
        "Persistence": validation_df["rrp"].to_numpy(),
    },
)
print_improvement(
    "XGBoost",
    validation_metrics["XGBoost"]["mae"],
    "Persistence",
    validation_metrics["Persistence"]["mae"],
)

evaluate_aemo_rows(
    "VALIDATION: PAIRED XGBOOST VS AEMO",
    validation_df,
    validation_predictions,
    EXPERIMENT,
    model_name="XGBoost",
    include_persistence=False,
)


# Refit the selected specification on all pre-test observations

final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    **XGBOOST_PARAMETERS,
)

print("\nRefitting final XGBoost on all pre-test training data...")
start_time = time.perf_counter()
final_model.fit(data.X_train, data.y_train, verbose=False)
refit_seconds = time.perf_counter() - start_time
print(f"Final refit finished in {refit_seconds:.1f} seconds")


# Evaluate final test forecasts

predicted_target = final_model.predict(data.X_test)
predictions = reconstruct_price_forecast(
    test_df,
    predicted_target,
    EXPERIMENT,
)
actual_test = test_df["target_rrp"].to_numpy()

test_metrics = print_metrics(
    "FINAL TEST RESULTS",
    actual_test,
    {
        "XGBoost": predictions,
        "Persistence": test_df["rrp"].to_numpy(),
    },
)
print_improvement(
    "XGBoost",
    test_metrics["XGBoost"]["mae"],
    "Persistence",
    test_metrics["Persistence"]["mae"],
)

evaluate_aemo_rows(
    "FINAL TEST: PAIRED XGBOOST VS AEMO",
    test_df,
    predictions,
    EXPERIMENT,
    model_name="XGBoost",
    include_persistence=True,
)


# Interval-level and regime diagnostics

results = build_results_frame(
    test_df,
    predictions,
    predicted_target,
    EXPERIMENT,
)
results = results.rename(columns={"model_error": "xgb_error"})

print("\nFIRST 20 PREDICTIONS")
print("--------------------")
pd.set_option("display.max_columns", None)
print(results.head(20))

print("\nLARGEST XGBOOST ERRORS")
print("----------------------")
print(results.sort_values("xgb_error", ascending=False).head(20))

print_price_regime_results(results, model_name="XGBoost")
print_target_diagnostics(data.y_test.to_numpy(), predicted_target, EXPERIMENT)


# XGBoost-specific feature importance

importance = (
    pd.DataFrame(
        {
            "feature": features,
            "importance": final_model.feature_importances_,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nFEATURE IMPORTANCE")
print("------------------")
print(importance.to_string(index=False))
