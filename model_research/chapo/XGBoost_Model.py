from data_cleaning.data import getTrainingData, getTestData

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


# ============================================================
# LOAD TRAINING AND TEST DATA
# ============================================================

train_data = getTrainingData()
test_data = getTestData()

train_df = train_data.df.copy()
test_df = test_data.df.copy()


print("Training data:")
print(
    train_df["settlement_date"].min(),
    "->",
    train_df["settlement_date"].max()
)

print("Test data:")
print(
    test_df["settlement_date"].min(),
    "->",
    test_df["settlement_date"].max()
)

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))


# ============================================================
# MARK TRAIN / TEST BEFORE COMBINING
# ============================================================

train_df["dataset"] = "train"
test_df["dataset"] = "test"

# Combine temporarily so test lag features can use
# price history from the end of the training set.
df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

df["settlement_date"] = pd.to_datetime(
    df["settlement_date"]
)

df = (
    df.sort_values(
        ["region", "settlement_date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# PRICE LAG FEATURES
# ============================================================

grouped_price = df.groupby("region")["rrp"]


df["rrp_lag_1"] = grouped_price.shift(1)      # 5 min ago

df["rrp_lag_2"] = grouped_price.shift(2)      # 10 min ago

df["rrp_lag_3"] = grouped_price.shift(3)      # 15 min ago

df["rrp_lag_6"] = grouped_price.shift(6)      # 30 min ago

df["rrp_lag_12"] = grouped_price.shift(12)    # 1 hour ago


# ============================================================
# PRICE CHANGE FEATURES
# ============================================================

df["rrp_change_1"] = grouped_price.diff(1)

df["rrp_change_3"] = grouped_price.diff(3)

df["rrp_change_12"] = grouped_price.diff(12)


# ============================================================
# ROLLING PRICE FEATURES
# ============================================================

df["rrp_mean_12"] = (
    df.groupby("region")["rrp"]
    .transform(
        lambda x: x.rolling(12).mean()
    )
)

df["rrp_std_12"] = (
    df.groupby("region")["rrp"]
    .transform(
        lambda x: x.rolling(12).std()
    )
)

df["rrp_min_12"] = (
    df.groupby("region")["rrp"]
    .transform(
        lambda x: x.rolling(12).min()
    )
)

df["rrp_max_12"] = (
    df.groupby("region")["rrp"]
    .transform(
        lambda x: x.rolling(12).max()
    )
)


# ============================================================
# PRICE REGIME FEATURES
# ============================================================

df["price_above_300"] = (
    df["rrp"] > 300
).astype(int)

df["price_above_1000"] = (
    df["rrp"] > 1000
).astype(int)

df["price_below_0"] = (
    df["rrp"] < 0
).astype(int)


# ============================================================
# TIME FEATURES
# ============================================================

df["hour"] = (
    df["settlement_date"].dt.hour
)

df["minute"] = (
    df["settlement_date"].dt.minute
)

df["day_of_week"] = (
    df["settlement_date"].dt.dayofweek
)


# ============================================================
# TARGET
# ============================================================
#
# We predict:
#
#     next price - current price
#
# rather than directly predicting next price.
#
# ============================================================

df["next_rrp"] = (
    df.groupby("region")["rrp"]
    .shift(-1)
)

df["target"] = (
    df["next_rrp"] - df["rrp"]
)


# ============================================================
# PREVENT TRAIN/TEST TARGET CROSSOVER
# ============================================================
#
# Important:
#
# The final training row should NOT use the first test price
# as its target.
#
# Similarly, the final test row has no next price.
#
# ============================================================

df["next_dataset"] = (
    df.groupby("region")["dataset"]
    .shift(-1)
)


# ============================================================
# FEATURES USED BY XGBOOST
# ============================================================

features = [

    # Current price
    "rrp",

    # Price lags
    "rrp_lag_1",
    "rrp_lag_2",
    "rrp_lag_3",
    "rrp_lag_6",
    "rrp_lag_12",

    # Price changes
    "rrp_change_1",
    "rrp_change_3",
    "rrp_change_12",

    # Rolling price statistics
    "rrp_mean_12",
    "rrp_std_12",
    "rrp_min_12",
    "rrp_max_12",

    # Current-price regimes
    "price_above_300",
    "price_above_1000",
    "price_below_0",

    # Time
    "hour",
    "minute",
    "day_of_week",
]


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

required_columns = (
    features
    + ["target", "next_rrp"]
)

df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# SPLIT BACK INTO TRAIN AND TEST
# ============================================================
#
# Only allow targets that stay within the same dataset.
#
# ============================================================

model_train_df = df[
    (df["dataset"] == "train")
    & (df["next_dataset"] == "train")
].copy()


model_test_df = df[
    (df["dataset"] == "test")
    & (df["next_dataset"] == "test")
].copy()


print("\nAfter feature engineering:")

print(
    "Training rows:",
    len(model_train_df)
)

print(
    "Test rows:",
    len(model_test_df)
)

print(
    "\nModel training dates:",
    model_train_df["settlement_date"].min(),
    "->",
    model_train_df["settlement_date"].max()
)

print(
    "Model test dates:",
    model_test_df["settlement_date"].min(),
    "->",
    model_test_df["settlement_date"].max()
)


# ============================================================
# CREATE X AND Y
# ============================================================

X_train = model_train_df[features]

y_train = model_train_df["target"]


X_test = model_test_df[features]

y_test = model_test_df["target"]


# Actual next electricity price
actual_test = model_test_df["next_rrp"]


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=5,
    objective="reg:squarederror",
    random_state=42,
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================
#
# Model predicts the CHANGE in price.
#
# ============================================================

predicted_change = model.predict(
    X_test
)


# Convert predicted change back to price:
#
# predicted next price =
# current price + predicted change

predictions = (
    X_test["rrp"].to_numpy()
    + predicted_change
)


# ============================================================
# PERSISTENCE BASELINE
# ============================================================
#
# Baseline simply predicts:
#
#     next price = current price
#
# ============================================================

baseline_predictions = (
    X_test["rrp"].to_numpy()
)


# ============================================================
# OVERALL METRICS
# ============================================================

xgb_mae = mean_absolute_error(
    actual_test,
    predictions
)

xgb_rmse = (
    mean_squared_error(
        actual_test,
        predictions
    ) ** 0.5
)

xgb_median_ae = (
    median_absolute_error(
        actual_test,
        predictions
    )
)


baseline_mae = mean_absolute_error(
    actual_test,
    baseline_predictions
)

baseline_rmse = (
    mean_squared_error(
        actual_test,
        baseline_predictions
    ) ** 0.5
)

baseline_median_ae = (
    median_absolute_error(
        actual_test,
        baseline_predictions
    )
)


print("\nOVERALL RESULTS")
print("-------------------------")

print(
    f"XGBoost MAE:        "
    f"{xgb_mae:.2f}"
)

print(
    f"Baseline MAE:       "
    f"{baseline_mae:.2f}"
)

print()

print(
    f"XGBoost RMSE:       "
    f"{xgb_rmse:.2f}"
)

print(
    f"Baseline RMSE:      "
    f"{baseline_rmse:.2f}"
)

print()

print(
    f"XGBoost Median AE:  "
    f"{xgb_median_ae:.2f}"
)

print(
    f"Baseline Median AE: "
    f"{baseline_median_ae:.2f}"
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({

    "settlement_date":
        model_test_df["settlement_date"].values,

    "current":
        X_test["rrp"].to_numpy(),

    "actual":
        actual_test.to_numpy(),

    "predicted_change":
        predicted_change,

    "predicted":
        predictions,

    "baseline":
        baseline_predictions,
})


results["xgb_error"] = abs(
    results["actual"]
    - results["predicted"]
)

results["baseline_error"] = abs(
    results["actual"]
    - results["baseline"]
)


# ============================================================
# FIRST 20 PREDICTIONS
# ============================================================

print("\nFIRST 20 PREDICTIONS")
print("-------------------------")

pd.set_option("display.max_columns", None)

print(results.head(20))

# ============================================================
# LARGEST XGBOOST ERRORS
# ============================================================

print("\nLARGEST XGBOOST ERRORS")
print("-------------------------")

print(
    results
    .sort_values(
        "xgb_error",
        ascending=False
    )
    .head(20)
)


# ============================================================
# RESULTS BY PRICE REGIME
# ============================================================

bands = [

    (
        "Normal (< $300)",
        results["actual"] < 300
    ),

    (
        "High ($300-$1000)",
        (
            (results["actual"] >= 300)
            & (results["actual"] < 1000)
        )
    ),

    (
        "Spike (> $1000)",
        results["actual"] >= 1000
    ),
]


print("\nRESULTS BY PRICE REGIME")
print("-------------------------")


for name, mask in bands:

    subset = results[mask]

    if len(subset) == 0:
        continue

    regime_xgb_mae = (
        mean_absolute_error(
            subset["actual"],
            subset["predicted"]
        )
    )

    regime_baseline_mae = (
        mean_absolute_error(
            subset["actual"],
            subset["baseline"]
        )
    )

    print(f"\n{name}")

    print(
        f"Observations: "
        f"{len(subset)}"
    )

    print(
        f"XGBoost MAE:  "
        f"${regime_xgb_mae:.2f}"
    )

    print(
        f"Baseline MAE: "
        f"${regime_baseline_mae:.2f}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        features,

    "importance":
        model.feature_importances_,
})


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\nFEATURE IMPORTANCE")
print("-------------------------")

print(
    importance.to_string(
        index=False
    )
)