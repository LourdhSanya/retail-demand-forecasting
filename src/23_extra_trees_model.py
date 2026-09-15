import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_FILE = "data/processed/product_21212_features.csv"


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# 2. FEATURES
# --------------------------------------------------

features = [
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "Quarter",
    "WeekOfYear",
    "IsWeekend",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_28"
]

target = "Demand"


# --------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]

y_train = df[target].iloc[:split_index]
y_test = df[target].iloc[split_index:]


print("=" * 70)
print("EXTRA TREES DEMAND FORECASTING")
print("=" * 70)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# --------------------------------------------------
# 4. CREATE MODEL
# --------------------------------------------------

model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. TRAIN
# --------------------------------------------------

print("\nTraining Extra Trees model...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)


# --------------------------------------------------
# 7. EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

wape = (
    np.abs(y_test - predictions).sum()
    / y_test.sum()
    * 100
)


# --------------------------------------------------
# 8. RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXTRA TREES PERFORMANCE")
print("=" * 70)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")


# --------------------------------------------------
# 9. COMPARISON
# --------------------------------------------------

old_rf_mae = 128.41
old_rf_rmse = 228.56
old_rf_wape = 64.24

print("\n" + "=" * 70)
print("RANDOM FOREST vs EXTRA TREES")
print("=" * 70)

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "WAPE"],
    "Random Forest": [
        old_rf_mae,
        old_rf_rmse,
        old_rf_wape
    ],
    "Extra Trees": [
        mae,
        rmse,
        wape
    ]
})

print(
    comparison.to_string(index=False)
)


# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# --------------------------------------------------
# 11. SAMPLE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "Date": df.iloc[split_index:]["Date"].values,
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

print(
    results.head(15).to_string(index=False)
)


print("\nExtra Trees modeling completed successfully!")