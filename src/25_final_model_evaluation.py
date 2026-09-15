import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 1. LOAD DATA
# ============================================================

INPUT_FILE = "data/processed/product_21212_features.csv"

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 2. FEATURES
# ============================================================

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


# ============================================================
# 3. TIME-BASED SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]

y_train = df[target].iloc[:split_index]
y_test = df[target].iloc[split_index:]


print("=" * 70)
print("FINAL RANDOM FOREST MODEL EVALUATION")
print("=" * 70)

print("\nTraining period:")
print(
    X_train.index.min(),
    "to",
    X_train.index.max()
)

print("\nTesting period:")
print(
    X_test.index.min(),
    "to",
    X_test.index.max()
)


# ============================================================
# 4. CREATE FINAL MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining final model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 6. PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)


# ============================================================
# 7. CALCULATE ERRORS
# ============================================================

errors = y_test.values - predictions


# ============================================================
# 8. METRICS
# ============================================================

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
    np.abs(errors).sum()
    / y_test.sum()
    * 100
)


print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")


# ============================================================
# 9. ERROR ANALYSIS
# ============================================================

results = pd.DataFrame({
    "Date": df.iloc[split_index:]["Date"].values,
    "Actual": y_test.values,
    "Predicted": predictions,
    "Error": errors
})

results["Absolute_Error"] = np.abs(
    results["Error"]
)

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    f"\nAverage Error: {results['Error'].mean():.2f}"
)

if results["Error"].mean() > 0:
    print("Model tendency: UNDER-PREDICTION")
else:
    print("Model tendency: OVER-PREDICTION")


# ============================================================
# 10. LARGEST ERRORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 LARGEST FORECAST ERRORS")
print("=" * 70)

largest_errors = results.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print(
    largest_errors[
        [
            "Date",
            "Actual",
            "Predicted",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FINAL FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

OUTPUT_FILE = "data/processed/final_predictions.csv"

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nPredictions saved to: {OUTPUT_FILE}"
)


# ============================================================
# 13. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    results["Date"],
    results["Actual"],
    label="Actual Demand"
)

plt.plot(
    results["Date"],
    results["Predicted"],
    label="Predicted Demand"
)

plt.title(
    "Actual vs Predicted Demand - Product 21212"
)

plt.xlabel("Date")
plt.ylabel("Demand")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "reports/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    results["Error"],
    bins=20
)

plt.title(
    "Forecast Error Distribution"
)

plt.xlabel(
    "Forecast Error (Actual - Predicted)"
)

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "reports/error_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 70)