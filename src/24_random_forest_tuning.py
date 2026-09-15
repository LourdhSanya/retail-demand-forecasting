import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


X = df[features]
y = df[target]


# ============================================================
# 3. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 4. BASE MODEL
# ============================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. PARAMETERS TO TEST
# ============================================================

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [8, 10, 15, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0]
}


# ============================================================
# 6. TIME SERIES CROSS VALIDATION
# ============================================================

tscv = TimeSeriesSplit(n_splits=3)


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

print("\nTuning completed!")


# ============================================================
# 8. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print("\nBest parameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


print(
    f"\nBest cross-validation MAE: "
    f"{-grid_search.best_score_:.2f}"
)


# ============================================================
# 9. TEST BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

predictions = np.maximum(predictions, 0)


# ============================================================
# 10. FINAL TEST METRICS
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
    np.abs(y_test - predictions).sum()
    / y_test.sum()
    * 100
)


print("\n" + "=" * 70)
print("TUNED MODEL TEST PERFORMANCE")
print("=" * 70)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")


# ============================================================
# 11. COMPARE WITH CURRENT RANDOM FOREST
# ============================================================

old_mae = 128.41
old_rmse = 228.56
old_wape = 64.24


print("\n" + "=" * 70)
print("CURRENT vs TUNED RANDOM FOREST")
print("=" * 70)

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "WAPE"],
    "Current Random Forest": [
        old_mae,
        old_rmse,
        old_wape
    ],
    "Tuned Random Forest": [
        mae,
        rmse,
        wape
    ]
})

print(
    comparison.to_string(index=False)
)


# ============================================================
# 12. MAE IMPROVEMENT
# ============================================================

mae_improvement = (
    (old_mae - mae)
    / old_mae
    * 100
)

print(
    f"\nMAE improvement: {mae_improvement:.2f}%"
)


if mae < old_mae:
    print("RESULT: Tuned model improved MAE.")
elif mae > old_mae:
    print("RESULT: Current model is better.")
else:
    print("RESULT: Both models have the same MAE.")


print("\nHyperparameter tuning completed successfully!")