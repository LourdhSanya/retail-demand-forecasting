import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
INPUT_FILE = "data/processed/product_21212_features.csv"
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Select the features
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

# Split the data based on time
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

# Create the base Random Forest model
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# Set the parameters to test
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [8, 10, 15, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0]
}

# Use time series cross validation
tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

# Train the models
print("\nStarting hyperparameter tuning...")
grid_search.fit(X_train, y_train)
print("\nTuning completed!")

# Display the best parameters
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

# Test the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
predictions = np.maximum(predictions, 0)

# Calculate final test metrics
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

# Compare with the current Random Forest
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

# Calculate MAE improvement
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