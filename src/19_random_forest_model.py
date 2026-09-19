import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
INPUT_FILE = "data/processed/product_21212_features.csv"

# Load the data
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Select features and target
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

# Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("=" * 70)
print("RANDOM FOREST DEMAND FORECASTING")
print("=" * 70)
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# Make predictions
predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)

# Calculate model performance
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
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")

# Compare with the baseline
baseline_mae = 171.78
print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)
print(f"\nBest baseline MAE : {baseline_mae:.2f}")
print(f"Random Forest MAE : {mae:.2f}")
improvement = (
    (baseline_mae - mae)
    / baseline_mae
    * 100
)
print(f"\nMAE improvement: {improvement:.2f}%")
if mae < baseline_mae:
    print("RESULT: Random Forest beats the baseline!")
else:
    print("RESULT: Random Forest does NOT beat the baseline yet.")

# Check feature importance
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

# Show some sample predictions
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
print("\nRandom Forest modeling completed successfully!")