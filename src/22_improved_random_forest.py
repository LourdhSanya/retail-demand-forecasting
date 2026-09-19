import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
INPUT_FILE = "data/processed/product_21212_improved_features.csv"

# Load the data
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
    "Rolling_Mean_28",
    "Demand_Change_1",
    "Demand_Change_7",
    "Short_Long_Ratio",
    "Recent_Trend"
]
target = "Demand"

# Split the data into train and test
split_index = int(len(df) * 0.80)
X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]
y_train = df[target].iloc[:split_index]
y_test = df[target].iloc[split_index:]

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
print("IMPROVED RANDOM FOREST DEMAND FORECASTING")
print("=" * 70)
print("\nNumber of features:", len(features))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# Make predictions
predictions = model.predict(X_test)
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
print("IMPROVED MODEL PERFORMANCE")
print("=" * 70)
print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")

# Compare with the old model
old_mae = 128.41
old_rmse = 228.56
old_wape = 64.24
print("\n" + "=" * 70)
print("OLD vs IMPROVED MODEL")
print("=" * 70)
comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "WAPE"],
    "Old Random Forest": [
        old_mae,
        old_rmse,
        old_wape
    ],
    "Improved Random Forest": [
        mae,
        rmse,
        wape
    ]
})
print(
    comparison.to_string(index=False)
)

# Calculate the improvement
mae_improvement = (
    (old_mae - mae)
    / old_mae
    * 100
)
print(
    f"\nMAE change: {mae_improvement:.2f}%"
)
if mae < old_mae:
    print(
        "RESULT: Improved features increased model performance!"
    )
elif mae > old_mae:
    print(
        "RESULT: Improved features reduced model performance."
    )
else:
    print(
        "RESULT: No change in MAE."
    )

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

# Show sample predictions
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
print("\nImproved Random Forest modeling completed successfully!")