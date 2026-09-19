import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
INPUT_FILE = "data/processed/product_21212_features.csv"

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
    "Rolling_Mean_28"
]
target = "Demand"

# Split the data into train and test
split_index = int(len(df) * 0.80)
X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]
y_train = df[target].iloc[:split_index]
y_test = df[target].iloc[split_index:]

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

# Create a table with actual and predicted values
results = pd.DataFrame({
    "Date": df.iloc[split_index:]["Date"].values,
    "Actual": y_test.values,
    "Predicted": predictions
})
results["Error"] = (
    results["Actual"] - results["Predicted"]
)
results["Absolute_Error"] = (
    results["Error"].abs()
)

# Calculate error metrics
mae = mean_absolute_error(
    results["Actual"],
    results["Predicted"]
)
rmse = np.sqrt(
    mean_squared_error(
        results["Actual"],
        results["Predicted"]
    )
)
print("=" * 70)
print("RANDOM FOREST ERROR ANALYSIS")
print("=" * 70)
print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# Check if the model has prediction bias
print("\n" + "=" * 70)
print("ERROR BIAS")
print("=" * 70)
print(
    f"\nAverage Error: {results['Error'].mean():.2f}"
)
if results["Error"].mean() > 0:
    print(
        "Overall tendency: Model tends to UNDER-PREDICT."
    )
elif results["Error"].mean() < 0:
    print(
        "Overall tendency: Model tends to OVER-PREDICT."
    )
else:
    print(
        "Overall tendency: No overall prediction bias."
    )

# Find the predictions with the largest errors
worst_predictions = results.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)
print("\n" + "=" * 70)
print("TOP 10 LARGEST FORECAST ERRORS")
print("=" * 70)
print(
    worst_predictions[
        [
            "Date",
            "Actual",
            "Predicted",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)

# Find the days with the highest actual demand
largest_demand = results.sort_values(
    "Actual",
    ascending=False
).head(10)
print("\n" + "=" * 70)
print("TOP 10 HIGHEST ACTUAL DEMAND DAYS")
print("=" * 70)
print(
    largest_demand[
        [
            "Date",
            "Actual",
            "Predicted",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)

# Plot actual and predicted demand
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
    "Actual vs Predicted Demand - Random Forest"
)
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the prediction errors
plt.figure(figsize=(14, 5))
plt.plot(
    results["Date"],
    results["Absolute_Error"]
)
plt.title(
    "Forecast Absolute Error Over Time"
)
plt.xlabel("Date")
plt.ylabel("Absolute Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\nError analysis completed successfully!")