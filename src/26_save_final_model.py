import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

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

# Use all available data
X = df[features]
y = df[target]

print("=" * 70)
print("TRAINING FINAL MODEL FOR DEPLOYMENT")
print("=" * 70)
print("\nTotal training rows:", len(X))
print("Number of features:", len(features))

# Create the final Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining final model...")
model.fit(X, y)
print("Training completed!")

# Save the trained model
MODEL_FILE = "models/random_forest_product_21212.joblib"
joblib.dump(model, MODEL_FILE)

print("\nModel saved successfully!")
print("Saved to:", MODEL_FILE)

# Check that the saved model can be loaded
loaded_model = joblib.load(MODEL_FILE)
print("\nModel verification:")
print("Loaded model type:", type(loaded_model).__name__)
print("Number of estimators:", loaded_model.n_estimators)
print("Max depth:", loaded_model.max_depth)
print("Minimum samples per leaf:", loaded_model.min_samples_leaf)

# Print the final message
print("\n" + "=" * 70)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 70)