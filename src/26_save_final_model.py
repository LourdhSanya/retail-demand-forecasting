import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


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
# 3. USE ALL AVAILABLE DATA
# ============================================================

X = df[features]

y = df[target]


print("=" * 70)
print("TRAINING FINAL MODEL FOR DEPLOYMENT")
print("=" * 70)

print("\nTotal training rows:", len(X))

print("Number of features:", len(features))


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
# 5. TRAIN MODEL
# ============================================================

print("\nTraining final model...")

model.fit(X, y)

print("Training completed!")


# ============================================================
# 6. SAVE MODEL
# ============================================================

MODEL_FILE = "models/random_forest_product_21212.joblib"

joblib.dump(model, MODEL_FILE)


print("\nModel saved successfully!")

print("Saved to:", MODEL_FILE)


# ============================================================
# 7. VERIFY MODEL
# ============================================================

loaded_model = joblib.load(MODEL_FILE)

print("\nModel verification:")

print("Loaded model type:", type(loaded_model).__name__)

print("Number of estimators:", loaded_model.n_estimators)

print("Max depth:", loaded_model.max_depth)

print("Minimum samples per leaf:", loaded_model.min_samples_leaf)


# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 70)