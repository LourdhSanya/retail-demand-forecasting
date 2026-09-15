import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE = "data/processed/product_21212_features.csv"

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# --------------------------------------------------
# 2. TIME-BASED SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

actual = test["Demand"]

# --------------------------------------------------
# 3. BASELINE 1 - PREVIOUS DAY
# --------------------------------------------------

prediction_previous_day = test["Lag_1"]

# --------------------------------------------------
# 4. BASELINE 2 - SAME DAY LAST WEEK
# --------------------------------------------------

prediction_last_week = test["Lag_7"]

# --------------------------------------------------
# 5. BASELINE 3 - 7-DAY MOVING AVERAGE
# --------------------------------------------------

prediction_moving_average = test["Rolling_Mean_7"]

# --------------------------------------------------
# 6. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, actual, prediction):

    mae = mean_absolute_error(actual, prediction)

    rmse = np.sqrt(
        mean_squared_error(actual, prediction)
    )

    # WAPE avoids the division-by-zero problem
    # that occurs with MAPE when actual demand = 0.
    total_actual = actual.sum()

    if total_actual != 0:
        wape = (
            np.abs(actual - prediction).sum()
            / total_actual
            * 100
        )
    else:
        wape = np.nan

    print(f"\n{name}")
    print("-" * 50)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"WAPE : {wape:.2f}%")

    return mae, rmse, wape


# --------------------------------------------------
# 7. EVALUATE BASELINES
# --------------------------------------------------

print("=" * 70)
print("FORECASTING BASELINE MODELS")
print("=" * 70)

print("\nTraining rows:", len(train))
print("Testing rows :", len(test))

results = []

results.append(
    evaluate_model(
        "Baseline 1 - Previous Day",
        actual,
        prediction_previous_day
    )
)

results.append(
    evaluate_model(
        "Baseline 2 - Same Day Last Week",
        actual,
        prediction_last_week
    )
)

results.append(
    evaluate_model(
        "Baseline 3 - 7-Day Moving Average",
        actual,
        prediction_moving_average
    )
)

# --------------------------------------------------
# 8. COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame(
    results,
    columns=["MAE", "RMSE", "WAPE"]
)

comparison.index = [
    "Previous Day",
    "Same Day Last Week",
    "7-Day Moving Average"
]

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(comparison.round(2))

# --------------------------------------------------
# 9. FIND BEST BASELINE
# --------------------------------------------------

best_model = comparison["MAE"].idxmin()

print("\nBest baseline according to MAE:")
print(best_model)

print("\nBaseline evaluation completed successfully!")