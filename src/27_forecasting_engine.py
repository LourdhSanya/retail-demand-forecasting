import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_FILE = "models/random_forest_product_21212.joblib"

DATA_FILE = "data/processed/product_21212_features.csv"


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# 3. LOAD HISTORICAL DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 4. FORECAST FUNCTION
# ============================================================

def forecast_demand(forecast_date):

    forecast_date = pd.to_datetime(forecast_date)

    # --------------------------------------------------------
    # Check whether date is already in historical data
    # --------------------------------------------------------

    if forecast_date <= df["Date"].max():
        print("Forecast date must be after the latest historical date.")
        return None

    # --------------------------------------------------------
    # Create calendar features
    # --------------------------------------------------------

    day_of_week = forecast_date.dayofweek
    day_of_month = forecast_date.day
    month = forecast_date.month
    quarter = forecast_date.quarter
    week_of_year = forecast_date.isocalendar().week
    is_weekend = int(day_of_week >= 5)

    # --------------------------------------------------------
    # Historical demand
    # --------------------------------------------------------

    historical_demand = df["Demand"].tolist()

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_1 = historical_demand[-1]
    lag_7 = historical_demand[-7]
    lag_14 = historical_demand[-14]
    lag_28 = historical_demand[-28]

    # --------------------------------------------------------
    # Rolling mean features
    # --------------------------------------------------------

    rolling_mean_7 = np.mean(
        historical_demand[-7:]
    )

    rolling_mean_14 = np.mean(
        historical_demand[-14:]
    )

    rolling_mean_28 = np.mean(
        historical_demand[-28:]
    )

    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "DayOfWeek": [day_of_week],
        "DayOfMonth": [day_of_month],
        "Month": [month],
        "Quarter": [quarter],
        "WeekOfYear": [week_of_year],
        "IsWeekend": [is_weekend],
        "Lag_1": [lag_1],
        "Lag_7": [lag_7],
        "Lag_14": [lag_14],
        "Lag_28": [lag_28],
        "Rolling_Mean_7": [rolling_mean_7],
        "Rolling_Mean_14": [rolling_mean_14],
        "Rolling_Mean_28": [rolling_mean_28]
    })

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    return prediction, input_data


# ============================================================
# 5. TEST THE FORECASTING ENGINE
# ============================================================

print("=" * 70)
print("RETAIL DEMAND FORECASTING ENGINE")
print("=" * 70)

print("\nProduct:")
print("21212 - PACK OF 72 RETRO SPOT CAKE CASES")

print("\nLatest historical date:")
print(df["Date"].max().date())


# ============================================================
# 6. FORECAST A FUTURE DATE
# ============================================================

forecast_date = "2010-12-10"

result = forecast_demand(forecast_date)


if result is not None:

    prediction, input_data = result

    print("\n" + "=" * 70)
    print("FORECAST RESULT")
    print("=" * 70)

    print("\nForecast date:", forecast_date)

    print(
        f"Predicted demand: {prediction:.2f} units"
    )

    print("\nFeatures supplied to model:")

    print(
        input_data.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("FORECASTING ENGINE TEST COMPLETED")
    print("=" * 70)