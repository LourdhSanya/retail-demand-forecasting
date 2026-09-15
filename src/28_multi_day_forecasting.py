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

def forecast_next_days(number_of_days):

    # Create a copy of historical demand
    history = df[["Date", "Demand"]].copy()

    forecasts = []

    # --------------------------------------------------------
    # Generate predictions one day at a time
    # --------------------------------------------------------

    for i in range(number_of_days):

        # Next date
        forecast_date = (
            history["Date"].max()
            + pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        day_of_week = forecast_date.dayofweek
        day_of_month = forecast_date.day
        month = forecast_date.month
        quarter = forecast_date.quarter
        week_of_year = forecast_date.isocalendar().week
        is_weekend = int(day_of_week >= 5)

        # ----------------------------------------------------
        # Historical / predicted demand
        # ----------------------------------------------------

        demand_values = history["Demand"].values

        # ----------------------------------------------------
        # Lag features
        # ----------------------------------------------------

        lag_1 = demand_values[-1]
        lag_7 = demand_values[-7]
        lag_14 = demand_values[-14]
        lag_28 = demand_values[-28]

        # ----------------------------------------------------
        # Rolling averages
        # ----------------------------------------------------

        rolling_mean_7 = np.mean(
            demand_values[-7:]
        )

        rolling_mean_14 = np.mean(
            demand_values[-14:]
        )

        rolling_mean_28 = np.mean(
            demand_values[-28:]
        )

        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        forecasts.append({
            "Date": forecast_date,
            "Predicted_Demand": prediction
        })

        # ----------------------------------------------------
        # IMPORTANT:
        # Add prediction to history so that the next
        # prediction can use it as a lag value.
        # ----------------------------------------------------

        new_row = pd.DataFrame({
            "Date": [forecast_date],
            "Demand": [prediction]
        })

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )


    return pd.DataFrame(forecasts)


# ============================================================
# 5. TEST
# ============================================================

print("=" * 70)
print("MULTI-DAY RETAIL DEMAND FORECASTING")
print("=" * 70)

print("\nProduct:")
print("21212 - PACK OF 72 RETRO SPOT CAKE CASES")

print("\nLatest historical date:")
print(df["Date"].max().date())


# ============================================================
# 6. FORECAST NEXT 7 DAYS
# ============================================================

forecast = forecast_next_days(7)


print("\n" + "=" * 70)
print("NEXT 7 DAYS FORECAST")
print("=" * 70)

print(
    forecast.to_string(index=False)
)


# ============================================================
# 7. TOTAL FORECAST
# ============================================================

total_demand = forecast["Predicted_Demand"].sum()

average_demand = forecast["Predicted_Demand"].mean()


print("\n" + "=" * 70)
print("FORECAST SUMMARY")
print("=" * 70)

print(
    f"\nTotal predicted demand: {total_demand:.2f} units"
)

print(
    f"Average predicted daily demand: {average_demand:.2f} units"
)


# ============================================================
# 8. SAVE FORECAST
# ============================================================

OUTPUT_FILE = "data/processed/7_day_forecast.csv"

forecast.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nForecast saved to: {OUTPUT_FILE}"
)


print("\n" + "=" * 70)
print("MULTI-DAY FORECASTING COMPLETED")
print("=" * 70)