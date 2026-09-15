import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/retail_cleaned.csv"

MODEL_FOLDER = "models"

DATA_FOLDER = "data/processed"


# ============================================================
# PRODUCTS TO SUPPORT
# ============================================================

PRODUCTS = {
    "21212": "PACK OF 72 RETRO SPOT CAKE CASES",
    "85123A": "WHITE HANGING HEART T-LIGHT HOLDER",
    "84077": "WORLD WAR 2 GLIDERS ASSTD DESIGNS",
    "85099B": "JUMBO BAG RED WHITE SPOTTY",
    "17003": "BROCADE RING PURSE"
}


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(MODEL_FOLDER, exist_ok=True)

os.makedirs(DATA_FOLDER, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

df["StockCode"] = df["StockCode"].astype(str)


# ============================================================
# FEATURES
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


# ============================================================
# TRAIN EACH PRODUCT
# ============================================================

print("=" * 70)
print("BUILDING MULTI-PRODUCT FORECASTING MODELS")
print("=" * 70)


for product_code, description in PRODUCTS.items():

    print("\n" + "-" * 70)

    print(
        f"Product: {product_code} - {description}"
    )

    # --------------------------------------------------------
    # Select product
    # --------------------------------------------------------

    product_df = df[
        df["StockCode"] == product_code
    ].copy()

    if product_df.empty:

        print("Product not found. Skipping.")

        continue

    # --------------------------------------------------------
    # Aggregate daily demand
    # --------------------------------------------------------

    daily = (
        product_df
        .groupby(
            product_df["InvoiceDate"].dt.normalize()
        )["Quantity"]
        .sum()
    )

    # --------------------------------------------------------
    # Complete calendar
    # --------------------------------------------------------

    full_dates = pd.date_range(
        start=df["InvoiceDate"].min().normalize(),
        end=df["InvoiceDate"].max().normalize(),
        freq="D"
    )

    daily = daily.reindex(
        full_dates,
        fill_value=0
    )

    product_ts = pd.DataFrame({
        "Date": daily.index,
        "Demand": daily.values
    })

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    product_ts["DayOfWeek"] = (
        product_ts["Date"].dt.dayofweek
    )

    product_ts["DayOfMonth"] = (
        product_ts["Date"].dt.day
    )

    product_ts["Month"] = (
        product_ts["Date"].dt.month
    )

    product_ts["Quarter"] = (
        product_ts["Date"].dt.quarter
    )

    product_ts["WeekOfYear"] = (
        product_ts["Date"].dt.isocalendar().week
    ).astype(int)

    product_ts["IsWeekend"] = (
        product_ts["DayOfWeek"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    product_ts["Lag_1"] = (
        product_ts["Demand"].shift(1)
    )

    product_ts["Lag_7"] = (
        product_ts["Demand"].shift(7)
    )

    product_ts["Lag_14"] = (
        product_ts["Demand"].shift(14)
    )

    product_ts["Lag_28"] = (
        product_ts["Demand"].shift(28)
    )

    # --------------------------------------------------------
    # Rolling means
    # --------------------------------------------------------

    product_ts["Rolling_Mean_7"] = (
        product_ts["Demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    product_ts["Rolling_Mean_14"] = (
        product_ts["Demand"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    product_ts["Rolling_Mean_28"] = (
        product_ts["Demand"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    # --------------------------------------------------------
    # Remove rows without enough history
    # --------------------------------------------------------

    product_ts = product_ts.dropna().reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    X = product_ts[features]

    y = product_ts["Demand"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = (
        f"{MODEL_FOLDER}/"
        f"random_forest_product_{product_code}.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    # --------------------------------------------------------
    # Save product time series
    # --------------------------------------------------------

    data_file = (
        f"{DATA_FOLDER}/"
        f"product_{product_code}_features.csv"
    )

    product_ts.to_csv(
        data_file,
        index=False
    )

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print(
        f"Model saved: {model_file}"
    )

    print(
        f"Feature data saved: {data_file}"
    )

    print(
        f"Training rows: {len(product_ts)}"
    )


print("\n" + "=" * 70)

print("MULTI-PRODUCT MODEL BUILDING COMPLETED")

print("=" * 70)