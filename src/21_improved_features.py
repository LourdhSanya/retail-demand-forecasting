import pandas as pd
INPUT_FILE = "data/processed/retail_cleaned.csv"
PRODUCT_CODE = "21212"

# Load the data
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Date"] = df["InvoiceDate"].dt.normalize()

# Select the required product
product = df[df["StockCode"] == PRODUCT_CODE].copy()

# Create daily demand
daily = (
    product.groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)
all_dates = pd.date_range(
    start=df["InvoiceDate"].min().normalize(),
    end=df["InvoiceDate"].max().normalize(),
    freq="D"
)
daily = (
    daily.set_index("Date")
    .reindex(all_dates, fill_value=0)
    .rename_axis("Date")
    .reset_index()
)
daily = daily.rename(
    columns={"Quantity": "Demand"}
)

# Create calendar features
daily["DayOfWeek"] = daily["Date"].dt.dayofweek
daily["DayOfMonth"] = daily["Date"].dt.day
daily["Month"] = daily["Date"].dt.month
daily["Quarter"] = daily["Date"].dt.quarter
daily["WeekOfYear"] = (
    daily["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)
daily["IsWeekend"] = (
    daily["DayOfWeek"] >= 5
).astype(int)

# Create lag features
daily["Lag_1"] = daily["Demand"].shift(1)
daily["Lag_7"] = daily["Demand"].shift(7)
daily["Lag_14"] = daily["Demand"].shift(14)
daily["Lag_28"] = daily["Demand"].shift(28)

# Create rolling average features
daily["Rolling_Mean_7"] = (
    daily["Demand"]
    .shift(1)
    .rolling(7)
    .mean()
)
daily["Rolling_Mean_14"] = (
    daily["Demand"]
    .shift(1)
    .rolling(14)
    .mean()
)
daily["Rolling_Mean_28"] = (
    daily["Demand"]
    .shift(1)
    .rolling(28)
    .mean()
)

# Create new trend features
daily["Demand_Change_1"] = (
    daily["Lag_1"] - daily["Lag_7"]
)
daily["Demand_Change_7"] = (
    daily["Lag_1"] - daily["Lag_14"]
)
daily["Short_Long_Ratio"] = (
    daily["Rolling_Mean_7"]
    / (daily["Rolling_Mean_28"] + 1)
)
daily["Recent_Trend"] = (
    daily["Rolling_Mean_7"]
    - daily["Rolling_Mean_28"]
)

# Remove rows with missing values
model_data = daily.dropna().copy()

# Display the results
print("=" * 70)
print("IMPROVED FEATURE ENGINEERING")
print("=" * 70)
print("\nRows before feature removal:", len(daily))
print("Rows after feature engineering:", len(model_data))
print("\nNew features added:")
print("Demand_Change_1")
print("Demand_Change_7")
print("Short_Long_Ratio")
print("Recent_Trend")
print("\nFeature columns:")
print(model_data.columns.tolist())
print("\nSample of new features:")
print(
    model_data[
        [
            "Date",
            "Demand",
            "Lag_1",
            "Lag_7",
            "Rolling_Mean_7",
            "Rolling_Mean_28",
            "Demand_Change_1",
            "Demand_Change_7",
            "Short_Long_Ratio",
            "Recent_Trend"
        ]
    ]
    .head(10)
    .to_string(index=False)
)
print("\nMissing values:")
print(model_data.isnull().sum())

# Save the improved feature data
OUTPUT_FILE = (
    "data/processed/"
    "product_21212_improved_features.csv"
)
model_data.to_csv(
    OUTPUT_FILE,
    index=False
)
print("\nImproved feature engineering completed!")
print(f"Saved to: {OUTPUT_FILE}")