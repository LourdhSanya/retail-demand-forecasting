import pandas as pd
INPUT_FILE = "data/processed/retail_cleaned.csv"
PRODUCT_CODE = "21212"

# Load the data
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Date"] = df["InvoiceDate"].dt.normalize()

# Select the required product
product = df[df["StockCode"] == PRODUCT_CODE].copy()
print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)
print(f"\nProduct Code: {PRODUCT_CODE}")
print(f"Product Description: {product['Description'].iloc[0]}")

# Create daily demand
daily = (
    product.groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)

# Create complete calendar
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
daily = daily.rename(columns={"Quantity": "Demand"})

# Create calendar features
daily["DayOfWeek"] = daily["Date"].dt.dayofweek
daily["DayOfMonth"] = daily["Date"].dt.day
daily["Month"] = daily["Date"].dt.month
daily["Quarter"] = daily["Date"].dt.quarter
daily["WeekOfYear"] = daily["Date"].dt.isocalendar().week.astype(int)
daily["IsWeekend"] = (daily["DayOfWeek"] >= 5).astype(int)

# Create lag features
daily["Lag_1"] = daily["Demand"].shift(1)
daily["Lag_7"] = daily["Demand"].shift(7)
daily["Lag_14"] = daily["Demand"].shift(14)
daily["Lag_28"] = daily["Demand"].shift(28)

# Create rolling average features
daily["Rolling_Mean_7"] = (
    daily["Demand"]
    .shift(1)
    .rolling(window=7)
    .mean()
)
daily["Rolling_Mean_14"] = (
    daily["Demand"]
    .shift(1)
    .rolling(window=14)
    .mean()
)
daily["Rolling_Mean_28"] = (
    daily["Demand"]
    .shift(1)
    .rolling(window=28)
    .mean()
)

# Remove rows with missing values
model_data = daily.dropna().copy()

# Display the results
print("\nOriginal number of days:", len(daily))
print("Rows after feature engineering:", len(model_data))
print("\nFeature columns:")
print(model_data.columns.tolist())
print("\nFirst 10 rows of model dataset:")
print(model_data.head(10).to_string(index=False))
print("\nMissing values:")
print(model_data.isnull().sum())

# Save the feature dataset
OUTPUT_FILE = "data/processed/product_21212_features.csv"
model_data.to_csv(
    OUTPUT_FILE,
    index=False
)
print("\nFeature engineering completed!")
print(f"Feature dataset saved to: {OUTPUT_FILE}")