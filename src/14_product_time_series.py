import pandas as pd
import matplotlib.pyplot as plt

# Retail demand forecasting - product time series
INPUT_FILE = "data/processed/retail_cleaned.csv"
PRODUCT_CODE = "21212"

# Load the cleaned data
print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Date"] = df["InvoiceDate"].dt.normalize()
print(f"Rows loaded: {len(df):,}")

# Select the required product
product = df[
    df["StockCode"] == PRODUCT_CODE
].copy()
print("\n" + "=" * 70)
print("SELECTED PRODUCT")
print("=" * 70)
print(f"StockCode: {PRODUCT_CODE}")
print(
    f"Description: "
    f"{product['Description'].mode().iloc[0]}"
)
print(
    f"Transaction rows: "
    f"{len(product):,}"
)

# Calculate daily demand for the product
daily_product = (
    product
    .groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)
print("\n" + "=" * 70)
print("PRODUCT DAILY DEMAND")
print("=" * 70)
print(daily_product.head(15).to_string(index=False))

# Create dates for the complete time period
all_dates = pd.date_range(
    start=df["InvoiceDate"].min().normalize(),
    end=df["InvoiceDate"].max().normalize(),
    freq="D"
)
daily_product = (
    daily_product
    .set_index("Date")
    .reindex(all_dates, fill_value=0)
    .rename_axis("Date")
    .reset_index()
)

# Check days with zero demand
zero_days = (
    daily_product["Quantity"] == 0
).sum()
nonzero_days = (
    daily_product["Quantity"] > 0
).sum()
print("\n" + "=" * 70)
print("COMPLETE TIME SERIES")
print("=" * 70)
print(
    f"Total calendar days: "
    f"{len(daily_product):,}"
)
print(
    f"Days with demand: "
    f"{nonzero_days:,}"
)
print(
    f"Zero-demand days: "
    f"{zero_days:,}"
)

# Show product demand statistics
print("\n" + "=" * 70)
print("PRODUCT DEMAND STATISTICS")
print("=" * 70)
print(
    daily_product["Quantity"].describe()
)

# Show the date range
print("\n" + "=" * 70)
print("DATE RANGE")
print("=" * 70)
print(
    f"First date: "
    f"{daily_product['Date'].min().date()}"
)
print(
    f"Last date: "
    f"{daily_product['Date'].max().date()}"
)

# Plot the product demand
plt.figure(figsize=(14, 6))
plt.plot(
    daily_product["Date"],
    daily_product["Quantity"]
)
plt.title(
    f"Daily Demand - Product {PRODUCT_CODE}"
)
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()