import pandas as pd

# Retail demand forecasting - investigate outlier days
INPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the data
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Date"] = df["InvoiceDate"].dt.date

# Calculate daily demand
daily_demand = (
    df.groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)
Q1 = daily_demand["Quantity"].quantile(0.25)
Q3 = daily_demand["Quantity"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

# Find dates with unusually high demand
outlier_dates = daily_demand[
    daily_demand["Quantity"] > upper_bound
].sort_values(
    "Quantity",
    ascending=False
)

print("=" * 70)
print("OUTLIER DAYS AND TOTAL DEMAND")
print("=" * 70)
print(
    outlier_dates.to_string(index=False)
)

# Check the top products on the main outlier days
print("\n" + "=" * 70)
print("TOP PRODUCTS ON MAJOR OUTLIER DAYS")
print("=" * 70)

for date in outlier_dates.head(5)["Date"]:
    day_data = df[df["Date"] == date]
    product_demand = (
        day_data
        .groupby(
            ["StockCode", "Description"]
        )["Quantity"]
        .sum()
        .reset_index()
        .sort_values(
            "Quantity",
            ascending=False
        )
        .head(10)
    )
    total = day_data["Quantity"].sum()
    print("\n" + "-" * 70)
    print(f"DATE: {date}")
    print(f"TOTAL DEMAND: {total:,}")
    print("-" * 70)
    print(product_demand.to_string(index=False))

# Check the number of transactions on outlier days
print("\n" + "=" * 70)
print("TRANSACTION COUNT ON MAJOR OUTLIER DAYS")
print("=" * 70)
for date in outlier_dates.head(5)["Date"]:
    day_data = df[df["Date"] == date]
    print(
        f"{date} -> "
        f"{len(day_data):,} transaction rows"
    )