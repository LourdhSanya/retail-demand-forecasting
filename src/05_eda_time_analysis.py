import pandas as pd
import matplotlib.pyplot as plt

# Retail demand forecasting - time analysis
INPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the cleaned data
print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
# Convert the date column to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"Rows loaded: {len(df):,}")

# Show basic date information
print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)
print(f"Earliest date: {df['InvoiceDate'].min()}")
print(f"Latest date:   {df['InvoiceDate'].max()}")

# Create a separate date column
df["Date"] = df["InvoiceDate"].dt.date

# Calculate total demand for each day
daily_demand = (
    df.groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)
daily_demand["Date"] = pd.to_datetime(daily_demand["Date"])

print("\n" + "=" * 70)
print("DAILY DEMAND")
print("=" * 70)
print(daily_demand.head(10))
print(f"\nNumber of days: {len(daily_demand):,}")
print("\nDaily demand statistics:")
print(daily_demand["Quantity"].describe())

# Plot daily demand
plt.figure(figsize=(14, 6))
plt.plot(
    daily_demand["Date"],
    daily_demand["Quantity"]
)
plt.title("Daily Retail Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()