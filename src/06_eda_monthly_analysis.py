import pandas as pd
import matplotlib.pyplot as plt

# Retail demand forecasting - monthly demand analysis
INPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the cleaned data
print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"Rows loaded: {len(df):,}")

# Create a month column
df["Month"] = df["InvoiceDate"].dt.to_period("M")

# Calculate total demand for each month
monthly_demand = (
    df.groupby("Month")["Quantity"]
    .sum()
    .reset_index()
)
monthly_demand["Month"] = monthly_demand["Month"].astype(str)

# Display monthly demand
print("\n" + "=" * 70)
print("MONTHLY DEMAND")
print("=" * 70)
print(monthly_demand.to_string(index=False))

# Show monthly demand statistics
print("\n" + "=" * 70)
print("MONTHLY DEMAND STATISTICS")
print("=" * 70)
print(monthly_demand["Quantity"].describe())

# Find the month with highest demand
highest_month = monthly_demand.loc[
    monthly_demand["Quantity"].idxmax()
]
print("\nHighest demand month:")
print(
    f"{highest_month['Month']} "
    f"-> {highest_month['Quantity']:,} units"
)

# Find the month with lowest demand
lowest_month = monthly_demand.loc[
    monthly_demand["Quantity"].idxmin()
]
print("\nLowest demand month:")
print(
    f"{lowest_month['Month']} "
    f"-> {lowest_month['Quantity']:,} units"
)

# Plot monthly demand
plt.figure(figsize=(14, 6))
plt.bar(
    monthly_demand["Month"],
    monthly_demand["Quantity"]
)
plt.title("Monthly Retail Demand")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()