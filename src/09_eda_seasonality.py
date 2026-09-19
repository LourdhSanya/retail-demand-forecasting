import pandas as pd
import matplotlib.pyplot as plt

# Retail demand forecasting - seasonality analysis
INPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the cleaned data
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create a month column
df["Month"] = df["InvoiceDate"].dt.to_period("M")

# Calculate monthly demand
monthly_demand = (
    df.groupby("Month")["Quantity"]
    .sum()
    .reset_index()
)
monthly_demand["Month"] = monthly_demand["Month"].astype(str)

# Calculate the change from the previous month
monthly_demand["Change"] = (
    monthly_demand["Quantity"].pct_change() * 100
)

# Display the results
print("=" * 70)
print("MONTHLY SEASONALITY ANALYSIS")
print("=" * 70)
print(
    monthly_demand.to_string(index=False)
)

# Compare complete months
# December 2010 is incomplete, so it is not included
complete_months = monthly_demand[
    monthly_demand["Month"] != "2010-12"
]
highest = complete_months.loc[
    complete_months["Quantity"].idxmax()
]
lowest = complete_months.loc[
    complete_months["Quantity"].idxmin()
]

print("\n" + "=" * 70)
print("COMPLETE MONTH COMPARISON")
print("=" * 70)
print(
    f"Highest complete month: "
    f"{highest['Month']} -> "
    f"{highest['Quantity']:,} units"
)
print(
    f"Lowest complete month: "
    f"{lowest['Month']} -> "
    f"{lowest['Quantity']:,} units"
)

# Plot monthly demand
plt.figure(figsize=(14, 6))
plt.plot(
    monthly_demand["Month"],
    monthly_demand["Quantity"],
    marker="o"
)
plt.title("Monthly Retail Demand and Seasonal Pattern")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()