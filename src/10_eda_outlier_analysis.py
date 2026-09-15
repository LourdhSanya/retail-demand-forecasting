import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RETAIL DEMAND FORECASTING
# EDA - OUTLIER ANALYSIS
# ============================================================

INPUT_FILE = "data/processed/retail_cleaned.csv"


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Date"] = df["InvoiceDate"].dt.date


# ------------------------------------------------------------
# 2. CALCULATE DAILY DEMAND
# ------------------------------------------------------------

daily_demand = (
    df.groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)

daily_demand["Date"] = pd.to_datetime(daily_demand["Date"])


# ------------------------------------------------------------
# 3. CALCULATE IQR
# ------------------------------------------------------------

Q1 = daily_demand["Quantity"].quantile(0.25)
Q3 = daily_demand["Quantity"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print("=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)

print(f"Q1: {Q1:,.2f}")
print(f"Q3: {Q3:,.2f}")
print(f"IQR: {IQR:,.2f}")

print(f"\nLower Bound: {lower_bound:,.2f}")
print(f"Upper Bound: {upper_bound:,.2f}")


# ------------------------------------------------------------
# 4. IDENTIFY OUTLIERS
# ------------------------------------------------------------

outliers = daily_demand[
    (daily_demand["Quantity"] < lower_bound) |
    (daily_demand["Quantity"] > upper_bound)
]


print("\n" + "=" * 70)
print("OUTLIER DAYS")
print("=" * 70)

print(f"Number of outlier days: {len(outliers)}")

print("\nTop 20 highest-demand days:")

top_days = daily_demand.sort_values(
    "Quantity",
    ascending=False
).head(20)

print(top_days.to_string(index=False))


# ------------------------------------------------------------
# 5. OUTLIER STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTLIER SUMMARY")
print("=" * 70)

print(
    f"Highest daily demand: "
    f"{daily_demand['Quantity'].max():,}"
)

print(
    f"Lowest daily demand: "
    f"{daily_demand['Quantity'].min():,}"
)

print(
    f"Average daily demand: "
    f"{daily_demand['Quantity'].mean():,.2f}"
)

print(
    f"Median daily demand: "
    f"{daily_demand['Quantity'].median():,.2f}"
)


# ------------------------------------------------------------
# 6. BOXPLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.boxplot(
    daily_demand["Quantity"],
    orientation="vertical"
)

plt.title("Distribution of Daily Retail Demand")
plt.ylabel("Quantity Sold")

plt.tight_layout()

plt.show()