import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RETAIL DEMAND FORECASTING
# EDA - MONTHLY DEMAND ANALYSIS
# ============================================================

INPUT_FILE = "data/processed/retail_cleaned.csv"


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------
# 2. CREATE MONTH COLUMN
# ------------------------------------------------------------

df["Month"] = df["InvoiceDate"].dt.to_period("M")


# ------------------------------------------------------------
# 3. CALCULATE MONTHLY DEMAND
# ------------------------------------------------------------

monthly_demand = (
    df.groupby("Month")["Quantity"]
    .sum()
    .reset_index()
)

monthly_demand["Month"] = monthly_demand["Month"].astype(str)


# ------------------------------------------------------------
# 4. DISPLAY MONTHLY DEMAND
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MONTHLY DEMAND")
print("=" * 70)

print(monthly_demand.to_string(index=False))


# ------------------------------------------------------------
# 5. MONTHLY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MONTHLY DEMAND STATISTICS")
print("=" * 70)

print(monthly_demand["Quantity"].describe())


# ------------------------------------------------------------
# 6. HIGHEST DEMAND MONTH
# ------------------------------------------------------------

highest_month = monthly_demand.loc[
    monthly_demand["Quantity"].idxmax()
]

print("\nHighest demand month:")
print(
    f"{highest_month['Month']} "
    f"-> {highest_month['Quantity']:,} units"
)


# ------------------------------------------------------------
# 7. LOWEST DEMAND MONTH
# ------------------------------------------------------------

lowest_month = monthly_demand.loc[
    monthly_demand["Quantity"].idxmin()
]

print("\nLowest demand month:")
print(
    f"{lowest_month['Month']} "
    f"-> {lowest_month['Quantity']:,} units"
)


# ------------------------------------------------------------
# 8. PLOT MONTHLY DEMAND
# ------------------------------------------------------------

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