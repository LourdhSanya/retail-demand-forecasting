import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RETAIL DEMAND FORECASTING
# EDA - PRODUCT DEMAND ANALYSIS
# ============================================================

INPUT_FILE = "data/processed/retail_cleaned.csv"


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------
# 2. PRODUCT DEMAND
# ------------------------------------------------------------

product_demand = (
    df.groupby(
        ["StockCode", "Description"]
    )["Quantity"]
    .sum()
    .reset_index()
    .sort_values(
        "Quantity",
        ascending=False
    )
)


# ------------------------------------------------------------
# 3. BASIC PRODUCT INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRODUCT INFORMATION")
print("=" * 70)

print(
    f"Unique products: "
    f"{df['StockCode'].nunique():,}"
)


# ------------------------------------------------------------
# 4. TOP 20 PRODUCTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 PRODUCTS BY TOTAL DEMAND")
print("=" * 70)

print(
    product_demand.head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 5. LOWEST DEMAND PRODUCTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRODUCT DEMAND STATISTICS")
print("=" * 70)

print(
    product_demand["Quantity"].describe()
)


# ------------------------------------------------------------
# 6. TOP 10 PRODUCT CONTRIBUTION
# ------------------------------------------------------------

total_demand = product_demand["Quantity"].sum()

top_10_demand = (
    product_demand.head(10)["Quantity"].sum()
)

top_20_demand = (
    product_demand.head(20)["Quantity"].sum()
)

print("\n" + "=" * 70)
print("DEMAND CONTRIBUTION")
print("=" * 70)

print(
    f"Total demand: "
    f"{total_demand:,} units"
)

print(
    f"Top 10 products demand: "
    f"{top_10_demand:,} units"
)

print(
    f"Top 10 contribution: "
    f"{(top_10_demand / total_demand) * 100:.2f}%"
)

print(
    f"\nTop 20 products demand: "
    f"{top_20_demand:,} units"
)

print(
    f"Top 20 contribution: "
    f"{(top_20_demand / total_demand) * 100:.2f}%"
)


# ------------------------------------------------------------
# 7. PLOT TOP 10 PRODUCTS
# ------------------------------------------------------------

top_10 = product_demand.head(10).copy()

labels = (
    top_10["StockCode"]
    + " - "
    + top_10["Description"].str[:25]
)

plt.figure(figsize=(14, 7))

plt.barh(
    labels[::-1],
    top_10["Quantity"][::-1]
)

plt.title("Top 10 Products by Total Demand")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Product")

plt.tight_layout()

plt.show()