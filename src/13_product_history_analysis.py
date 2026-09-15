import pandas as pd

# ============================================================
# RETAIL DEMAND FORECASTING
# EDA - PRODUCT HISTORY ANALYSIS
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

df["Date"] = df["InvoiceDate"].dt.date

print(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------
# 2. DESCRIPTION CONSISTENCY
# ------------------------------------------------------------

description_count = (
    df.groupby("StockCode")["Description"]
    .nunique()
    .sort_values(ascending=False)
)

multiple_descriptions = description_count[
    description_count > 1
]


print("\n" + "=" * 70)
print("STOCKCODE / DESCRIPTION CONSISTENCY")
print("=" * 70)

print(
    f"Unique StockCodes: "
    f"{df['StockCode'].nunique():,}"
)

print(
    f"StockCodes with multiple descriptions: "
    f"{len(multiple_descriptions):,}"
)

print("\nExamples:")

print(
    multiple_descriptions.head(20)
)


# ------------------------------------------------------------
# 3. PRODUCT HISTORY
# ------------------------------------------------------------

product_history = (
    df.groupby("StockCode")
    .agg(
        Description=("Description", "first"),
        Total_Demand=("Quantity", "sum"),
        Active_Days=("Date", "nunique"),
        Transaction_Rows=("StockCode", "size"),
        Average_Quantity_Per_Row=("Quantity", "mean")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 4. SORT BY TOTAL DEMAND
# ------------------------------------------------------------

top_by_demand = product_history.sort_values(
    "Total_Demand",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP 20 PRODUCTS BY TOTAL DEMAND")
print("=" * 70)

print(
    top_by_demand.head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 5. SORT BY ACTIVE DAYS
# ------------------------------------------------------------

top_by_active_days = product_history.sort_values(
    "Active_Days",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP 20 PRODUCTS BY NUMBER OF ACTIVE DAYS")
print("=" * 70)

print(
    top_by_active_days.head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 6. PRODUCTS WITH SUFFICIENT HISTORY
# ------------------------------------------------------------

sufficient_history = product_history[
    product_history["Active_Days"] >= 50
].copy()


print("\n" + "=" * 70)
print("PRODUCT HISTORY COVERAGE")
print("=" * 70)

print(
    f"Products active on at least 50 days: "
    f"{len(sufficient_history):,}"
)

print(
    f"Products active on at least 100 days: "
    f"{len(product_history[product_history['Active_Days'] >= 100]):,}"
)

print(
    f"Products active on at least 150 days: "
    f"{len(product_history[product_history['Active_Days'] >= 150]):,}"
)


# ------------------------------------------------------------
# 7. TOP CANDIDATES FOR FORECASTING
# ------------------------------------------------------------

candidates = product_history[
    (product_history["Active_Days"] >= 100) &
    (product_history["Total_Demand"] >= 10000)
].sort_values(
    "Total_Demand",
    ascending=False
)


print("\n" + "=" * 70)
print("POTENTIAL FORECASTING CANDIDATES")
print("=" * 70)

print(
    f"Number of candidate products: "
    f"{len(candidates):,}"
)

print("\nTop 20 candidates:")

print(
    candidates.head(20).to_string(index=False)
)