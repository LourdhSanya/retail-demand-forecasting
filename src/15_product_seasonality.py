import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RETAIL DEMAND FORECASTING
# PRODUCT-SPECIFIC SEASONALITY
# ============================================================

INPUT_FILE = "data/processed/retail_cleaned.csv"

PRODUCT_CODE = "21212"


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Date"] = df["InvoiceDate"].dt.normalize()


# ------------------------------------------------------------
# 2. SELECT PRODUCT
# ------------------------------------------------------------

product = df[
    df["StockCode"] == PRODUCT_CODE
].copy()


# ------------------------------------------------------------
# 3. DAILY DEMAND
# ------------------------------------------------------------

daily = (
    product
    .groupby("Date")["Quantity"]
    .sum()
    .reset_index()
)


# ------------------------------------------------------------
# 4. COMPLETE DATE RANGE
# ------------------------------------------------------------

all_dates = pd.date_range(
    start=df["InvoiceDate"].min().normalize(),
    end=df["InvoiceDate"].max().normalize(),
    freq="D"
)

daily = (
    daily
    .set_index("Date")
    .reindex(all_dates, fill_value=0)
    .rename_axis("Date")
    .reset_index()
)


# ------------------------------------------------------------
# 5. CREATE TIME FEATURES
# ------------------------------------------------------------

daily["DayOfWeek"] = daily["Date"].dt.day_name()

daily["Month"] = daily["Date"].dt.month

daily["MonthName"] = daily["Date"].dt.month_name()


# ------------------------------------------------------------
# 6. DEMAND BY DAY OF WEEK
# ------------------------------------------------------------

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_demand = (
    daily
    .groupby("DayOfWeek")["Quantity"]
    .mean()
    .reindex(weekday_order)
)


print("=" * 70)
print("PRODUCT-SPECIFIC WEEKLY SEASONALITY")
print("=" * 70)

print("\nAverage daily demand by day of week:")

print(
    weekday_demand
)


print("\nHighest average-demand day:")

print(
    f"{weekday_demand.idxmax()} "
    f"-> {weekday_demand.max():,.2f} units/day"
)


print("\nLowest average-demand day:")

print(
    f"{weekday_demand.idxmin()} "
    f"-> {weekday_demand.min():,.2f} units/day"
)


# ------------------------------------------------------------
# 7. DEMAND BY MONTH
# ------------------------------------------------------------

monthly_demand = (
    daily
    .groupby("Month")["Quantity"]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_demand.index = month_names


print("\n" + "=" * 70)
print("PRODUCT-SPECIFIC MONTHLY SEASONALITY")
print("=" * 70)

print("\nAverage daily demand by month:")

print(
    monthly_demand
)


print("\nHighest average-demand month:")

print(
    f"{monthly_demand.idxmax()} "
    f"-> {monthly_demand.max():,.2f} units/day"
)


print("\nLowest average-demand month:")

print(
    f"{monthly_demand.idxmin()} "
    f"-> {monthly_demand.min():,.2f} units/day"
)


# ------------------------------------------------------------
# 8. PLOT WEEKLY PATTERN
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    weekday_demand.index,
    weekday_demand.values
)

plt.title(
    f"Average Daily Demand by Weekday - Product {PRODUCT_CODE}"
)

plt.xlabel("Day of Week")

plt.ylabel("Average Quantity")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 9. PLOT MONTHLY PATTERN
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_demand.index,
    monthly_demand.values,
    marker="o"
)

plt.title(
    f"Average Daily Demand by Month - Product {PRODUCT_CODE}"
)

plt.xlabel("Month")

plt.ylabel("Average Daily Quantity")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()