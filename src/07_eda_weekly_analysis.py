import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RETAIL DEMAND FORECASTING
# EDA - DAY OF WEEK ANALYSIS
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
# 2. CREATE DATE AND DAY OF WEEK
# ------------------------------------------------------------

df["Date"] = df["InvoiceDate"].dt.date
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()


# ------------------------------------------------------------
# 3. TOTAL DEMAND BY DAY OF WEEK
# ------------------------------------------------------------

weekly_demand = (
    df.groupby("DayOfWeek")["Quantity"]
    .sum()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ])
)


print("\n" + "=" * 70)
print("DEMAND BY DAY OF WEEK")
print("=" * 70)

print(weekly_demand)


# ------------------------------------------------------------
# 4. DAILY DEMAND
# ------------------------------------------------------------

daily_demand = (
    df.groupby(["Date", "DayOfWeek"])["Quantity"]
    .sum()
    .reset_index()
)


# ------------------------------------------------------------
# 5. AVERAGE DAILY DEMAND BY DAY OF WEEK
# ------------------------------------------------------------

average_by_day = (
    daily_demand.groupby("DayOfWeek")["Quantity"]
    .mean()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ])
)


print("\n" + "=" * 70)
print("AVERAGE DAILY DEMAND BY DAY")
print("=" * 70)

print(average_by_day)


# ------------------------------------------------------------
# 6. HIGHEST AVERAGE DEMAND DAY
# ------------------------------------------------------------

highest_day = average_by_day.idxmax()

print("\nHighest average-demand day:")
print(
    f"{highest_day} -> "
    f"{average_by_day[highest_day]:,.2f} units/day"
)


# ------------------------------------------------------------
# 7. LOWEST AVERAGE DEMAND DAY
# ------------------------------------------------------------

lowest_day = average_by_day.idxmin()

print("\nLowest average-demand day:")
print(
    f"{lowest_day} -> "
    f"{average_by_day[lowest_day]:,.2f} units/day"
)


# ------------------------------------------------------------
# 8. PLOT
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    average_by_day.index,
    average_by_day.values
)

plt.title("Average Daily Retail Demand by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Quantity Sold")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()