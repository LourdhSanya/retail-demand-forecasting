import pandas as pd

# ============================================================
# RETAIL DEMAND FORECASTING
# EDA - DATE COVERAGE ANALYSIS
# ============================================================

INPUT_FILE = "data/processed/retail_cleaned.csv"


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Date"] = df["InvoiceDate"].dt.date
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()


# ------------------------------------------------------------
# 2. UNIQUE DATES
# ------------------------------------------------------------

daily_data = (
    df.groupby(["Date", "DayOfWeek"])["Quantity"]
    .sum()
    .reset_index()
)


# ------------------------------------------------------------
# 3. NUMBER OF DAYS BY WEEKDAY
# ------------------------------------------------------------

days_by_weekday = (
    daily_data.groupby("DayOfWeek")["Date"]
    .nunique()
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


print("=" * 70)
print("DATE COVERAGE BY DAY OF WEEK")
print("=" * 70)

print(days_by_weekday)


# ------------------------------------------------------------
# 4. SHOW SATURDAY DATES
# ------------------------------------------------------------

saturday_dates = daily_data[
    daily_data["DayOfWeek"] == "Saturday"
]

print("\n" + "=" * 70)
print("SATURDAY DATA")
print("=" * 70)

print(saturday_dates.to_string(index=False))


# ------------------------------------------------------------
# 5. ALL UNIQUE DATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATE COVERAGE")
print("=" * 70)

print(f"Total unique transaction dates: {daily_data['Date'].nunique()}")

print(
    f"First transaction date: "
    f"{daily_data['Date'].min()}"
)

print(
    f"Last transaction date: "
    f"{daily_data['Date'].max()}"
)