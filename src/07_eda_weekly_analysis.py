import pandas as pd
import matplotlib.pyplot as plt

# Retail demand forecasting - day of week analysis
INPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the cleaned data
print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"Rows loaded: {len(df):,}")

# Create date and day columns
df["Date"] = df["InvoiceDate"].dt.date
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

# Find total demand for each day
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

# Calculate daily demand
daily_demand = (
    df.groupby(["Date", "DayOfWeek"])["Quantity"]
    .sum()
    .reset_index()
)

# Find average demand for each day
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

# Find the day with highest average demand
highest_day = average_by_day.idxmax()
print("\nHighest average-demand day:")
print(
    f"{highest_day} -> "
    f"{average_by_day[highest_day]:,.2f} units/day"
)

# Find the day with lowest average demand
lowest_day = average_by_day.idxmin()
print("\nLowest average-demand day:")
print(
    f"{lowest_day} -> "
    f"{average_by_day[lowest_day]:,.2f} units/day"
)

# Plot average demand by day
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