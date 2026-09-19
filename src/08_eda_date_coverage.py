import pandas as pd

# Retail demand forecasting - date coverage analysis
INPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the cleaned data
df = pd.read_csv(INPUT_FILE)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Date"] = df["InvoiceDate"].dt.date
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

# Get daily demand
daily_data = (
    df.groupby(["Date", "DayOfWeek"])["Quantity"]
    .sum()
    .reset_index()
)

# Count the number of days for each weekday
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

# Show all Saturday dates
saturday_dates = daily_data[
    daily_data["DayOfWeek"] == "Saturday"
]
print("\n" + "=" * 70)
print("SATURDAY DATA")
print("=" * 70)
print(saturday_dates.to_string(index=False))

# Show the overall date range
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