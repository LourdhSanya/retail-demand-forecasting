import pandas as pd

INPUT_FILE = "data/processed/product_21212_features.csv"

# --------------------------------------------------
# 1. LOAD FEATURE DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

# Make absolutely sure data is chronological
df = df.sort_values("Date").reset_index(drop=True)

# --------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# --------------------------------------------------

features = [
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "Quarter",
    "WeekOfYear",
    "IsWeekend",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_28"
]

target = "Demand"

X = df[features]
y = df[target]

# --------------------------------------------------
# 3. TIME-BASED SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# --------------------------------------------------
# 4. DISPLAY RESULTS
# --------------------------------------------------

print("=" * 70)
print("TIME-SERIES TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTotal rows:", len(df))

print("\nTraining data:")
print("Rows:", len(X_train))
print("Start date:", df.iloc[0]["Date"].date())
print("End date:", df.iloc[split_index - 1]["Date"].date())

print("\nTesting data:")
print("Rows:", len(X_test))
print("Start date:", df.iloc[split_index]["Date"].date())
print("End date:", df.iloc[-1]["Date"].date())

print("\nTraining percentage:", round(len(X_train) / len(df) * 100, 2), "%")
print("Testing percentage:", round(len(X_test) / len(df) * 100, 2), "%")

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\ny_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --------------------------------------------------
# 5. VERIFY CHRONOLOGICAL ORDER
# --------------------------------------------------

print("\n" + "=" * 70)
print("CHRONOLOGICAL VERIFICATION")
print("=" * 70)

print("\nLast 5 training dates:")
print(df.iloc[split_index - 5:split_index][["Date", "Demand"]].to_string(index=False))

print("\nFirst 5 testing dates:")
print(df.iloc[split_index:split_index + 5][["Date", "Demand"]].to_string(index=False))

print("\nFuture data leakage check:")

if df.iloc[split_index - 1]["Date"] < df.iloc[split_index]["Date"]:
    print("PASS - Training data ends before testing data begins.")
else:
    print("FAIL - Possible data leakage!")

print("\nTime-series split completed successfully!")