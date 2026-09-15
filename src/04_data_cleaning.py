import pandas as pd

# ============================================================
# RETAIL DEMAND FORECASTING
# DATA CLEANING
# ============================================================

INPUT_FILE = "data/raw/online_retail_II.xlsx"
OUTPUT_FILE = "data/processed/retail_cleaned.csv"


# ------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------

print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)

print(f"Original number of rows: {len(df):,}")


# ------------------------------------------------------------
# 2. REMOVE DUPLICATES
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\n1. DUPLICATE REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")


# ------------------------------------------------------------
# 3. REMOVE CANCELLED INVOICES
# ------------------------------------------------------------

before = len(df)

cancelled = df["Invoice"].astype(str).str.upper().str.startswith("C")

df = df[~cancelled]

after = len(df)

print("\n2. CANCELLED INVOICE REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")


# ------------------------------------------------------------
# 4. REMOVE NEGATIVE QUANTITIES
# ------------------------------------------------------------

before = len(df)

df = df[df["Quantity"] > 0]

after = len(df)

print("\n3. NEGATIVE / NON-POSITIVE QUANTITY REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")


# ------------------------------------------------------------
# 5. REMOVE NEGATIVE PRICES
# ------------------------------------------------------------

before = len(df)

df = df[df["Price"] >= 0]

after = len(df)

print("\n4. NEGATIVE PRICE REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")


# ------------------------------------------------------------
# 6. REMOVE MISSING PRODUCT DESCRIPTIONS
# ------------------------------------------------------------

before = len(df)

df = df.dropna(subset=["Description"])

after = len(df)

print("\n5. MISSING DESCRIPTION REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")


# ------------------------------------------------------------
# 7. CHECK CUSTOMER ID
# ------------------------------------------------------------

print("\n6. CUSTOMER ID CHECK")
print(f"Missing Customer IDs remaining: {df['Customer ID'].isnull().sum():,}")

print("Customer IDs are retained as optional information.")


# ------------------------------------------------------------
# 8. CHECK ZERO PRICE
# ------------------------------------------------------------

print("\n7. ZERO PRICE CHECK")
print(f"Zero-price transactions remaining: {(df['Price'] == 0).sum():,}")


# ------------------------------------------------------------
# 9. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

print("\nFinal shape:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nRemaining negative quantities:")
print((df["Quantity"] < 0).sum())

print("\nRemaining negative prices:")
print((df["Price"] < 0).sum())

print("\nRemaining cancelled invoices:")
remaining_cancelled = (
    df["Invoice"]
    .astype(str)
    .str.upper()
    .str.startswith("C")
    .sum()
)

print(remaining_cancelled)


# ------------------------------------------------------------
# 10. SAVE CLEAN DATA
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print(f"\nCleaned dataset saved to:")
print(OUTPUT_FILE)

print(f"\nFinal number of rows: {len(df):,}")