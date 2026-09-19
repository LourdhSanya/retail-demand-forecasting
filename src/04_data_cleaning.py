import pandas as pd

# Retail demand forecasting - data cleaning
INPUT_FILE = "data/raw/online_retail_II.xlsx"
OUTPUT_FILE = "data/processed/retail_cleaned.csv"

# Load the raw data
print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)
df = pd.read_excel(INPUT_FILE)
print(f"Original number of rows: {len(df):,}")

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
after = len(df)
print("\n1. DUPLICATE REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")

# Remove cancelled invoices
before = len(df)
cancelled = df["Invoice"].astype(str).str.upper().str.startswith("C")
df = df[~cancelled]
after = len(df)
print("\n2. CANCELLED INVOICE REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")

# Remove invalid quantity values
before = len(df)
df = df[df["Quantity"] > 0]
after = len(df)
print("\n3. NEGATIVE / NON-POSITIVE QUANTITY REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")

# Remove negative prices
before = len(df)
df = df[df["Price"] >= 0]
after = len(df)
print("\n4. NEGATIVE PRICE REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")

# Remove rows without product descriptions
before = len(df)
df = df.dropna(subset=["Description"])
after = len(df)
print("\n5. MISSING DESCRIPTION REMOVAL")
print(f"Rows removed: {before - after:,}")
print(f"Rows remaining: {after:,}")

# Check missing customer IDs
print("\n6. CUSTOMER ID CHECK")
print(f"Missing Customer IDs remaining: {df['Customer ID'].isnull().sum():,}")
print("Customer IDs are retained as optional information.")

# Check zero-price transactions
print("\n7. ZERO PRICE CHECK")
print(f"Zero-price transactions remaining: {(df['Price'] == 0).sum():,}")

# Check the final data quality
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

# Save the cleaned data
df.to_csv(OUTPUT_FILE, index=False)
print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)
print(f"\nCleaned dataset saved to:")
print(OUTPUT_FILE)
print(f"\nFinal number of rows: {len(df):,}")