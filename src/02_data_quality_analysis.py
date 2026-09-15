import pandas as pd

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

file_path = "data/raw/online_retail_II.xlsx"

df = pd.read_excel(file_path)

print("=" * 70)
print("DATA QUALITY ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 2. BASIC INFORMATION
# --------------------------------------------------

print("\n1. TOTAL RECORDS")
print(len(df))

print("\n2. TOTAL COLUMNS")
print(len(df.columns))


# --------------------------------------------------
# 3. UNIQUE VALUES
# --------------------------------------------------

print("\n3. UNIQUE INVOICES")
print(df["Invoice"].nunique())

print("\n4. UNIQUE PRODUCTS")
print(df["StockCode"].nunique())

print("\n5. UNIQUE CUSTOMERS")
print(df["Customer ID"].nunique())

print("\n6. UNIQUE COUNTRIES")
print(df["Country"].nunique())


# --------------------------------------------------
# 4. MISSING VALUES
# --------------------------------------------------

print("\n7. MISSING VALUES")
print(df.isnull().sum())


# --------------------------------------------------
# 5. DUPLICATE RECORDS
# --------------------------------------------------

print("\n8. DUPLICATE ROWS")
print(df.duplicated().sum())


# --------------------------------------------------
# 6. QUANTITY ANALYSIS
# --------------------------------------------------

print("\n9. QUANTITY STATISTICS")
print(df["Quantity"].describe())

print("\n10. NEGATIVE QUANTITY RECORDS")
print((df["Quantity"] < 0).sum())

print("\n11. ZERO QUANTITY RECORDS")
print((df["Quantity"] == 0).sum())


# --------------------------------------------------
# 7. PRICE ANALYSIS
# --------------------------------------------------

print("\n12. PRICE STATISTICS")
print(df["Price"].describe())

print("\n13. ZERO PRICE RECORDS")
print((df["Price"] == 0).sum())

print("\n14. NEGATIVE PRICE RECORDS")
print((df["Price"] < 0).sum())


# --------------------------------------------------
# 8. CANCELLED INVOICES
# --------------------------------------------------

print("\n15. CANCELLED INVOICES")

cancelled = df["Invoice"].astype(str).str.upper().str.startswith("C")

print("Cancelled invoice records:", cancelled.sum())

print("\nSample cancelled transactions:")
print(df[cancelled].head())


# --------------------------------------------------
# 9. DATE RANGE
# --------------------------------------------------

print("\n16. DATE RANGE")

print("Earliest transaction:")
print(df["InvoiceDate"].min())

print("Latest transaction:")
print(df["InvoiceDate"].max())


# --------------------------------------------------
# 10. COUNTRIES
# --------------------------------------------------

print("\n17. TOP 10 COUNTRIES")

print(
    df["Country"]
    .value_counts()
    .head(10)
)


print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETED")
print("=" * 70)