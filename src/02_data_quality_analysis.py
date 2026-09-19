import pandas as pd

# Load the dataset
file_path = "data/raw/online_retail_II.xlsx"

df = pd.read_excel(file_path)

print("=" * 70)

print("DATA QUALITY ANALYSIS")

print("=" * 70)


# Check basic details of the data

print("\n1. TOTAL RECORDS")

print(len(df))

print("\n2. TOTAL COLUMNS")

print(len(df.columns))


# Find unique values in important columns

print("\n3. UNIQUE INVOICES")

print(df["Invoice"].nunique())

print("\n4. UNIQUE PRODUCTS")

print(df["StockCode"].nunique())

print("\n5. UNIQUE CUSTOMERS")

print(df["Customer ID"].nunique())

print("\n6. UNIQUE COUNTRIES")

print(df["Country"].nunique())


# Check for missing values

print("\n7. MISSING VALUES")

print(df.isnull().sum())


# Check for duplicate rows

print("\n8. DUPLICATE ROWS")

print(df.duplicated().sum())


# Check quantity values

print("\n9. QUANTITY STATISTICS")

print(df["Quantity"].describe())

print("\n10. NEGATIVE QUANTITY RECORDS")

print((df["Quantity"] < 0).sum())

print("\n11. ZERO QUANTITY RECORDS")

print((df["Quantity"] == 0).sum())


# Check price values

print("\n12. PRICE STATISTICS")

print(df["Price"].describe())

print("\n13. ZERO PRICE RECORDS")

print((df["Price"] == 0).sum())

print("\n14. NEGATIVE PRICE RECORDS")

print((df["Price"] < 0).sum())


# Find cancelled invoices

print("\n15. CANCELLED INVOICES")

cancelled = df["Invoice"].astype(str).str.upper().str.startswith("C")

print("Cancelled invoice records:", cancelled.sum())

print("\nSample cancelled transactions:")

print(df[cancelled].head())


# Check the date range of the data

print("\n16. DATE RANGE")

print("Earliest transaction:")

print(df["InvoiceDate"].min())

print("Latest transaction:")

print(df["InvoiceDate"].max())


# Find the top 10 countries

print("\n17. TOP 10 COUNTRIES")

print(
    df["Country"]
    .value_counts()
    .head(10)
)


print("\n" + "=" * 70)

print("DATA QUALITY ANALYSIS COMPLETED")

print("=" * 70)