import pandas as pd
file_path = "data/raw/online_retail_II.xlsx"
df = pd.read_excel(file_path)

# Check transactions with zero price
zero_price = df[df["Price"] == 0]
print("=" * 70)
print("ZERO PRICE TRANSACTION ANALYSIS")
print("=" * 70)
print("\nNumber of zero-price records:")
print(len(zero_price))
print("\nSample zero-price records:")
print(zero_price.head(20))
print("\nQuantity statistics for zero-price records:")
print(zero_price["Quantity"].describe())
print("\nTop products with zero price:")
print(
    zero_price["StockCode"]
    .value_counts()
    .head(20)
)
print("\nCountries with zero-price transactions:")
print(
    zero_price["Country"]
    .value_counts()
    .head(20)
)

# Check transactions with negative price
negative_price = df[df["Price"] < 0]
print("\n" + "=" * 70)
print("NEGATIVE PRICE TRANSACTION ANALYSIS")
print("=" * 70)
print("\nNegative-price records:")
print(negative_price)

# Check transactions with large quantity
large_quantity = df[df["Quantity"] > 1000]
print("\n" + "=" * 70)
print("LARGE QUANTITY ANALYSIS")
print("=" * 70)
print("\nRecords with Quantity > 1000:")
print(len(large_quantity))
print("\nSample:")
print(large_quantity.head(20))
print("\n" + "=" * 70)
print("INVESTIGATION COMPLETED")
print("=" * 70)