import pandas as pd

file_path = "data/raw/online_retail_II.xlsx"

# Load Excel file
df = pd.read_excel(file_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("\n1. Dataset Shape")
print(df.shape)

print("\n2. Column Names")
print(df.columns.tolist())

print("\n3. First 5 Rows")
print(df.head())

print("\n4. Data Types")
print(df.dtypes)

print("\n5. Missing Values")
print(df.isnull().sum())