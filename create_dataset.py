import pandas as pd
import numpy as np

np.random.seed(42)

n = 200

size = np.random.randint(500, 3001, n)

bedrooms = np.clip(
    (size / 450 + np.random.normal(0, 0.7, n)).round().astype(int),
    1,
    6
)

bathrooms = np.clip(
    (bedrooms * 0.7 + np.random.normal(0, 0.5, n)).round().astype(int),
    1,
    5
)

age = np.random.randint(0, 31, n)

parking = np.clip(
    (bedrooms / 2 + np.random.normal(0, 0.7, n)).round().astype(int),
    0,
    3
)

price = (
    55000
    + size * 210
    + bedrooms * 18000
    + bathrooms * 22000
    - age * 2500
    + parking * 12000
    + np.random.normal(0, 25000, n)
)

price = np.maximum(price, 50000).round(-3).astype(int)

df = pd.DataFrame({
    "size": size,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "parking": parking,
    "price": price
})

df.to_csv("house_data.csv", index=False)

print("Dataset created successfully!")
print("Rows:", len(df))
print("\nFirst 5 rows:")
print(df.head())