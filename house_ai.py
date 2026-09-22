import pandas as pd

# Load dataset
df = pd.read_csv("house_data.csv")

# Display first 5 rows
print("\n===== FIRST 5 ROWS =====")
print(df.head())

# Dataset information
print("\n===== DATASET INFO =====")
print(df.info())

# Statistical summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATASET SHAPE =====")
print("Rows :", df.shape[0])
print("Columns :", df.shape[1])


# ==============================
# GRAPHS
# ==============================

import matplotlib.pyplot as plt

# House Size vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df["size"], df["price"])
plt.xlabel("House Size (sq.ft)")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.show()

# Bedrooms vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df["bedrooms"], df["price"])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs Price")
plt.show()


# ==============================
# MACHINE LEARNING MODEL
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Features
X = df[["size", "bedrooms", "bathrooms", "age", "parking"]]

# Target
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Example house
new_house = [[2000, 4, 3, 5, 2]]

predicted_price = model.predict(new_house)

print("\n===== HOUSE PRICE PREDICTION =====")
print("Predicted Price: ₹", round(predicted_price[0], 2))
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ------------------------------
# Model 1: Linear Regression
# ------------------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)


# ------------------------------
# Model 2: Random Forest
# ------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)


# ------------------------------
# Model 3: Gradient Boosting
# ------------------------------

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)


# ==========================================
# DISPLAY MODEL COMPARISON
# ==========================================

print("\n================================")
print("       MODEL COMPARISON")
print("================================")

print("\nLinear Regression")
print("MAE  :", round(linear_mae, 2))
print("RMSE :", round(linear_rmse, 2))
print("R2   :", round(linear_r2, 4))

print("\nRandom Forest")
print("MAE  :", round(rf_mae, 2))
print("RMSE :", round(rf_rmse, 2))
print("R2   :", round(rf_r2, 4))

print("\nGradient Boosting")
print("MAE  :", round(gb_mae, 2))
print("RMSE :", round(gb_rmse, 2))
print("R2   :", round(gb_r2, 4))