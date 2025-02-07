import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ================================
# 1. Read Soybean Crush Data and Compute Synthetic Price
# ================================

col_names = ['date', 'open', 'high', 'low', 'close', 'volume']

# Read the CSV files.
soybean_oil = pd.read_csv("data/Zl Soybean Oil.csv", skiprows=range(1, 501), header=None, names=col_names)
soybean_meal = pd.read_csv("data/Zm Soybean Meal.csv", skiprows=range(1, 501), header=None, names=col_names)
soybean_mini = pd.read_csv("data/Zs Soybean Mini.csv", skiprows=range(1, 501), header=None, names=col_names)

# Immediately convert numeric columns.
numeric_cols_csv = ['open', 'high', 'low', 'close', 'volume']
for df in [soybean_oil, soybean_meal, soybean_mini]:
    df[numeric_cols_csv] = df[numeric_cols_csv].apply(pd.to_numeric, errors='coerce')

print("soybean done")

conn = sqlite3.connect("data.db")
weather_df = pd.read_sql_query("SELECT * FROM temperatures", conn)
conn.close()

weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
weather_df = weather_df.dropna(subset=['date'])

# For the soybean CSV data (after parsing dates)
print("Soybean Oil raw date range:", soybean_oil['date'].min(), "to", soybean_oil['date'].max())
print("Soybean Meal raw date range:", soybean_meal['date'].min(), "to", soybean_meal['date'].max())
print("Soybean Mini raw date range:", soybean_mini['date'].min(), "to", soybean_mini['date'].max())

# For the weather data (after parsing dates)
print("Weather raw date range:", weather_df['date'].min(), "to", weather_df['date'].max())
