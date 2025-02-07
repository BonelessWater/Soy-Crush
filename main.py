import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ================================
# 1. Read Soybean Crush Data and Compute Synthetic Price
# ================================

# Read CSV files (using the header in the CSV file)
soybean_oil = pd.read_csv("data/Zl Soybean Oil.csv", header=0)
soybean_meal = pd.read_csv("data/Zm Soybean Meal.csv", header=0)
soybean_mini = pd.read_csv("data/Zs Soybean Mini.csv", header=0)

# Rename the "time" column to "date" for consistency.
soybean_oil.rename(columns={'time': 'date'}, inplace=True)
soybean_meal.rename(columns={'time': 'date'}, inplace=True)
soybean_mini.rename(columns={'time': 'date'}, inplace=True)

# Convert numeric columns to proper types.
numeric_cols_csv = ['open', 'high', 'low', 'close', 'volume']
for df in [soybean_oil, soybean_meal, soybean_mini]:
    df[numeric_cols_csv] = df[numeric_cols_csv].apply(pd.to_numeric, errors='coerce')

# Parse the 'date' column.
for df in [soybean_oil, soybean_meal, soybean_mini]:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Remove any rows with invalid dates.
soybean_oil = soybean_oil.dropna(subset=['date'])
soybean_meal = soybean_meal.dropna(subset=['date'])
soybean_mini = soybean_mini.dropna(subset=['date'])

# Remove timezone information from the date columns so they become tz-naive.
for df in [soybean_oil, soybean_meal, soybean_mini]:
    df['date'] = df['date'].dt.tz_convert(None)

# Compute synthetic spread components.
soybean_spread_open  = (11 * soybean_oil['open'])  + (48 * soybean_meal['open'])  - (50 * soybean_mini['open'])
soybean_spread_high  = (11 * soybean_oil['high'])  + (48 * soybean_meal['high'])  - (50 * soybean_mini['low'])
soybean_spread_low   = (11 * soybean_oil['low'])   + (48 * soybean_meal['low'])   - (50 * soybean_mini['high'])
soybean_spread_close = (11 * soybean_oil['close']) + (48 * soybean_meal['close']) - (50 * soybean_mini['close'])
soybean_spread_volume = (soybean_oil['volume'] + soybean_meal['volume'] + soybean_mini['volume']) // 3

# Build the synthetic asset DataFrame.
soybean_df = pd.DataFrame({
    'date': soybean_oil['date'],
    'open': soybean_spread_open,
    'high': soybean_spread_high,
    'low': soybean_spread_low,
    'close': soybean_spread_close,
    'volume': soybean_spread_volume
})

# Normalize the date (remove time-of-day information)
soybean_df['date'] = soybean_df['date'].dt.normalize()

# ================================
# 2. Retrieve and Aggregate Weather Data from SQLite
# ================================

conn = sqlite3.connect("data.db")
weather_df = pd.read_sql_query("SELECT * FROM temperatures", conn)
conn.close()

# Convert weather dates to datetime.
weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
weather_df = weather_df.dropna(subset=['date'])
weather_numeric_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wpgt', 'pres', 'tsun']
weather_df[weather_numeric_cols] = weather_df[weather_numeric_cols].apply(pd.to_numeric, errors='coerce')

# Aggregate weather data by date.
weather_agg = weather_df.groupby('date').agg({
    'tavg': 'mean',
    'tmin': 'mean',
    'tmax': 'mean',
    'prcp': 'sum',
    'snow': 'sum',
    'wdir': 'mean',
    'wpgt': 'mean',
    'pres': 'mean',
    'tsun': 'sum'
}).reset_index()

# Normalize weather dates.
weather_agg['date'] = weather_agg['date'].dt.normalize()

# ================================
# 3. Filter Both Datasets to the Common Date Range
# ================================

# Use data from 2011-01-01 onward.
start_date = pd.to_datetime("2013-01-01")
soybean_df = soybean_df[soybean_df['date'] >= start_date]
weather_agg = weather_agg[weather_agg['date'] >= start_date]

# Determine the common end date (the earliest maximum date from both datasets).
if not soybean_df.empty and not weather_agg.empty:
    common_end_date = min(soybean_df['date'].max(), weather_agg['date'].max())
    soybean_df = soybean_df[soybean_df['date'] <= common_end_date]
    weather_agg = weather_agg[weather_agg['date'] <= common_end_date]
else:
    print("One of the datasets is empty after filtering by start_date!")

print("Soybean date range:", soybean_df['date'].min(), "to", soybean_df['date'].max())
print("Weather date range:", weather_agg['date'].min(), "to", weather_agg['date'].max())

# ================================
# 4. Merge Soybean Data with Weather Data
# ================================

data_df = pd.merge(soybean_df, weather_agg, on='date', how='inner')
data_df.sort_values('date', inplace=True)
print("Merged DataFrame has", len(data_df), "rows.")

numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wpgt', 'pres', 'tsun']
data_df[numeric_cols] = data_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Optionally, handle missing weather data (e.g., forward-fill).
data_df[numeric_cols[5:]] = data_df[numeric_cols[5:]].fillna(method='ffill')

# ================================
# 5. Prepare Features and Target for ML Model
# ================================

target = data_df['close']
features = data_df.drop(['date', 'close'], axis=1)
print("Feature columns:", features.columns.tolist())

# ================================
# 6. Train/Test Split (for reference)
# ================================
# Here we first train a model on 80% of the data and evaluate on the remaining 20%.
split_index = int(len(data_df) * 0.8)
X_train = features.iloc[:split_index]
X_test  = features.iloc[split_index:]
y_train = target.iloc[:split_index]
y_test  = target.iloc[split_index:]

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE (static split):", rmse)

# ================================
# 7. Walk-Forward Backtesting
# ================================
# We use an expanding window backtest. For each day (starting after an initial training period),
# we train the model on all prior data, then predict the next day's close.
# Our trading rule is:
#    - If the predicted price is higher than the previous day's close, take a long position (signal = +1).
#    - Otherwise, take a short position (signal = -1).
# Daily trade return is computed as:
#    signal * ( (today's actual close - previous day's actual close) / previous day's actual close ).
# We then compute the cumulative return.

initial_train_period = 200  # e.g., first 200 observations used for initial training
signals = []      # trading signals: +1 for long, -1 for short
predictions = []  # predicted prices
backtest_dates = []  # dates for each trade

# Loop from the end of the initial training period until the end of our dataset.
for i in range(initial_train_period, len(data_df)):
    train_data = data_df.iloc[:i]
    test_data = data_df.iloc[i:i+1]  # predict one day ahead
    
    X_train_back = train_data.drop(['date', 'close'], axis=1)
    y_train_back = train_data['close']
    X_test_back = test_data.drop(['date', 'close'], axis=1)
    
    # Train a new model on the expanding window
    model_bt = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_bt.fit(X_train_back, y_train_back)
    
    # Predict the close price for the current test day
    pred_price = model_bt.predict(X_test_back)[0]
    predictions.append(pred_price)
    backtest_dates.append(test_data['date'].iloc[0])
    
    # Define signal based on comparison with the previous day's actual close.
    # (If predicted > previous close, we go long; otherwise, short.)
    prev_close = train_data['close'].iloc[-1]
    if pred_price > prev_close:
        signals.append(1)
    else:
        signals.append(-1)

# Compute daily returns based on actual close prices.
# For each backtest day (i from initial_train_period to end), the trade return is computed as:
#    trade_return = signal * ( (price_today - price_yesterday) / price_yesterday )
actual_prices = data_df['close'].values  # all actual close prices
trade_returns = []
for j, i in enumerate(range(initial_train_period, len(data_df))):
    # Use previous day's close (i-1) and today's close (i)
    daily_return = (actual_prices[i] - actual_prices[i - 1]) / actual_prices[i - 1]
    trade_return = signals[j] * daily_return
    trade_returns.append(trade_return)

# Compute cumulative returns (starting from 1)
cumulative_returns = np.cumprod([1 + r for r in trade_returns])

# Build a DataFrame to store backtesting results.
backtest_results = pd.DataFrame({
    'date': backtest_dates,
    'predicted_close': predictions,
    'signal': signals,
    'daily_return': trade_returns,
    'cumulative_return': cumulative_returns
})

print(backtest_results.head())
print("Final cumulative return: {:.2%}".format(cumulative_returns[-1] - 1))

# ================================
# 8. Plot Backtest Performance
# ================================
plt.figure(figsize=(12, 6))
plt.plot(backtest_results['date'], backtest_results['cumulative_return'], label='Cumulative Return', color='blue')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Walk-Forward Backtest Performance')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
