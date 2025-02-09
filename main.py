import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Define file path for storing the synthetic soybean spread data

def soybean_spread(soybean_spread_csv = "data/soybean_spread.csv"):
    # Try to read the soybean spread data from the CSV if it exists
    try:
        soybean_df = pd.read_csv(soybean_spread_csv, parse_dates=['date'])
        print("Soybean spread data loaded from CSV.")
    except FileNotFoundError:
        print("Soybean spread CSV not found. Computing synthetic asset data.")

        # Read CSV files for soybean oil, meal, and mini contracts
        soybean_oil = pd.read_csv("data/Zl Soybean Oil.csv", header=0)
        soybean_meal = pd.read_csv("data/Zm Soybean Meal.csv", header=0)
        soybean_mini = pd.read_csv("data/Zs Soybean Mini.csv", header=0)

        # Rename the "time" column to "date" for consistency
        for df in [soybean_oil, soybean_meal, soybean_mini]:
            df.rename(columns={'time': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime
            df.dropna(subset=['date'], inplace=True)  # Remove rows with invalid dates

        # Convert numeric columns to proper types
        numeric_cols_csv = ['open', 'high', 'low', 'close', 'volume']
        for df in [soybean_oil, soybean_meal, soybean_mini]:
            df[numeric_cols_csv] = df[numeric_cols_csv].apply(pd.to_numeric, errors='coerce')

        # Compute synthetic spread components
        soybean_spread_open  = (11 * soybean_oil['open'])  + (48 * soybean_meal['open'])  - (50 * soybean_mini['open'])
        soybean_spread_high  = (11 * soybean_oil['high'])  + (48 * soybean_meal['high'])  - (50 * soybean_mini['low'])
        soybean_spread_low   = (11 * soybean_oil['low'])   + (48 * soybean_meal['low'])   - (50 * soybean_mini['high'])
        soybean_spread_close = (11 * soybean_oil['close']) + (48 * soybean_meal['close']) - (50 * soybean_mini['close'])
        soybean_spread_volume = (soybean_oil['volume'] + soybean_meal['volume'] + soybean_mini['volume']) // 3

        # Build the synthetic asset DataFrame
        soybean_df = pd.DataFrame({
            'date': soybean_oil['date'],
            'open': soybean_spread_open,
            'high': soybean_spread_high,
            'low': soybean_spread_low,
            'close': soybean_spread_close,
            'volume': soybean_spread_volume
        })

        # Normalize the date column
        soybean_df['date'] = soybean_df['date'].dt.normalize()

        # Save the synthetic soybean spread data to a CSV file
        soybean_df.to_csv(soybean_spread_csv, index=False)
        print("Synthetic soybean spread data saved to CSV.")

    return soybean_df

def weather_agg(table_name = "data.db"):

    # Connect to the database and read weather data
    conn = sqlite3.connect(table_name)
    weather_df = pd.read_sql_query("SELECT * FROM temperatures", conn)
    conn.close()

    # Convert 'date' to datetime and filter from 2011-01-01 onward
    weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
    weather_df = weather_df.dropna(subset=['date'])
    weather_df = weather_df[weather_df['date'] >= pd.to_datetime("2011-01-01")]

    # Select only relevant columns
    weather_numeric_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'tsun']
    weather_df[weather_numeric_cols] = weather_df[weather_numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Aggregate weather data by date
    weather_agg = weather_df.groupby('date').agg({
        'tavg': 'mean',
        'tmin': 'mean',
        'tmax': 'mean',
        'prcp': 'sum',
        'tsun': 'sum'
    }).reset_index()

    weather_agg['date'] = weather_agg['date'].dt.normalize()

    return weather_agg, weather_numeric_cols

def create_features(weather_df, weather_numeric_cols, soybean_df):
    # Compute rolling statistics (7-day mean and standard deviation)
    for col in weather_numeric_cols:
        weather_df[f"{col}_mean_7"] = weather_df[col].rolling(window=7).mean()
        weather_df[f"{col}_std_7"] = weather_df[col].rolling(window=7).std()

    # Drop original columns to reduce redundancy
    weather_df.drop(columns=weather_numeric_cols, inplace=True)
    weather_df.dropna(inplace=True)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)  # Reduce to 3 principal components
    weather_pca = pca.fit_transform(weather_df.drop(columns=['date']))
    weather_pca_df = pd.DataFrame(weather_pca, columns=['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7'])
    weather_pca_df['date'] = weather_df['date']

    # Ensure 'date' column in both DataFrames is timezone-naive
    soybean_df['date'] = soybean_df['date'].dt.tz_localize(None)
    weather_pca_df['date'] = weather_pca_df['date'].dt.tz_localize(None)

    # Now merge the data
    data_df = pd.merge(soybean_df, weather_pca_df, on='date', how='inner')

    # Merge the weather data with soybean data
    data_df = pd.merge(soybean_df, weather_pca_df, on='date', how='inner')

    return data_df, weather_df

def make_stationary(data_df):
    # Compute log returns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_cols:
        data_df[f'{col}_log_return'] = np.log(data_df[col] / data_df[col].shift(1))

    # Drop original OHLCV columns
    data_df.drop(columns=ohlcv_cols, inplace=True)
    data_df.dropna(inplace=True)

    # Define target as next day's log return
    data_df['target_log_return'] = data_df['close_log_return'].shift(-1)
    data_df.dropna(inplace=True)

    return data_df

def train(data_df):

    features = data_df.drop(['date', 'target_log_return'], axis=1)
    target = data_df['target_log_return']

    split_index = int(len(data_df) * 0.8)
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return y_pred, y_test, rmse

def backtest(y_pred, y_test):

    signals = np.where(y_pred > 0, 1, -1)
    actual_returns = y_test.values
    strategy_returns = signals * actual_returns

    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    cumulative_returns = np.cumprod(1 + strategy_returns)

    return sharpe_ratio, cumulative_returns

if __name__ == "__main__":

    data_df = soybean_spread()
    print("Soy spread calculated")

    weather_df, weather_numeric_cols = weather_agg()
    print("Weather agg calculated")

    data_df, weather_df = create_features(weather_df, weather_numeric_cols, data_df)
    print("Features created")
    data_df = make_stationary(data_df)
    print("Stationarity completed")

    y_pred, y_test, rmse = train(data_df)
    print("Trained")
    sharpe_ratio, cumulative_returns = backtest(y_pred, y_test)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Strategy Cumulative Returns', color='blue')
    plt.axhline(1, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Performance of Predicted % Returns')
    plt.legend()
    plt.show()

    # Display RMSE and Sharpe Ratio
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
