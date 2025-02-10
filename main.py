import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import optuna
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split

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

# ================================
# 2. Retrieve and Aggregate Weather Data from SQLite (Filtered from 2011-01-01 Onward)
# ================================

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

# ================================
# 3. Apply Rolling Statistics & PCA to Weather Data
# ================================

def create_features(weather_df, weather_numeric_cols, soybean_df):
    # Compute rolling statistics (7-day mean and standard deviation)
    for col in weather_numeric_cols:
        weather_df[f"{col}_mean_7"] = weather_df[col].rolling(window=7).mean()
        weather_df[f"{col}_std_7"] = weather_df[col].rolling(window=7).std()

    # Drop original columns to reduce redundancy
    weather_df.drop(columns=weather_numeric_cols, inplace=True)
    weather_df.dropna(inplace=True)

    # Check the number of samples and features
    n_samples, n_features = weather_df.drop(columns=['date']).shape
    print(f"Number of samples: {n_samples}, Number of features: {n_features}")

    # Adjust num_components based on available features
    num_components = min(40, n_features, n_samples) # 10 features

    print(f"Applying PCA with n_components={num_components}")

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=num_components)
    weather_pca = pca.fit_transform(weather_df.drop(columns=['date']))

    # Create a DataFrame with dynamic column names
    weather_pca_df = pd.DataFrame(weather_pca, columns=[f'pca{i+1}' for i in range(num_components)])
    weather_pca_df['date'] = weather_df['date']

    # Ensure 'date' column in both DataFrames is timezone-naive
    soybean_df['date'] = soybean_df['date'].dt.tz_localize(None)
    weather_pca_df['date'] = weather_pca_df['date'].dt.tz_localize(None)

    # Now merge the data
    data_df = pd.merge(soybean_df, weather_pca_df, on='date', how='inner')

    # Merge the weather data with soybean data
    data_df = pd.merge(soybean_df, weather_pca_df, on='date', how='inner')

    return data_df, weather_df

# ================================
# 4. Compute First Derivative (Log Returns) to Make Data Stationary
# ================================

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

# ================================
# 5. Train Model on Log Returns
# ================================

def train(data_df):

    features = data_df.drop(['date', 'target_log_return'], axis=1)
    target = data_df['target_log_return']

    split_index = int(len(data_df) * 0.95)
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=10000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return y_pred, y_test, rmse

# ================================
# 6. Backtesting & Sharpe Ratio
# ================================

def backtest(y_pred, y_test):

    signals = np.where(y_pred > 0, 1, -1)
    actual_returns = y_test.values
    strategy_returns = signals * actual_returns

    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    cumulative_returns = np.cumprod(1 + strategy_returns)

    return sharpe_ratio, cumulative_returns

# ================================
# 7. Plot Backtest Performance
# ================================

if __name__ == "__main__":
    data_df = soybean_spread()
    print("Soy spread calculated")

    weather_df, weather_numeric_cols = weather_agg()
    print("Weather agg calculated")

    # Create features based on your weather data and soybean spread
    data_df, weather_df = create_features(weather_df, weather_numeric_cols, data_df)
    print("Features created")

    # Make the series stationary
    data_df = make_stationary(data_df)
    print("Stationarity completed")

    # Prepare feature matrix X and target vector y.
    # (Adjust the column names if your data has different naming.)
    drop_cols = ['target_log_return', 'price', 'volume', 'log_return']  # extra columns you may not want as features
    y = (data_df['target_log_return'] > 0).astype(int)
    X = data_df.drop([col for col in drop_cols if col in data_df.columns], axis=1)
    X['date'] = X['date'].apply(lambda d: d.toordinal())

    # =============================================================================
    # 2. Hyperparameter Optimization with Optuna
    # =============================================================================

    def objective(trial, X, y):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eval_metric': 'auc'
        }
        # Split a portion of the data for validation
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        
        clf = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            random_state=42,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            device='cuda'
        )
        clf.fit(X_train_opt, y_train_opt)
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, y_val_prob)
        return auc_val

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=200, show_progress_bar=True)

    # Retrieve the best parameters and update with GPU options and eval_metric
    best_params = study.best_params
    best_params.update({
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'device': 'cuda',
        'use_label_encoder': False
    })
    print("Best Params:", best_params)

    # =============================================================================
    # 3. Create the Final Pipeline
    # =============================================================================

    final_pipeline = ImbPipeline([
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('feature_selection', SelectFromModel(
            xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        )),
        ('classifier', xgb.XGBClassifier(**best_params, random_state=42))
    ])

    # =============================================================================
    # 4. Train-Test Split and Model Training
    # =============================================================================

    # Use an 85/15 train–test split
    split_point = int(0.85 * len(X))
    # If X is a DataFrame, use .iloc for indexing
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # Fit the pipeline on the training data
    final_pipeline.fit(X_train, y_train)

    # =============================================================================
    # 5. Predictions and Evaluation Metrics
    # =============================================================================

    # Make predictions and predict probabilities for AUC calculation
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # =============================================================================
    # 6. Backtesting (Optional)
    # =============================================================================

    # If your strategy includes a backtest function that takes the predicted signals,
    # you can run it here. (Ensure your backtest() function is compatible with binary signals.)
    try:
        sharpe_ratio, cumulative_returns = backtest(y_pred, y_test)
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns, label='Strategy Cumulative Returns', color='blue')
        plt.axhline(1, linestyle='--', color='gray', alpha=0.5)
        plt.xlabel('Days')
        plt.ylabel('Cumulative Return')
        plt.title('Backtest Performance of Predicted Signals')
        plt.legend()
        plt.show()
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    except Exception as e:
        print("Error during backtesting:", e)

    # =============================================================================
    # 7. Save the Trained Model and Pipeline
    # =============================================================================

    # Extract the XGBoost classifier from the pipeline and save it in JSON format
    xgb_classifier = final_pipeline.named_steps['classifier']
    xgb_classifier.save_model("xgboost_model.json")

    # Save the entire pipeline using joblib
    joblib.dump(final_pipeline, "final_pipeline.pkl")

    # =============================================================================
    # 8. Feature Importance Plot
    # =============================================================================

    # Get feature importances from the classifier (make sure your feature selection step does not alter the order)
    feature_importance = xgb_classifier.feature_importances_
    feature_names = X.columns  # Using the original feature names from X
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 10 Feature Importances')
    plt.show()