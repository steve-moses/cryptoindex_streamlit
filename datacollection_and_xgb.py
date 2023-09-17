# This file is used to extract the data required for the dashboard and is used to train pretrain and dump the XGBoost model locally

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from dotenv import load_dotenv
import requests
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from joblib import dump

# Function to fetch Kaiko data
API_KEY = os.environ.get('KAIKO_API_KEY')
def fetch_kaiko_data(api_key, base_assets, quote_asset, start_time, end_time):
    base_url = "https://us.market-api.kaiko.io/v2/data/trades.v1/spot_direct_exchange_rate"
    headers = {
        'Accept': 'application/json',
        'X-Api-Key': api_key
    }
    
    data_frames = []

    for base in base_assets:
        quote = 'usd'
        endpoint_url = f"{base_url}/{base}/{quote_asset}"
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "interval": "1d",
            "page_size": 1000
        }
        
        response = requests.get(endpoint_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()["data"]
            df = pd.DataFrame(data)
            df['asset'] = base
            data_frames.append(df)
        else:
            print(f"Failed to fetch data for {base}. Status Code: {response.status_code}")
            print(response.text)
            
    final_df = pd.concat(data_frames, ignore_index=True)
    return final_df

# Parameters
API_KEY = os.getenv("KAIKO_API_KEY")
BASE_ASSETS = ['bch', 'eth', 'xrp', 'ltc', 'dot']
QUOTE_ASSET = 'usd'
START_TIME = "2021-01-03T00:00:00Z"
END_TIME = "2023-09-15T23:59:59Z"

df = fetch_kaiko_data(API_KEY, BASE_ASSETS, QUOTE_ASSET, START_TIME, END_TIME)
df.to_csv('data.csv', index=False)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df_pivot = df.pivot(index='timestamp', columns='asset', values='price')
df_pivot.sort_index(inplace=True)

for column in df_pivot.columns:
    df_pivot[column] = pd.to_numeric(df_pivot[column], errors='coerce')

initial_average = df_pivot.iloc[0].mean()
initial_index_level = 1000
divisor = initial_average / initial_index_level
df_pivot['average_price'] = df_pivot.mean(axis=1)
df_pivot['index_level'] = df_pivot['average_price'] / divisor

# Feature Engineering: Incorporating rolling mean as an additional feature
rolling_window_size = 7
df_pivot['rolling_mean'] = df_pivot['index_level'].rolling(window=rolling_window_size).mean().shift(-rolling_window_size + 1)

df_xgb = df_pivot.reset_index().copy()
df_xgb['days_from_start'] = (df_xgb['timestamp'] - df_xgb['timestamp'].min()).dt.days

X = df_xgb[['days_from_start', 'rolling_mean']].dropna()
y = df_xgb['index_level'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'colsample_bytree': [0.7, 0.8, 0.9, 1],
    'gamma': [0, 0.1, 0.2]
}
xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
xgb_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=10, cv=3, verbose=3, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
dump(xgb_search.best_estimator_, 'xgb_model.joblib')
