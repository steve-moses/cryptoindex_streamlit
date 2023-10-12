import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from joblib import load
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm


API_KEY = st.secrets["api_key"]

def fetch_kaiko_data(api_key, base_assets, quote_asset, start_time, end_time, page_size=1000):
    base_url = "https://us.market-api.kaiko.io/v2/data/trades.v1/spot_direct_exchange_rate"
    headers = {
        'Accept': 'application/json',
        'X-Api-Key': api_key
    }
    
    data_frames = []

    for base in tqdm(base_assets, desc="Fetching assets"):
        endpoint_url = f"{base_url}/{base}/{quote_asset}"
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "interval": "1d",
            "page_size": page_size
        }
        
        while True: # Loop until no continuation_token is found
            response = requests.get(endpoint_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()["data"]
                df = pd.DataFrame(data)
                df['asset'] = base
                data_frames.append(df)
                
                # Check for continuation_token and update params
                continuation_token = response.json().get("continuation_token", None)
                if continuation_token:
                    params = {
                        "continuation_token": continuation_token
                    }
                else:
                    break
            else:
                print(f"Failed to fetch data for {base}. Status Code: {response.status_code}")
                print(response.text)
                break
            
    final_df = pd.concat(data_frames, ignore_index=True)
    return final_df

# Parameters
BASE_ASSETS = ['bch', 'eth', 'xrp', 'ltc', 'dot'] #Top 5 non-Bitcoin and non-stablecoin cryptocurrencies by market cap as of January 3, 2021.
QUOTE_ASSET = 'usd'
START_TIME = "2021-01-03T00:00:00Z"
END_TIME = "2023-09-27T23:59:59Z"

df = fetch_kaiko_data(API_KEY, BASE_ASSETS, QUOTE_ASSET, START_TIME, END_TIME)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Rename asset codes to their full names
asset_name_map = {
    'bch': 'Bitcoin Cash',
    'eth': 'Ethereum',
    'xrp': 'Ripple',
    'ltc': 'Litecoin',
    'dot': 'Polkadot'
}

df['asset'] = df['asset'].map(asset_name_map)
df_pivot = df.pivot(index='timestamp', columns='asset', values='price')
df_pivot.sort_index(inplace=True)

for column in df_pivot.columns:
    df_pivot[column] = pd.to_numeric(df_pivot[column], errors='coerce')

# Construction of Index 
initial_average = df_pivot.iloc[0].mean()
initial_index_level = 1000
divisor = initial_average / initial_index_level
df_pivot['average_price'] = df_pivot.mean(axis=1)
df_pivot['Index'] = df_pivot['average_price'] / divisor

# Build StreamLit app

# Dashboard
st.title('Cryptocurrency Portfolio Index & Analytics Suite')
component_assets = list(asset_name_map.values())
all_assets = ['Index'] + component_assets
selected_assets = st.multiselect('Select cryptocurrencies for comparision', component_assets, default=component_assets)

fig = go.Figure()
norm_factor_index = df_pivot['Index'].iloc[0]
fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Index'] / norm_factor_index, mode='lines', name='Normalized Index'))

for asset in selected_assets:
    norm_factor_asset = df_pivot[asset].iloc[0]
    fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[asset] / norm_factor_asset, mode='lines', name=asset))

fig.update_layout(title='Cryptocurrencies vs. Index Value', xaxis_title='Date', yaxis_title='Normalized Value', template="plotly_dark")
st.plotly_chart(fig)

st.write('Note: The data is normalized, so both asset prices and the index start at the same initial value for better comparison.')

# Plotting daily returns

daily_returns = df_pivot[selected_assets].pct_change().dropna()
initial_value = df_pivot['Index'].iloc[-1]
st.subheader('Daily Returns of Component Assets')
fig = go.Figure()

for asset in selected_assets:
    fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns[asset], mode='lines', name=asset))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Daily Return',
    template="plotly_white"
)

st.plotly_chart(fig)

# Portfolio Simulation and VaR 

st.subheader("Simulated Portfolio Returns and Value at Risk (VaR)")
st.write("Value at Risk (VaR) estimates the potential loss an investment portfolio might face over a specified period for a given confidence interval.")
num_simulations = st.slider("Select number of Simulations", 1000, 20000, 10000)  # Slider for simulations
st.write("The confidence level represents the likelihood that losses will not exceed the estimated VaR. For instance, a 95% confidence level means that there's a 95% chance that losses won't exceed the VaR amount in the specified period.")
confidence_level = st.slider("Select confidence level for VaR", 0.80, 0.99, 0.95, 0.01) # Slider for Confidence Level

# Run Monte Carlo Simulation for VaR
agg_data = daily_returns.mean(axis=1)
mean = agg_data.mean()
std_dev = agg_data.std()
simulated_returns = np.random.normal(loc=mean, scale=std_dev, size=num_simulations)
sorted_returns = np.sort(simulated_returns)

# Calculate VaR for Monte Carlo returns
var_index = int((1 - confidence_level) * num_simulations)
var = sorted_returns[var_index]
var_value = initial_value * var

# Simulate returns with Cholesky Decomposition
cov_matrix = daily_returns.cov()
L = np.linalg.cholesky(cov_matrix)
simulated_returns_chol = [np.dot(L, norm.rvs(size=len(selected_assets))) for _ in range(num_simulations)]
simulated_returns_chol = np.array(simulated_returns_chol)

# Calculate VaR for Cholesky-based returns
sorted_cholesky_returns = np.sort(simulated_returns_chol[:, 0])
var_cholesky = sorted_cholesky_returns[var_index]
var_value_cholesky = initial_value * var_cholesky


# Histogram visualization
fig_hist = go.Figure()

fig_hist.add_trace(go.Histogram(x=simulated_returns, 
                                name='Monte Carlo based Simulated Returns', 
                                marker_color='blue', 
                                opacity=0.3))
fig_hist.add_trace(go.Histogram(x=simulated_returns_chol[:, 0], 
                                name='Cholesky-based Simulated Returns', 
                                marker_color='red', 
                                opacity=0.3))
fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                              line=dict(color="blue", width=2, dash="dash"),
                              name=f"VaR (Monte Carlo, {confidence_level * 100}% Confidence)"))

fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                              line=dict(color="red", width=2, dash="dash"),
                              name=f"VaR (Cholesky, {confidence_level * 100}% Confidence)"))

fig_hist.add_shape(go.layout.Shape(
    type="line",
    x0=var,
    x1=var,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color="red", width=2, dash="dash")
))
fig_hist.add_shape(go.layout.Shape(
    type="line",
    x0=var_cholesky,
    x1=var_cholesky,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color="blue", width=2, dash="dash")
))

fig_hist.update_layout(title=f'Histogram of Simulated Returns after {num_simulations} simulations',
                       xaxis_title='Returns',
                       yaxis_title='Frequency',
                       barmode='overlay',
                       template="plotly_dark")
st.plotly_chart(fig_hist)


# Technical Indicators
st.subheader('Technical Indicators') 
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_bollinger_bands(data, window=20, num_std_dev=2):
    sma = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    
    return upper_band, lower_band

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    return macd_line, signal_line


crypto_choice = st.selectbox("Choose an asset to view its technical indicators:", all_assets)

# SMA
st.write('### Simple Moving Average (SMA)')
sma_window = st.slider("SMA Window", 30, 100, 50)
df_pivot['SMA'] = df_pivot[crypto_choice].rolling(window=sma_window).mean()

fig_sma = go.Figure()
fig_sma.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[crypto_choice], mode='lines', name='Asset Price'))
fig_sma.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['SMA'], mode='lines', name='SMA', line=dict(color='orange')))
fig_sma.update_layout(title=f'{crypto_choice} SMA ({sma_window}-Day Rolling)', xaxis_title='Date', yaxis_title='Value', template="plotly_dark")
st.plotly_chart(fig_sma)

# Volatility 
st.write('### Volatility')
vol_window = st.slider(f"Choose the rolling window size for {crypto_choice} volatility calculation:", 10, 120, 30)

smoothed_volatility = df_pivot[crypto_choice].pct_change().rolling(window=vol_window).std() * np.sqrt(365)
fig_volatility = go.Figure()
fig_volatility.add_trace(go.Scatter(x=smoothed_volatility.index, y=smoothed_volatility, mode='lines', name=f'{crypto_choice.upper()} Volatility', line=dict(color='blue')))
fig_volatility.update_layout(title=f'{crypto_choice} Smoothed Volatility ({vol_window}-Day Rolling)',
                            xaxis_title='Date',
                            yaxis_title='Volatility',
                            template="plotly_dark")
st.plotly_chart(fig_volatility)
# RSI
st.write('### Relative Strength Index (RSI)')
rsi_window = st.slider("RSI Window", 7, 14, 21)
df_pivot['RSI'] = compute_rsi(df_pivot[crypto_choice], rsi_window)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['RSI'], mode='lines', name='RSI'))
fig_rsi.update_layout(title=f'{crypto_choice} RSI ({rsi_window}-Day Rolling)', xaxis_title='Date', yaxis_title='RSI', template="plotly_dark")
st.plotly_chart(fig_rsi)

# Bollinger Bands
st.write('### Bollinger Bands')
bb_window = st.slider("Bollinger Bands Window", 10, 20, 40)
num_std_dev = st.slider("Bollinger Bands Standard Deviations", 1, 3, 2)
df_pivot['Upper_Bollinger'], df_pivot['Lower_Bollinger'] = compute_bollinger_bands(df_pivot[crypto_choice], bb_window, num_std_dev)

fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[crypto_choice], mode='lines', name='Asset Price'))
fig_bb.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Upper_Bollinger'], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
fig_bb.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Lower_Bollinger'], mode='lines', name='Lower Bollinger Band', line=dict(color='green')))
fig_bb.update_layout(title=f'{crypto_choice} Bollinger Bands)', xaxis_title='Date', yaxis_title='Value', template="plotly_dark")
st.plotly_chart(fig_bb)

# MACD
st.write('### Moving Average Convergence Divergence (MACD)')
macd_short_window = st.slider("MACD Short Window", 5, 15, 12)
macd_long_window = st.slider("MACD Long Window", 20, 40, 26)
df_pivot['MACD'], df_pivot['Signal_Line'] = compute_macd(df_pivot[crypto_choice], macd_short_window, macd_long_window)

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['MACD'], mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Signal_Line'], mode='lines', name='Signal Line'))
fig_macd.update_layout(title=f'{crypto_choice} MACD ', xaxis_title='Date', yaxis_title='Value', template="plotly_dark")
st.plotly_chart(fig_macd)