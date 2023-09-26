import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
from scipy.stats import norm
import plotly.graph_objects as go
import requests
import plotly.express as px


API_KEY = st.secrets["api_key"]
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
BASE_ASSETS = ['bch', 'eth', 'xrp', 'ltc', 'dot']
QUOTE_ASSET = 'usd'
START_TIME = "2021-01-03T00:00:00Z"
END_TIME = "2023-09-15T23:59:59Z"

df = fetch_kaiko_data(API_KEY, BASE_ASSETS, QUOTE_ASSET, START_TIME, END_TIME)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df_pivot = df.pivot(index='timestamp', columns='asset', values='price')
df_pivot.sort_index(inplace=True)

for column in df_pivot.columns:
    df_pivot[column] = pd.to_numeric(df_pivot[column], errors='coerce')

# Aggreagate prices of individual assets
initial_average = df_pivot.iloc[0].mean()
initial_index_level = 1000
divisor = initial_average / initial_index_level
df_pivot['average_price'] = df_pivot.mean(axis=1)
df_pivot['index_level'] = df_pivot['average_price'] / divisor

# Build StreamLit app
st.title('Cryptocurrency Index Dashboard')
selected_assets = st.multiselect('Select cryptocurrencies for analysis', BASE_ASSETS, default=BASE_ASSETS)

if not selected_assets:
    st.write("Please select at least one cryptocurrency for analysis.")
else:
    fig = go.Figure()

    norm_factor_index = df_pivot['index_level'].iloc[0]
    fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['index_level'] / norm_factor_index, mode='lines', name='Normalized Index'))

    for asset in selected_assets:
        norm_factor_asset = df_pivot[asset].iloc[0]
        fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[asset] / norm_factor_asset, mode='lines', name=asset))

    fig.update_layout(title='Normalized Cryptocurrencies vs. Index', xaxis_title='Date', yaxis_title='Normalized Value', template="plotly_dark")
    st.plotly_chart(fig)

    st.write('Note: The data is normalized, so both asset prices and the index start at the same initial value for better comparison.')


st.subheader("Simulated Portfolio Returns and Value at Risk (VaR)")

# Create the slider for simulations
st.write("Value at Risk (VaR) estimates the potential loss an investment portfolio might face over a specified period for a given confidence interval.")
num_simulations = st.slider("Select number of Simulations", 1000, 20000, 10000)

# Create the slider for Confidence Level
confidence_level = st.slider("Select confidence level for VaR", 0.80, 0.99, 0.95, 0.01)
st.write("The confidence level represents the likelihood that losses will not exceed the estimated VaR. For instance, a 95% confidence level means that there's a 95% chance that losses won't exceed the VaR amount in the specified period.")

# Daily returns
daily_returns = df_pivot[selected_assets].pct_change().dropna()
initial_value = df_pivot['index_level'].iloc[-1]

# Run Monte Carlo Simulation for VaR
agg_data = daily_returns.mean(axis=1)
mean = agg_data.mean()
std_dev = agg_data.std()
simulated_returns = np.random.normal(loc=mean, scale=std_dev, size=num_simulations)
sorted_returns = np.sort(simulated_returns)

# Calculate VaR for standard returns
var_index = int((1 - confidence_level) * num_simulations)
var = sorted_returns[var_index]
var_value = initial_value * var

# Simulated returns with Cholesky Decomposition
cov_matrix = daily_returns.cov()
L = np.linalg.cholesky(cov_matrix)
simulated_returns_chol = [np.dot(L, norm.rvs(size=len(selected_assets))) for _ in range(num_simulations)]
simulated_returns_chol = np.array(simulated_returns_chol)

# Calculate VaR for Cholesky-based returns
sorted_cholesky_returns = np.sort(simulated_returns_chol[:, 0])
var_cholesky = sorted_cholesky_returns[var_index]
var_value_cholesky = initial_value * var_cholesky


# Combined histogram visualization
fig_hist = go.Figure()

fig_hist.add_trace(go.Histogram(x=simulated_returns, 
                                name='Monte Carlo based Simulated Returns', 
                                marker_color='blue', 
                                opacity=0.3))
fig_hist.add_trace(go.Histogram(x=simulated_returns_chol[:, 0], 
                                name='Cholesky-based Simulated Returns', 
                                marker_color='red', 
                                opacity=0.3))

# Add dummy traces for VaRs to show them in legend
fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                              line=dict(color="blue", width=2, dash="dash"),
                              name=f"VaR (Monte Carlo, {confidence_level * 100}% Confidence)"))

fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                              line=dict(color="red", width=2, dash="dash"),
                              name=f"VaR (Cholesky, {confidence_level * 100}% Confidence)"))

# Now, add the actual VaR lines as shapes
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


st.subheader('Future Index Prediction using XGBoost')
model = load('xgb_model.joblib')
rolling_window_size = 7
df_pivot['rolling_mean'] = df_pivot['index_level'].rolling(window=rolling_window_size).mean().shift(-rolling_window_size + 1)
df_xgb = df_pivot.reset_index().copy()
df_xgb['days_from_start'] = (df_xgb['timestamp'] - df_xgb['timestamp'].min()).dt.days
df_xgb['rolling_mean'] = df_pivot['index_level'].rolling(window=rolling_window_size).mean().shift(-rolling_window_size + 1)
X = df_xgb[['days_from_start', 'rolling_mean']]
y = df_xgb['index_level'].loc[X.index]

# Predict using xgb
xgb_predictions = model.predict(X)

# Plot the predictions
fig_predictions = go.Figure()
fig_predictions.add_trace(go.Scatter(x=df_xgb['timestamp'], y=y, mode='lines', name='Actual Data', line=dict(color='blue')))
fig_predictions.add_trace(go.Scatter(x=df_xgb['timestamp'], y=xgb_predictions, mode='lines', name='XGBoost Prediction', line=dict(color='red')))
fig_predictions.update_layout(title='Index Value prediction',
                      xaxis_title='Date',
                      yaxis_title='Index Level',
                      template="plotly_dark")
st.plotly_chart(fig_predictions)


# Historical Volatility
st.subheader('Historical Volatility')
crypto_choice = st.selectbox("Choose a cryptocurrency to view its historical volatility:", selected_assets)
window_size = st.slider("Choose the rolling window size for volatility calculation:", 10, 120, 30)

smoothed_volatility = df_pivot[crypto_choice].pct_change().rolling(window=window_size).std() * np.sqrt(252)

# Create a Plotly figure for smoothed volatility
fig_volatility = go.Figure()
fig_volatility.add_trace(go.Scatter(x=smoothed_volatility.index, y=smoothed_volatility, mode='lines', name=f'{crypto_choice.upper()} Volatility', line=dict(color='blue')))
fig_volatility.update_layout(title=f'{crypto_choice.upper()} Smoothed Volatility ({window_size}-Day Rolling)',
                            xaxis_title='Date',
                            yaxis_title='Volatility',
                            template="plotly_dark")
st.plotly_chart(fig_volatility)

