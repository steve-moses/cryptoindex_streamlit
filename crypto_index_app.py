import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
from scipy.stats import norm
import plotly.graph_objects as go
import requests
import plotly.express as px

# Parameters
BASE_ASSETS = ['bch', 'eth', 'xrp', 'ltc', 'dot']
QUOTE_ASSET = 'usd'
START_TIME = "2021-01-03T00:00:00Z"
END_TIME = "2023-09-15T23:59:59Z"
df =pd.read_csv('data.csv')
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


st.subheader("Number of Simulations for Value at Risk (VaR) and Simulated Returns")

# Create the slider for simulations
num_simulations = st.slider("", 1000, 20000, 10000)

# VaR with Monte Carlo Simulations
st.subheader('VaR with Monte Carlo Simulations')
confidence_level = st.slider("Confidence Level for VaR", 0.80, 0.99, 0.95, 0.01)

# Daily returns
daily_returns = df_pivot[selected_assets].pct_change().dropna()

# Aggregate data for VaR calculations
agg_data = daily_returns.mean(axis=1)

# Calculate mean and standard deviation for VaR
mean = agg_data.mean()
std_dev = agg_data.std()

# Run Monte Carlo Simulation for VaR
simulated_returns = np.random.normal(loc=mean, scale=std_dev, size=num_simulations)
sorted_returns = np.sort(simulated_returns)

# Calculate VaR
var_index = int((1 - confidence_level) * num_simulations)
var = sorted_returns[var_index]
initial_value = df_pivot['index_level'].iloc[-1]
var_value = initial_value * var

var_display_string = f"VaR after {num_simulations} simulations at {confidence_level * 100}% confidence level: ${var_value:.2f}"
st.write(var_display_string)

# Simulated returns with Choelsky Decomposition
st.subheader('Simulated returns with Choelsky Decomposition')
cov_matrix = daily_returns.cov()
L = np.linalg.cholesky(cov_matrix)
simulated_returns_chol = []

for _ in range(num_simulations):
    z = norm.rvs(size=len(selected_assets))
    simulated_return = np.dot(L, z)
    simulated_returns_chol.append(simulated_return)

simulated_returns_chol = np.array(simulated_returns_chol)

# Plot the histogram of the simulated returns with Plotly
fig_hist = px.histogram(x=simulated_returns_chol[:, 0], nbins=50, title=f'Histogram of Simulated Returns using Cholesky Decomposition after {num_simulations} simulations')
st.plotly_chart(fig_hist)
# Plotting daily return values of the index and the individual returns
st.subheader('Daily Returns')
fig = go.Figure()

for asset in selected_assets:
    fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns[asset], mode='lines', name=asset))

fig.update_layout(
    title='Daily Returns of Selected Assets',
    xaxis_title='Date',
    yaxis_title='Daily Return',
    template="plotly_white"
)
st.plotly_chart(fig)

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

# Create a Plotly figure for predictions
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

