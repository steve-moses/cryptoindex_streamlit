# Cryptocurrency Portfolio Index & Analytics Suite

The Cryptocurrency Portfolio Index & Analytics Suite is designed to be customizable. Every analytical tool provided, from the technical indicators to the Value at Risk (VaR) simulations, offers user defined inputs. This ensures that users have the flexibility to adjust settings based on their preferences or specific analysis requirements. 


## Analytics Tools:

- **Individual Asset Performance**: A dashboard where users can choose to compare component asset performance to the index performance.
- **Daily Return Values**: A Graphical representation of the daily return values of the component assets.
- **Simulated Portfolio Returns**: Provides simulated returns using Monte Carlo simulations and Cholesky Decomposition.
- **Value at Risk (VaR)**: Estimate VaR at user-defined confidence levels.
- **Technical Analysis**:
    - **Historical Volatility**: Displays the historical volatility, providing insights on the stability and potential risk of assets.
    - **Simple Moving Average (SMA)**: Used to smooth out short-term fluctuations and highlight longer-term trends or cycles.
    - **Relative Strength Index (RSI)**: Measures the speed and change of price movements, often used to identify overbought or oversold conditions.
    - **Bollinger Bands**: Represents the volatility around a moving average, indicating periods of potential overbought or oversold conditions in the market.
    - **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that reveals changes in the strength, direction, momentum, and duration of a trend.

  
## Features

- **Index Construction**:
    - **Selected Assets**: The cryptocurrencies chosen for the index are Bitcoin Cash (bch), Ethereum (eth), Ripple (xrp), Litecoin (ltc), and Polkadot (dot). These were the top 5 non-Bitcoin and non-stablecoin cryptocurrencies by market cap as of January 3, 2021. 
    - **Data Processing & Aggregation**: Using the prices of the selected cryptocurrencies, an average is computed. Based on this average, an index level is  initialized to 1000. A divisor is utilized to ensure the index's continuity, allowing it to remain consistent and unaffected by structural changes like cryptocurrency splits, mergers, or any other corporate actions that can alter the number of available coins or their price. The divisor is then calculated using the formula:
  $$\text{Divisor} = \frac{\text{Initial Average Price}}{\text{Initial Index Level}}$$

      The average price and index level for the cryptocurrencies are then computed using this divisor, resulting in the aggregated index.

## Setup

### Requirements: 
1. An API key from Kaiko with access to Aggregated VWAP prices is required to fetch pricing data for the index. 

### Installation
1. Clone the repository.
2. Navigate to the project directory.

## Usage
### Streamlit (Local):
1. Install the requirements using: `pip install -r requirements.txt`
2. Run the Streamlit app with: `streamlit run crypto_index_app.py`
3. Navigate to the provided localhost URL in your browser.


## Acknowledgements
- Data sourced from the Kaiko API.
