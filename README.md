# Cryptocurrency Index Analysis

Check out the deployed streamlit app [here](cryptoindexapp-bspok5sudvepnmqe4gvam7.streamlit.app/)

## Analytics Tools:

- **Individual Asset Performance**: A dashboard where users can choose to compare the individual cryptocurrency performance to the index performance.
- **Value at Risk (VaR)**: Offers analysis with Monte Carlo simulations to estimate VaR at user-defined confidence levels.
- **Simulated Portfolio Returns**: Provides simulated returns using Cholesky Decomposition.
- **Prediciton of Index value**: Uses a vanilla XGBoost model to predict future index values.
- **Daily Return Values**: Graphical representation of the daily return values of the index and individual assets.
- **Historical Volatility**: Displays the historical volatility of all individual assets based on a user-defined rolling window.
  
## Features

- **Index Construction**:
    - **Selected Assets**: The cryptocurrencies chosen for the index are Bitcoin Cash (bch), Ethereum (eth), Ripple (xrp), Litecoin (ltc), and Polkadot (dot). These were the top 5 non-Bitcoin and non-stablecoin cryptocurrencies by market cap as of January 3, 2021. 
    - **Data Processing & Aggregation**: Using the prices of the selected cryptocurrencies, an average is computed. Based on this average, an index level is  initialized to 1000. A divisor is utilized to ensure the index's continuity, allowing it to remain consistent and unaffected by structural changes like cryptocurrency splits, mergers, or any other corporate actions that can alter the number of available coins or their price. The divisor is then calculated using the formula:
  $$\text{Divisor} = \frac{\text{Initial Average Price}}{\text{Initial Index Level}}$$

      The average price and index level for the cryptocurrencies are then computed using this divisor, resulting in the aggregated index.

## Setup

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
