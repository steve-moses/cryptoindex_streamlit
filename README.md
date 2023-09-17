# Cryptocurrency Index Analysis

## Description
This is a Streamlit-based application that constructs a comprehensive cryptocurrency index using a selection of top-performing cryptocurrencies. The application offers insights into the performance of the selected cryptocurrencies, the methodology behind the index construction, and provides additional analytics tools.

## Features

- **Index Construction**:
    - **Selected Assets**: The cryptocurrencies chosen for the index are Bitcoin Cash (bch), Ethereum (eth), Ripple (xrp), Litecoin (ltc), and Polkadot (dot). These were the top 5 non-Bitcoin and non-stablecoin cryptocurrencies by market cap as on January 3, 2021. 
    - **Data Processing & Aggregation**: Using the prices of the selected cryptocurrencies, an average is computed. Based on this average, an index level is  initialized to 1000. A divisor is utilized to ensure the index's continuity, allowing it to remain consistent and unaffected by structural changes like cryptocurrency splits, mergers, or any other corporate actions that can alter the number of available coins or their price. The divisor is then calculated using the formula:
  $$\text{Divisor} = \frac{\text{Initial Average Price}}{\text{Initial Index Level}}$$

      The average price and index level for the cryptocurrencies are then computed using this divisor, resulting in the aggregated index.

- **Interactive Dashboard**:
    - A multi-select widget allows users to choose which cryptocurrencies they wish to analyze.
    - The app visualizes the values of the selected cryptocurrencies against the index. 

## Analytics Tools:
This section highlights the various analytical tools embedded within the application:

- **Individual Asset Performance**: A dashboard where users can choose to compare the individual cryptocurrency performance to the index performance.
- **Value at Risk (VaR)**: Offers analysis with Monte Carlo simulations to estimate VaR at user-defined confidence levels.
- **Simulated Portfolio Returns**: Provides simulated returns using Cholesky Decomposition.
- **Prediciton of Index value**: Uses a vanilla XGBoost model to predict future index values.
- **Daily Return Values**: Graphical representation of the daily return values of the index and individual assets.
- **Historical Volatility**: Displays the historical volatility of all individual assets based on a user-defined rolling window.


## Setup

### Prerequisites
- Docker

### Installation
1. Clone the repository.
2. Navigate to the project directory.

## Usage

### Docker (Recommended):
1. Build the Docker image: `docker build -t crypto-streamlit-app .`
2. Run the Docker container: `docker run -p 8502:8501 crypto-streamlit-app`
3. Access the dashboard by visiting `http://localhost:8501` in your web browser.

### Streamlit (Standalone):
Note: Running the application outside Docker is not recommended for consistent results. If you still wish to run it standalone:
1. Install the requirements using: `pip install -r requirements.txt`
2. Run the Streamlit app with: `streamlit run crypto_index_app.py`
3. Navigate to the provided localhost URL in your browser.


## Acknowledgements
- Data sourced from the Kaiko API.
- The application is built using Streamlit, Python, and Docker.
