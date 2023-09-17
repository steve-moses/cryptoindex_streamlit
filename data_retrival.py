# Data is fetched seperately for security purposes of API keys

import pandas as pd
import requests
import os
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()


def fetch_kaiko_data(api_key, base_assets, quote_asset, start_time, end_time):
    base_url = "https://us.market-api.kaiko.io/v2/data/trades.v1/spot_direct_exchange_rate"
    headers = {
        'Accept': 'application/json',
        'X-Api-Key': api_key
    }
    
    data_frames = []

    for base in base_assets:
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

API_KEY = os.getenv("KAIKO_API_KEY")
BASE_ASSETS = ['bch', 'eth', 'xrp', 'ltc', 'dot']
QUOTE_ASSET = 'usd'
START_TIME = "2021-01-03T00:00:00Z"
END_TIME = "2023-09-15T23:59:59Z"

df = fetch_kaiko_data(API_KEY, BASE_ASSETS, QUOTE_ASSET, START_TIME, END_TIME)
df.to_csv('data.csv', index=False)
