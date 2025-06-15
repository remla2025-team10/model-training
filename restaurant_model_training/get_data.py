"""
This module loads the raw datasets for the restaurant sentiment predcition.
"""
from pathlib import Path
import requests
import config

# Create directory
Path(config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Data URL
URL_HISTORIC = (
    "https://raw.githubusercontent.com/proksch/restaurant-sentiment/"
    "refs/heads/main/a1_RestaurantReviews_HistoricDump.tsv"
)
URL_FRESH = (
    "https://raw.githubusercontent.com/proksch/restaurant-sentiment/"
    "refs/heads/main/a2_RestaurantReviews_FreshDump.tsv"
)

# Download and save
response = requests.get(URL_HISTORIC, timeout=10)
response.raise_for_status()  # Raise error for bad status

with open(config.DEFAULT_RAW_DATA_PATH, "wb") as f:
    f.write(response.content)

# Download and save
response = requests.get(URL_FRESH, timeout=10)
response.raise_for_status()  # Raise error for bad status

with open(config.DEFAULT_RAW_DATA_FRESH_PATH, "wb") as f:
    f.write(response.content)

print(f"Downloaded historic data to {config.DEFAULT_RAW_DATA_PATH}")
print(f"Downloaded fresh data to {config.DEFAULT_RAW_DATA_FRESH_PATH}")
