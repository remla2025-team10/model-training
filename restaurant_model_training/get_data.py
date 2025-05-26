import requests
from pathlib import Path
import config


# Create directory
Path(config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Data URL
url_historic = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/refs/heads/main/a1_RestaurantReviews_HistoricDump.tsv"
url_fresh = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/refs/heads/main/a2_RestaurantReviews_FreshDump.tsv"

# Download and save
response = requests.get(url_historic)
response.raise_for_status()  # Raise error for bad status

with open(config.DEFAULT_RAW_DATA_PATH, "wb") as f:
    f.write(response.content)

# Download and save
response = requests.get(url_fresh)
response.raise_for_status()  # Raise error for bad status

with open(config.DEFAULT_RAW_DATA_FRESH_PATH, "wb") as f:
    f.write(response.content)

print(f"Downloaded historic data to {config.DEFAULT_RAW_DATA_PATH}")
print(f"Downloaded fresh data to {config.DEFAULT_RAW_DATA_FRESH_PATH}")
