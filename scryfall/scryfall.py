import json
import logging
import requests
import pandas as pd
import pickle
from datetime import datetime
import os


# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_dir = os.path.join(os.path.expanduser('~'), 'points_bot')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'logs_{datetime.now().strftime("%Y%m%d")}.jsonl')

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_entry)

handler = logging.FileHandler(log_file)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


def fetch_latest_bulk_data_uuid():
    response = requests.get('https://api.scryfall.com/bulk-data')
    data = response.json()
    for item in data['data']:
        if item['type'] == 'oracle_cards':
            return item['id']
    return None

def fetch_bulk_data(uuid):
    url = f'https://api.scryfall.com/bulk-data/{uuid}'
    response = requests.get(url)
    data = response.json()
    dl_uri = data['download_uri']
    dl_response = requests.get(dl_uri)

    return dl_response.json()

def build_dataframe(cards):
    df = pd.DataFrame(cards)
    return df

def save_dataframe(df, filename):
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f'cards_{timestamp}.pkl'
    
    try:
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.error(f"Error saving DataFrame: {e}")
        return False

    return True

def check_for_updates():
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f'cards_{timestamp}.pkl'
    filepath = os.path.join('data', filename)
    return filename if os.path.exists(filepath) else False

def get_cards_df():
    filename = check_for_updates()
    if filename:
        logger.info("Data is already up to date.")
        return pickle.load(open(os.path.join('data', filename), 'rb'))

    uuid = fetch_latest_bulk_data_uuid()
    if not uuid:
        logger.error("Could not find bulk data UUID.")
        return False

    cards = fetch_bulk_data(uuid)
    df = build_dataframe(cards)
    if save_dataframe(df, f'cards_{datetime.now().strftime("%Y%m%d")}.pkl'):
        logger.info("Data saved successfully.")
    else:
        logger.error("Failed to save data.")
    return df

if __name__ == "__main__":
    get_cards_df()