import logging
import requests
import pandas as pd
import pickle
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Union


logger: logging.Logger = logging.getLogger(__name__)


def fetch_latest_bulk_data_uuid() -> Optional[str]:
    """Fetches the latest bulk data UUID for oracle cards from Scryfall API.

    Returns:
        Optional[str]: The UUID of the latest oracle cards bulk data, or None if not found.
    """
    response: requests.Response = requests.get('https://api.scryfall.com/bulk-data', headers={'User-Agent': 'BIGTRUCKPointsBot/1.0'})
    data = response.json()
    for item in data['data']:
        if item['type'] == 'oracle_cards':
            return item['id']
    return None

def fetch_bulk_data(uuid: str) -> List[Dict[str, Any]]:
    """Fetches the bulk data for the given UUID from Scryfall API.

    Args:
        uuid (str): The UUID of the bulk data to fetch.

    Returns:
        List[Dict[str, Any]]: The list of card data dictionaries.
    """
    url: str = f'https://api.scryfall.com/bulk-data/{uuid}'
    response: requests.Response = requests.get(url, headers={'User-Agent': 'BIGTRUCKPointsBot/1.0'})
    data = response.json()
    dl_uri = data['download_uri']
    dl_response: requests.Response = requests.get(dl_uri, headers={'User-Agent': 'BIGTRUCKPointsBot/1.0'})

    return dl_response.json()

def build_dataframe(cards: List[Dict[str, Any]]) -> pd.DataFrame:
    """Builds a pandas DataFrame from the list of card dictionaries.

    Args:
        cards (List[Dict[str, Any]]): The list of card data.

    Returns:
        pd.DataFrame: The DataFrame containing card data.
    """
    df = pd.DataFrame(cards)
    return df

def save_dataframe(df: pd.DataFrame) -> bool:
    """Saves the DataFrame to a pickle file in the data directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    timestamp: str = datetime.now().strftime('%Y%m%d')
    filename: str = f'cards_{timestamp}.pkl'
    
    try:
        os.makedirs('data', exist_ok=True)
        filepath: str = os.path.join('data', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.error("Error saving DataFrame", exc_info=True, extra={'error': e})
        return False

    return True

def check_for_updates() -> Optional[str]:
    """Checks if the cards data for today already exists.

    Returns:
        Optional[str]: The filename if it exists, None otherwise.
    """
    timestamp: str = datetime.now().strftime('%Y%m%d')
    filename: str = f'cards_{timestamp}.pkl'
    filepath: str = os.path.join('data', filename)
    return filename if os.path.exists(filepath) else None

def get_cards_df() -> pd.DataFrame:
    """Retrieves the cards DataFrame, fetching and saving if necessary.

    Returns:
        pd.DataFrame: The DataFrame of cards.
    """
    filename: Optional[str] = check_for_updates()
    if filename:
        logger.info("Data is already up to date.")
        return pickle.load(open(os.path.join('data', filename), 'rb'))

    logger.info("Cards data is old, fetching latest bulk data UUID...")
    uuid: Optional[str] = fetch_latest_bulk_data_uuid()
    if not uuid:
        logger.error("Could not find bulk data UUID.")
        raise ValueError("Could not find bulk data UUID.")

    cards: List[Dict[str, Any]] = fetch_bulk_data(uuid)
    df: pd.DataFrame = build_dataframe(cards)
    if save_dataframe(df):
        logger.info("Data saved successfully.")
    else:
        logger.error("Failed to save data.")
    return df

if __name__ == "__main__":
    get_cards_df()