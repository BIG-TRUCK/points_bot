import logging
from pyedhrec import EDHRec

logger = logging.getLogger(__name__)

edhr = EDHRec()

def get_card_details(card_name: str) -> dict:
    """Fetches card details from EDHRec for a given card name.

    Args:
        card_name (str): The name of the card to fetch details for.
    Returns:
        dict: A dictionary containing card details such as tags, relationships, and more.
    """
    try:
        card_details = edhr.get_card_details(card_name)
        return card_details
    except Exception as e:
        logger.error(f"Error fetching details for card '{card_name}': {e}")
        return {}
    
def get_salt_and_tags(card_name: str) -> dict:
    """Fetches the salt and tags for a given card from EDHRec.

    Args:
        card_name (str): The name of the card to fetch salt and tags for.
    Returns:
        dict: A dictionary containing the salt and tags for the card.
    """
    details = get_card_details(card_name)
    return {
        'salt': details.get('salt', ''),
        'tags': details.get('tags', [])
    }