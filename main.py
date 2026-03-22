import logging_config
logging_config.setup_logging()

import api.scryfall as scryfall
import api.edhrec as edhrec
from typing import Optional
import pandas as pd

cards: Optional[pd.DataFrame] = scryfall.get_cards_df()

if cards is not None:
    print(cards.head())
else:
    print("Failed to get cards data.")