import logging_config
logging_config.setup_logging()

"""
The goal of this file is to prepare a data set for use in training an ML model. At a high level the process is:

1. Get the current points list. 
    a. Do this using the moxfield api and the decklist ID "GwH6Gikx-UOzLMicS2iqFA". 
    a. This will generate a txt file in the format of `n <card_name>` (e.g., `4 Lightning Bolt`). n is the point value of the card.
    a. Use this txt file to generate a dictionary mapping card names to point values. Store it in the data directory as `chl_points_dict.json`.
1. Get a list of all occurrences of all cards.
    a. Do this using the mtgtop8 api. 
    a. This will populate two files per decklist in the data/CHL directory: `decklist_name.txt` and `decklist_name.json`. 
    a. The txt file will contain the card names and the json file will contain deck metadata.
    a. The txt file is formatted as `n <card_name>` (e.g., `4 Lightning Bolt`). n is the number of copies of the card in the decklist, but will always be 1 in this format. Remove the leading number and space to get the card name.
    a. Disregard any cards with the name "Plains", "Island", "Swamp", "Mountain", or "Forest" since these are not relevant to the points system.
    a. Also disregard the snow-covered variants i.e. "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp", "Snow-Covered Mountain", and "Snow-Covered Forest".
    a. We can now populate a dataframe with the intial structre of:
        | card_name | points | level | placement | date |
        | --- | --- | --- | --- | --- |
        where:
        - card_name is the name of the card
        - points is the point value of the card from the points list
        - level is the level of the tournament (get this from the json metadata file)
        - placement is the position of the deck in the tournament (get this from the json metadata file)
        - date is the date of the tournament (get this from the json metadata file)
        There will be one row per card occurrence, so if a decklist has 4 copies of Lightning Bolt, there will be 4 rows with card_name "Lightning Bolt" and the same values for the other columns.
1. Once this dataframe is populated, we extend it with additional features using the Scryfall API.

"""