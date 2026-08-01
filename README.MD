## Points Bot

This is a project for attempting to identify point-worthy cards in the 7-point Highlander MtG format. 

### Process
1. Top-8 decklists are scraped from 7ph.com.au tournament result reports.
    - These reports are painfully non-standardised, so the usual joys of web-scraping abound here.
2. Individual cardnames are extracted from decklists, and added to a master table (alongside the placing of the deck, sideboard and mainboard copy counts, and the quarter that the result was posted.)
3. Additional card information is extracted from scryfall's database, including card tags, CMC, spell type, color identity, and oracle text.
    - Oracle text is vectorised with TF-IDF (English stopwords pruned) into `oracle_tfidf_*` features for the supervised model.
4. Card information is further extended with point value as at the time of play.
5. Lands are removed from the data set completely, with exception of those with special abilities (e.g. Otawara, Karakas, Boseiju, etc.).
6. Pointed cards are upsampled with perturbation applied to sideboard/maindeck counts and color identity.
7. Model is trained.