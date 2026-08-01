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

### Streamlit companion app

`streamlit_app.py` is a small web app for collecting community feedback: a
"Top Suspects" tab for rating the model's highest-probability unlabeled
cards, and a "Score a Card" tab for rating any card by name (falls back to a
live Scryfall lookup for cards with no CHL tournament history).

Run locally:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Deploying (Streamlit Community Cloud):**
1. Push to GitHub, then deploy from [share.streamlit.io](https://share.streamlit.io) pointing at this repo, main file `streamlit_app.py`.
2. In the app's Settings → Secrets, set:
   ```toml
   GITHUB_TOKEN = "ghp_..."   # fine-grained PAT, Issues: write, scoped to this repo
   GITHUB_REPO  = "BIG-TRUCK/points_bot"
   ```
   Without these, both tabs still work but feedback submission is disabled.

**Data:** the app reads from `app_data/` (small, committed snapshot), not
`data/` (gitignored, and the Scryfall cache alone is >100MB — over GitHub's
file size limit). After every `model/train.py` run, refresh the bundle and
commit it:
```
python -m model.predict
python -m scripts.export_app_data
git add app_data/
```

**Feedback loop:** visitor ratings land as GitHub Issues labeled `feedback`
on this repo (not `data/feedback.json` directly, since the hosted app's
filesystem doesn't persist across restarts). Reconcile them into
`data/feedback.json` manually, or write a sync script, before the next
`model/train.py` run picks them up.

### GitHub Pages model report

`docs/index.html` is a static, no-JS summary of the latest trained model:
dataset size, which model candidates were evaluated per stage and which won,
AUPRC/F1 for the classifier (MAE for the regressor), SHAP feature importance
for every candidate — not just the winner — and the top-15 suspect cards.

**Enable once:** repo Settings → Pages → "Deploy from a branch" → branch
`main`, folder `/docs`.

**Regenerate after every retrain:**
```
python -m model.train
python -m model.predict
python -m scripts.build_report
git add docs/
```