"""Exports a small snapshot of data/ into app_data/ for the Streamlit app.

data/ is entirely gitignored (it holds the ~180MB Scryfall bulk cache, which
is over GitHub's 100MB file limit anyway), so the deployed app can't read
from it directly. app_data/ holds just the three small files the app needs,
and — unlike data/ — is meant to be committed.

Re-run this after every model/train.py run so the deployed app picks up the
latest model and predictions:

    python -m scripts.export_app_data
"""

import logging
import os
import pickle
import shutil
from glob import glob

logger = logging.getLogger(__name__)

DATA_DIR = "data"
APP_DATA_DIR = "app_data"


def _latest(pattern: str) -> str:
    candidates = sorted(glob(os.path.join(DATA_DIR, pattern)))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {DATA_DIR}/")
    return candidates[-1]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    os.makedirs(APP_DATA_DIR, exist_ok=True)

    model_src = os.path.join(DATA_DIR, "chl_model.pkl")
    predictions_src = os.path.join(DATA_DIR, "chl_predictions.csv")
    card_pool_src = _latest("chl_dataset_per_card_*.pkl")

    for src, dst_name in [
        (model_src, "chl_model.pkl"),
        (predictions_src, "chl_predictions.csv"),
        (card_pool_src, "chl_card_pool.pkl"),
    ]:
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} not found — run model/train.py and model/predict.py first.")
        dst = os.path.join(APP_DATA_DIR, dst_name)
        shutil.copyfile(src, dst)
        logger.info(f"{src} -> {dst} ({os.path.getsize(dst) / 1024:.0f} KB)")

    with open(os.path.join(APP_DATA_DIR, "chl_model.pkl"), "rb") as f:
        artifact = pickle.load(f)
    logger.info(f"Bundled model has {len(artifact['feature_columns'])} feature columns.")


if __name__ == "__main__":
    main()
