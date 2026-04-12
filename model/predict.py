"""Run inference on unlabeled cards to produce a ranked candidate list.

Usage:
    python -m model.predict                     # loads latest per_card pkl + saved model
    python -m model.predict data/my_data.pkl    # explicit data path

Outputs:
    data/chl_predictions.csv  — all unlabeled cards ranked by predicted points desc
"""

import logging
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd

from model.features import build_feature_matrix
from model.feedback import review_predictions

logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "chl_model.pkl")
OUTPUT_PATH = os.path.join(DATA_DIR, "chl_predictions.csv")

VALID_POINTS = {0, 1, 2, 3, 5, 7, 8}


def _snap_to_valid(value: float) -> int:
    return min(VALID_POINTS, key=lambda v: abs(v - value))


def _load_per_card_df(path: str | None = None) -> pd.DataFrame:
    if path:
        with open(path, "rb") as f:
            return pickle.load(f)
    candidates = sorted(glob(os.path.join(DATA_DIR, "chl_dataset_per_card_*.pkl")))
    if not candidates:
        raise FileNotFoundError("No per_card dataset found. Run pipeline.py first.")
    latest = candidates[-1]
    logger.info(f"Loading {latest}")
    with open(latest, "rb") as f:
        return pickle.load(f)


def main(data_path: str | None = None, review: bool = True, top_n: int = 30) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)

    df = _load_per_card_df(data_path)
    unlabeled = df[df["points"].isna()].copy()
    logger.info(f"Unlabeled cards: {len(unlabeled)}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    X = build_feature_matrix(unlabeled)

    # Align columns in case feature set has drifted
    missing = set(feature_cols) - set(X.columns)
    for col in missing:
        X[col] = 0.0
    X = X[feature_cols]

    raw_preds = model.predict(X)
    snapped = [_snap_to_valid(p) for p in raw_preds]

    results = pd.DataFrame({
        "card_name": unlabeled["card_name"].values,
        "predicted_points_raw": raw_preds,
        "predicted_points": snapped,
        "appearances": unlabeled["appearances"].values,
        "avg_placement": unlabeled["avg_placement"].values,
        "top4_rate": unlabeled["top4_rate"].values,
        "avg_level": unlabeled["avg_level"].values,
    }).sort_values("predicted_points_raw", ascending=False).reset_index(drop=True)

    results.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Predictions saved to {OUTPUT_PATH}")

    print(f"\n--- Top {top_n} candidate cards for points ---")
    print(results.head(top_n).to_string(index=False))

    if review:
        review_predictions(results, top_n=top_n)

    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
