"""Run two-stage inference to produce a ranked suspect list.

Stage 1 — Classifier: assigns a probability that each unlabeled card should
           be pointed at all. This drives the ranking.
Stage 2 — Regressor:  estimates how many points, shown as context only.

Usage:
    python -m model.predict                     # loads latest per_card pkl + saved model
    python -m model.predict data/my_data.pkl    # explicit data path

Outputs:
    data/chl_predictions.csv  — all unlabeled cards ranked by pointed_prob desc
"""

import logging
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    clf = artifact["classifier"]
    scaler: StandardScaler | None = artifact.get("classifier_scaler")
    reg = artifact["regressor"]
    reg_scaler: StandardScaler | None = artifact.get("regressor_scaler")
    feature_cols = artifact["feature_columns"]
    tfidf_vectorizer = artifact.get("tfidf_vectorizer")

    X, _ = build_feature_matrix(unlabeled, tfidf_vectorizer=tfidf_vectorizer)

    # Align columns in case feature set has drifted
    for col in set(feature_cols) - set(X.columns):
        X[col] = 0.0
    X = X[feature_cols]

    # Stage 1 — classifier probabilities
    if scaler is not None:
        X_clf = scaler.transform(X)
        pointed_probs = clf.predict_proba(X_clf)[:, 1]
    else:
        pointed_probs = clf.predict_proba(X)[:, 1]

    # Stage 2 — point estimate (context only, not used for ranking)
    if reg_scaler is not None:
        raw_reg = reg.predict(reg_scaler.transform(X))
    else:
        raw_reg = reg.predict(X)
    snapped = [_snap_to_valid(p) for p in raw_reg]

    results = pd.DataFrame({
        "card_name":       unlabeled["card_name"].values,
        "pointed_prob":    np.round(pointed_probs * 100, 1),
        "est_points":      snapped,
        "appearances":     unlabeled["appearances"].values,
        "avg_placement":   unlabeled["avg_placement"].values,
        "top4_rate":       unlabeled["top4_rate"].values,
        "avg_level":       unlabeled["avg_level"].values,
        "_reg_raw":        raw_reg,
    }).sort_values("pointed_prob", ascending=False).reset_index(drop=True)

    results.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Predictions saved to {OUTPUT_PATH}")

    print(f"\n--- Top {top_n} suspects ---")
    print(results.drop(columns="_reg_raw").head(top_n).to_string(index=False))

    if review:
        review_predictions(results, top_n=top_n)

    return results


def lookup(card_name: str) -> dict | None:
    """Returns pointed_prob and est_points for a single card by name.

    Loads from the saved predictions CSV if available, otherwise runs full
    inference first. Returns None if the card is not found (e.g. it is already
    in the labeled/pointed set).

    Args:
        card_name: Exact card name as it appears in the dataset.

    Returns:
        dict with keys 'card_name', 'pointed_prob', 'est_points', or None.
    """
    if os.path.exists(OUTPUT_PATH):
        results = pd.read_csv(OUTPUT_PATH)
    else:
        results = main(review=False)

    match = results[results["card_name"].str.lower() == card_name.lower()]
    if match.empty:
        logger.warning(f"'{card_name}' not found in predictions (may already be labeled).")
        return None

    row = match.iloc[0]
    result = {
        "card_name":    row["card_name"],
        "pointed_prob": round(float(row["pointed_prob"]), 1),
        "est_points":   int(row["est_points"]),
    }
    print(f"  {result['card_name']}: {result['pointed_prob']}% likely to be pointed, est. {result['est_points']}pt")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].endswith(".pkl"):
        lookup(sys.argv[1])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
