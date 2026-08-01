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
import requests
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


def _score(X: pd.DataFrame, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    """Applies the saved classifier + regressor (with their scalers, if any)
    to an already-built, already-column-aligned feature matrix.

    Returns:
        (pointed_probs, raw_point_estimates) — same length as X.
    """
    clf = artifact["classifier"]
    scaler: StandardScaler | None = artifact.get("classifier_scaler")
    reg = artifact["regressor"]
    reg_scaler: StandardScaler | None = artifact.get("regressor_scaler")

    if scaler is not None:
        pointed_probs = clf.predict_proba(scaler.transform(X))[:, 1]
    else:
        pointed_probs = clf.predict_proba(X)[:, 1]

    if reg_scaler is not None:
        raw_reg = reg.predict(reg_scaler.transform(X))
    else:
        raw_reg = reg.predict(X)

    return pointed_probs, raw_reg


def _align_columns(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Reindexes X to the training-time feature columns, zero-filling any that
    are missing (e.g. a TF-IDF term never seen in this batch)."""
    for col in set(feature_cols) - set(X.columns):
        X[col] = 0.0
    return X[feature_cols]


def main(data_path: str | None = None, review: bool = True, top_n: int = 30) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)

    df = _load_per_card_df(data_path)
    unlabeled = df[df["points"].isna()].copy()
    logger.info(f"Unlabeled cards: {len(unlabeled)}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    feature_cols = artifact["feature_columns"]
    preprocessors = artifact.get("preprocessors")

    X, _ = build_feature_matrix(unlabeled, preprocessors=preprocessors)
    X = _align_columns(X, feature_cols)

    pointed_probs, raw_reg = _score(X, artifact)
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


SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named"


def _fetch_scryfall_card(card_name: str) -> dict | None:
    """Live single-card lookup via Scryfall's fuzzy-named endpoint.

    Used only as a fallback for a card not present in the local card pool —
    far lighter than pulling Scryfall's full bulk dataset for one card.
    """
    try:
        resp = requests.get(
            SCRYFALL_NAMED_URL,
            params={"fuzzy": card_name},
            headers={"User-Agent": "BIGTRUCKPointsBot/1.0"},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(f"Scryfall lookup failed for '{card_name}': {e}")
        return None
    if resp.status_code != 200:
        return None
    return resp.json()


def _card_row_from_scryfall(card: dict) -> dict:
    """Maps a raw Scryfall card object onto the per-card schema build_feature_matrix
    expects. Double-faced cards keep oracle_text/mana_cost/power/toughness only
    on their `card_faces`, not the top-level object, so fall back to face 0."""
    face = card.get("card_faces", [{}])[0] if card.get("card_faces") else {}
    return {
        "card_name":       card.get("name"),
        "points":          np.nan,
        "appearances":     0,
        "avg_placement":   np.nan,
        "top4_rate":       np.nan,
        "avg_level":       np.nan,
        "high_level_rate": np.nan,
        "mana_cost":       card.get("mana_cost", face.get("mana_cost")),
        "cmc":             card.get("cmc"),
        "type_line":       card.get("type_line", face.get("type_line")),
        "power":           card.get("power", face.get("power")),
        "toughness":       card.get("toughness", face.get("toughness")),
        "color_identity":  card.get("color_identity", []),
        "keywords":        card.get("keywords", []),
        "game_changer":    card.get("game_changer", False),
        "rarity":          card.get("rarity"),
        "edhrec_rank":     card.get("edhrec_rank"),
        "produced_mana":   card.get("produced_mana", []),
        "oracle_text":     card.get("oracle_text", face.get("oracle_text", "")),
        "card_tags":       [],
    }


def score_card(
    card_name: str,
    model_path: str = MODEL_PATH,
    card_pool: pd.DataFrame | None = None,
) -> dict | None:
    """Scores a single card by name — whether or not it has CHL tournament history.

    If `card_pool` (the per-card training dataframe) contains the card, its
    real tournament stats, tags, and relationships are used. Otherwise the
    card is looked up live via Scryfall and scored from its intrinsic
    properties alone (oracle text, cost, type, ...) — a materially weaker
    signal, since tournament performance is the model's strongest predictor,
    so the result is flagged with `has_tournament_history: False`.

    Args:
        card_name: Card name. Exact match against `card_pool`; fuzzy-matched
            by Scryfall for the live-lookup fallback.
        model_path: Path to the saved chl_model.pkl artifact.
        card_pool: Per-card dataframe to check for existing tournament
            history before falling back to a live Scryfall lookup. If None,
            every card is treated as never-played.

    Returns:
        dict with card_name, pointed_prob, est_points, has_tournament_history,
        already_pointed, and oracle summary fields for display — or None if
        the card can't be found anywhere (including on Scryfall).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run model/train.py first.")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    row = None
    has_history = False
    if card_pool is not None:
        match = card_pool[card_pool["card_name"].str.lower() == card_name.lower()]
        if not match.empty:
            row = match.iloc[0]
            has_history = (row.get("appearances") or 0) > 0

    if row is not None and pd.notna(row.get("points")) and row.get("points", 0) > 0:
        return {
            "card_name":              row["card_name"],
            "already_pointed":        True,
            "points":                 int(row["points"]),
            "has_tournament_history": has_history,
            "mana_cost":              row.get("mana_cost"),
            "type_line":              row.get("type_line"),
            "oracle_text":            row.get("oracle_text"),
            "rarity":                 row.get("rarity"),
        }

    if row is not None:
        row_df = card_pool.loc[[row.name]]
        display_name = row["card_name"]
    else:
        card = _fetch_scryfall_card(card_name)
        if card is None:
            return None
        record = _card_row_from_scryfall(card)
        row_df = pd.DataFrame([record])
        display_name = record["card_name"]

    feature_cols = artifact["feature_columns"]
    X, _ = build_feature_matrix(row_df, preprocessors=artifact.get("preprocessors"))
    X = _align_columns(X, feature_cols)

    pointed_probs, raw_reg = _score(X, artifact)

    return {
        "card_name":              display_name,
        "already_pointed":        False,
        "pointed_prob":           round(float(pointed_probs[0]) * 100, 1),
        "est_points":             _snap_to_valid(raw_reg[0]),
        "raw_points":             float(raw_reg[0]),
        "has_tournament_history": has_history,
        "mana_cost":              row_df.iloc[0].get("mana_cost"),
        "type_line":              row_df.iloc[0].get("type_line"),
        "oracle_text":            row_df.iloc[0].get("oracle_text"),
        "rarity":                 row_df.iloc[0].get("rarity"),
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].endswith(".pkl"):
        lookup(sys.argv[1])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
