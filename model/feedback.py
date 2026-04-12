"""Interactive feedback loop for the CHL points prediction model.

After predict.py produces a ranked list, call review_predictions() to walk
through the top-N candidates and record your own point judgement where the
model is wrong. Feedback is stored in data/feedback.json and merged into the
labeled set on the next training run.

Feedback format (data/feedback.json):
    {
        "Card Name": {
            "points": 2,
            "predicted": 5,
            "note": "optional free-text note"
        },
        ...
    }
"""

import json
import logging
import os
import pickle
from glob import glob

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = "data"
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")
VALID_POINTS = {0, 1, 2, 3, 5, 7, 8}

_ORACLE_INDEX: pd.DataFrame | None = None


def _oracle_index() -> pd.DataFrame:
    """Loads the latest cards_DATESTAMP.pkl and returns it indexed by name."""
    global _ORACLE_INDEX
    if _ORACLE_INDEX is not None:
        return _ORACLE_INDEX
    candidates = sorted(glob(os.path.join(DATA_DIR, "cards_*.pkl")))
    if not candidates:
        raise FileNotFoundError("No cards_*.pkl found in data/. Run scryfall oracle fetch first.")
    with open(candidates[-1], "rb") as f:
        df = pickle.load(f)
    _ORACLE_INDEX = df.set_index("name")
    return _ORACLE_INDEX


def load_feedback() -> dict:
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_feedback(feedback: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, sort_keys=True)
    logger.info(f"Feedback saved to {FEEDBACK_PATH} ({len(feedback)} entries)")


def review_predictions(
    results: pd.DataFrame,
    top_n: int = 30,
    min_discrepancy: float = 1.5,
) -> dict:
    """Interactively review top-N predictions and record corrections.

    Shows each candidate card with its oracle details for context. Cards where
    your correction differs from the model prediction by >= min_discrepancy are
    flagged as high-value feedback.

    Args:
        results:          Output dataframe from predict.main().
        card_df:          Per-card dataframe (from pipeline) for oracle lookups.
        top_n:            How many candidates to review.
        min_discrepancy:  Difference threshold for flagging a correction as notable.

    Returns:
        The updated feedback dict (also written to disk).
    """
    feedback = load_feedback()
    candidates = results.head(top_n)
    oracle = _oracle_index()

    print(f"\n{'='*65}")
    print(f"  Reviewing top {top_n} predictions  (feedback: {FEEDBACK_PATH})")
    print(f"  Enter a point value to record feedback, or press Enter to skip.")
    print(f"  Valid values: {sorted(VALID_POINTS)}")
    print(f"{'='*65}\n")

    for _, row in candidates.iterrows():
        name = row["card_name"]
        pointed_prob = row["pointed_prob"]
        pred = row["est_points"]
        raw = row.get("_reg_raw", pred)

        # Oracle context
        if name in oracle.index:
            card = oracle.loc[name]
            mana   = card.get("mana_cost") or ""
            tl     = card.get("type_line") or ""
            pw     = card.get("power")
            tg     = card.get("toughness")
            pt_str = f"  {pw}/{tg}" if pd.notna(pw) and pd.notna(tg) else ""
            oracle_text = card.get("oracle_text") or ""
        else:
            mana = tl = oracle_text = ""
            pt_str = ""

        existing = feedback.get(name, {})
        existing_str = f"  [previously labelled: {existing['points']}]" if existing else ""

        print(f"  {'─'*61}")
        print(f"  {name}{existing_str}")
        print(f"  {mana}  {tl}{pt_str}")
        if oracle_text:
            words, line = [], ""
            for word in oracle_text.split():
                if len(line) + len(word) + 1 > 61:
                    print(f"    {line}")
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                print(f"    {line}")
        print(f"  suspect={pointed_prob:.1f}%  est={pred}pts (raw={raw:.2f})  apps={int(row['appearances'])}  top4={row['top4_rate']:.0%}")

        while True:
            raw_input = input("    Your points (Enter=skip, q=quit): ").strip().lower()
            if raw_input == "q":
                save_feedback(feedback)
                print("\nExiting review early.")
                return feedback
            if raw_input == "":
                break
            try:
                value = int(raw_input)
                if value not in VALID_POINTS:
                    print(f"    Invalid value. Choose from {sorted(VALID_POINTS)}.")
                    continue
                entry: dict = {"points": value, "est_points": int(pred), "pointed_prob": round(float(pointed_prob), 1)}
                discrepancy = abs(value - pred)
                if discrepancy >= min_discrepancy:
                    print(f"    ** Notable discrepancy: your={value} vs est={pred} (diff={discrepancy}) — recorded.")
                else:
                    print(f"    Recorded.")
                feedback[name] = entry
                break
            except ValueError:
                print("    Please enter a whole number.")

    save_feedback(feedback)
    print(f"\nReview complete. {len(feedback)} total feedback entries.\n")
    return feedback


def apply_feedback_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Merges feedback labels into the per-card dataframe.

    Cards in feedback.json that are currently unlabeled (points=NaN) will have
    their points column set to the feedback value, making them available as
    training examples on the next run.

    Cards already in the official labeled set are NOT overwritten — feedback
    only extends the training set, it does not replace ground truth.

    Args:
        df: Per-card aggregated dataframe from pipeline.aggregate_per_card().

    Returns:
        A copy of df with feedback labels applied.
    """
    feedback = load_feedback()
    if not feedback:
        return df

    df = df.copy()
    df = df.set_index("card_name")

    added = 0
    for card_name, entry in feedback.items():
        if card_name in df.index and pd.isna(df.at[card_name, "points"]):
            df.at[card_name, "points"] = float(entry["points"])
            added += 1

    df = df.reset_index()
    logger.info(f"Applied feedback: {added} cards added to labeled set (total feedback entries: {len(feedback)})")
    return df
