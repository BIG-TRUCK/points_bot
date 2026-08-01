import json
import logging
import os
import pickle
import re
from datetime import datetime

import pandas as pd

import api.moxfield as moxfield
import api.mtgtop8 as mtgtop8
import api.scryfall as scryfall

logger = logging.getLogger(__name__)

DATA_DIR = "data"
CHL_DIR = os.path.join(DATA_DIR, "CHL")
POINTS_DECK_ID = "GwH6Gikx-UOzLMicS2iqFA"
POINTS_DICT_PATH = os.path.join(DATA_DIR, "chl_points_dict.json")

BASIC_LANDS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest",
}

ORACLE_FEATURES = [
    "mana_cost", "cmc", "type_line", "power", "toughness",
    "color_identity", "keywords", "game_changer", "rarity",
    "edhrec_rank", "produced_mana", "oracle_text",
]


# ---------------------------------------------------------------------------
# Step 1: Points list
# ---------------------------------------------------------------------------

def build_points_dict() -> dict[str, int]:
    """Downloads the points list from Moxfield and returns a card->points dict.

    Also saves it to data/chl_points_dict.json.
    """
    txt_path = os.path.join(DATA_DIR, "moxfield", f"{POINTS_DECK_ID}.txt")
    moxfield.download_decklist(POINTS_DECK_ID)

    points: dict[str, int] = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s+(.+)$", line)
            if match:
                n, name = int(match.group(1)), match.group(2).strip()
                points[name] = n

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(POINTS_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)

    logger.info(f"Points dict built with {len(points)} cards.")
    return points


def load_points_dict() -> dict[str, int]:
    if os.path.exists(POINTS_DICT_PATH):
        with open(POINTS_DICT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return build_points_dict()


# ---------------------------------------------------------------------------
# Step 2: Tournament decklists → initial dataframe
# ---------------------------------------------------------------------------

def parse_decklist_txt(path: str) -> list[str]:
    """Returns a list of card names from a decklist txt file, excluding basics."""
    cards = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\d+\s+(.+)$", line)
            name = match.group(1).strip() if match else line
            if name not in BASIC_LANDS:
                cards.append(name)
    return cards


def build_occurrences_df(points: dict[str, int]) -> pd.DataFrame:
    """Scrapes all CHL decklists and builds the initial occurrences dataframe."""
    mtgtop8.scrape_all()

    rows = []
    for filename in os.listdir(CHL_DIR):
        if not filename.endswith(".txt"):
            continue
        deck_id = filename[:-4]
        txt_path = os.path.join(CHL_DIR, filename)
        json_path = os.path.join(CHL_DIR, f"{deck_id}.json")

        if not os.path.exists(json_path):
            logger.warning(f"Missing metadata for deck {deck_id}, skipping.")
            continue

        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        level = meta.get("level")
        placement = meta.get("placement")
        date = meta.get("date")

        for card_name in parse_decklist_txt(txt_path):
            rows.append({
                "card_name": card_name,
                "points": points.get(card_name),
                "level": level,
                "placement": placement,
                "date": date,
            })

    df = pd.DataFrame(rows, columns=["card_name", "points", "level", "placement", "date"])
    logger.info(f"Occurrences dataframe built with {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Step 3: Extend with Scryfall features
# ---------------------------------------------------------------------------

def extend_with_scryfall(df: pd.DataFrame) -> pd.DataFrame:
    """Adds oracle features and tagger data columns to the dataframe."""
    oracle_df = scryfall.get_cards_df()
    oracle_index = oracle_df.set_index("name")

    # Pre-fetch tagger data per unique card to avoid redundant requests
    unique_cards = df["card_name"].unique()
    tagger_cache: dict[str, dict] = {}

    for card_name in unique_cards:
        if card_name not in oracle_index.index:
            logger.warning(f"Card not found in oracle data: {card_name}")
            tagger_cache[card_name] = {"card_tags": [], "relationships": []}
            continue
        row = oracle_index.loc[card_name]
        set_code = row["set"] if isinstance(row, pd.Series) else row.iloc[0]["set"]
        collector = row["collector_number"] if isinstance(row, pd.Series) else row.iloc[0]["collector_number"]
        try:
            tagger_cache[card_name] = scryfall.get_tagger_data(set_code, collector)
        except Exception as e:
            logger.warning(f"Tagger fetch failed for {card_name}: {e}")
            tagger_cache[card_name] = {"card_tags": [], "relationships": []}

    # Collect all relationship types across all cards
    all_rel_types: set[str] = set()
    for tagger_data in tagger_cache.values():
        for rel in tagger_data.get("relationships", []):
            all_rel_types.add(rel["relationship"])

    def get_oracle_feature(card_name: str, feature: str):
        if card_name not in oracle_index.index:
            return None
        row = oracle_index.loc[card_name]
        val = row[feature] if isinstance(row, pd.Series) else row.iloc[0][feature]
        return val if not (isinstance(val, float) and pd.isna(val)) else None

    def get_card_tags(card_name: str) -> list[str]:
        return tagger_cache.get(card_name, {}).get("card_tags", [])

    def get_relationships(card_name: str, rel_type: str) -> list[str]:
        rels = tagger_cache.get(card_name, {}).get("relationships", [])
        return [r["card"] for r in rels if r["relationship"] == rel_type]

    for feature in ORACLE_FEATURES:
        df[feature] = df["card_name"].map(lambda n, f=feature: get_oracle_feature(str(n), f))

    df["card_tags"] = df["card_name"].map(lambda n: get_card_tags(str(n)))

    for rel_type in sorted(all_rel_types):
        df[rel_type] = df["card_name"].map(lambda n, r=rel_type: get_relationships(str(n), r))

    logger.info(f"Scryfall features added. Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 4: Aggregate to one row per card
# ---------------------------------------------------------------------------

REL_COLUMNS = [
    "better-than", "colorshifted", "mirrors", "referenced-by",
    "references-to", "related-to", "similar-to", "with-body",
    "without-body", "worse-than",
]


def aggregate_per_card(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses the occurrences dataframe to one row per unique card.

    Adds tournament-derived features and deduplicates card-level oracle/tag columns.
    """
    # Tournament aggregate features
    tourn = df.groupby("card_name").agg(
        appearances=("card_name", "count"),
        avg_placement=("placement", "mean"),
        top4_rate=("placement", lambda x: (x <= 4).mean()),
        avg_level=("level", "mean"),
        high_level_rate=("level", lambda x: (x >= 2).mean()),
    ).reset_index()

    # Card-level columns are the same for every row of a given card — take first
    card_cols = ["card_name", "points"] + [
        c for c in df.columns
        if c not in {"card_name", "points", "level", "placement", "date"}
    ]
    card_level = df[card_cols].drop_duplicates(subset=["card_name"])

    agg = tourn.merge(card_level, on="card_name", how="left")
    logger.info(f"Aggregated dataframe: {len(agg)} unique cards.")
    return agg


# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, tag: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d")
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(DATA_DIR, f"chl_dataset{suffix}_{timestamp}.pkl")
    with open(path, "wb") as f:
        pickle.dump(df, f)
    logger.info(f"Dataset saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full pipeline.

    Returns:
        (full_df, per_card_df): the raw occurrences dataframe and the
        aggregated per-card dataframe ready for model training.
    """
    logger.info("Pipeline started.")

    points = load_points_dict()
    
    # Check if full dataset already exists
    existing_files = [f for f in os.listdir(DATA_DIR) if f.startswith("chl_dataset_full_")]
    if existing_files:
        existing_files.sort(reverse=True)
        latest_file = existing_files[0]
        date_str = latest_file.replace("chl_dataset_full_", "").replace(".pkl", "")
        print(f"\nFull dataset already exists from {date_str}.")
        response = input("Rebuild it? (y/n): ").strip().lower()
        if response != "y":
            logger.info("Skipping dataset rebuild.")
            with open(os.path.join(DATA_DIR, latest_file), "rb") as f:
                df = pickle.load(f)
        else:
            df = build_occurrences_df(points)
            df = extend_with_scryfall(df)
            save_dataset(df, tag="full")
    else:
        df = build_occurrences_df(points)
        df = extend_with_scryfall(df)
        save_dataset(df, tag="full")
    
    agg = aggregate_per_card(df)
    save_dataset(agg, tag="per_card")

    agg = aggregate_per_card(df)
    save_dataset(agg, tag="per_card")

    logger.info("Pipeline complete.")
    return df, agg


if __name__ == "__main__":
    run()
