"""Feature engineering for the CHL points prediction model.

Transforms the per-card aggregated dataframe into a numeric feature matrix
ready for model training or inference.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Sentence transformer model — downloaded once and cached locally by the library
_EMBEDDER: Optional[SentenceTransformer] = None
EMBED_MODEL = "all-MiniLM-L6-v2"

COLORS = ["W", "U", "B", "R", "G"]
RARITY_ORDER = [["common", "uncommon", "rare", "mythic"]]

REL_COLUMNS = [
    "better-than", "colorshifted", "mirrors", "referenced-by",
    "references-to", "related-to", "similar-to", "with-body",
    "without-body", "worse-than",
]


def _embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def _parse_type_line(type_line) -> dict:
    s = str(type_line) if type_line else ""
    return {
        "is_creature":     int("Creature" in s),
        "is_instant":      int("Instant" in s),
        "is_sorcery":      int("Sorcery" in s),
        "is_artifact":     int("Artifact" in s),
        "is_land":         int("Land" in s),
        "is_enchantment":  int("Enchantment" in s),
        "is_planeswalker": int("Planeswalker" in s),
        "is_legendary":    int("Legendary" in s),
    }


def _parse_mana_cost(mana_cost) -> dict:
    s = str(mana_cost) if mana_cost else ""
    pips: dict[str, int] = {f"pips_{c}": 0 for c in COLORS}
    pips["pips_generic"] = 0
    pips["pips_X"] = 0
    for sym in re.findall(r"\{([^}]+)\}", s):
        if sym in COLORS:
            pips[f"pips_{sym}"] += 1
        elif sym == "X":
            pips["pips_X"] += 1
        elif re.match(r"^\d+$", sym):
            pips["pips_generic"] += int(sym)
        elif "/" in sym:
            # hybrid pip — count toward each component colour
            for part in sym.split("/"):
                if part in COLORS:
                    pips[f"pips_{part}"] += 1
    return pips


def _multihot_colors(color_list) -> dict:
    colors = color_list if isinstance(color_list, list) else []
    return {f"color_{c}": int(c in colors) for c in COLORS}


def _multihot_produced_mana(mana_list) -> dict:
    mana = mana_list if isinstance(mana_list, list) else []
    return {f"produces_{c}": int(c in mana) for c in COLORS}


def _rel_counts(row: pd.Series) -> dict:
    counts = {}
    for col in REL_COLUMNS:
        val = row.get(col, [])
        counts[f"n_{col}"] = len(val) if isinstance(val, list) else 0
    return counts


def _tag_keyword_string(row: pd.Series) -> str:
    tags = row.get("card_tags", []) or []
    keywords = row.get("keywords", []) or []
    parts = list(tags) + list(keywords)
    return " ".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the per-card dataframe into a numeric feature matrix.

    Args:
        df: Per-card aggregated dataframe from pipeline.aggregate_per_card().

    Returns:
        Feature matrix with one row per card, all-numeric columns.
        Row order matches df.index.
    """
    records = []

    for _, row in df.iterrows():
        feat: dict = {}

        # Tournament aggregates
        feat["appearances"]      = row.get("appearances", 0) or 0
        feat["avg_placement"]    = row.get("avg_placement") or np.nan
        feat["top4_rate"]        = row.get("top4_rate") or np.nan
        feat["avg_level"]        = row.get("avg_level") or np.nan
        feat["high_level_rate"]  = row.get("high_level_rate") or np.nan

        # Numeric oracle
        feat["cmc"]          = row.get("cmc") or np.nan
        feat["edhrec_rank"]  = row.get("edhrec_rank") or np.nan
        try:
            feat["power"]    = float(row.get("power")) if row.get("power") is not None else np.nan
        except (ValueError, TypeError):
            feat["power"]    = np.nan
        try:
            feat["toughness"] = float(row.get("toughness")) if row.get("toughness") is not None else np.nan
        except (ValueError, TypeError):
            feat["toughness"] = np.nan

        # game_changer boolean
        gc = row.get("game_changer")
        feat["game_changer"] = int(bool(gc)) if gc is not None else 0

        # Structured categoricals
        type_feats = _parse_type_line(row.get("type_line"))
        mana_feats = _parse_mana_cost(row.get("mana_cost"))
        produced_feats = _multihot_produced_mana(row.get("produced_mana"))
        feat.update(type_feats)
        feat.update(mana_feats)
        feat.update(_multihot_colors(row.get("color_identity")))
        feat.update(produced_feats)
        feat.update(_rel_counts(row))

        # Domain-specific derived features
        cmc = feat.get("cmc") or np.nan
        feat["is_zero_cost"] = int(not np.isnan(cmc) and cmc == 0)

        total_colored_pips = sum(mana_feats.get(f"pips_{c}", 0) for c in COLORS)
        feat["total_colored_pips"] = total_colored_pips
        feat["pip_density"] = (total_colored_pips / cmc) if (not np.isnan(cmc) and cmc > 0) else np.nan

        feat["produces_mana_count"] = sum(produced_feats.get(f"produces_{c}", 0) for c in COLORS)

        power = feat.get("power", np.nan)
        feat["power_above_cost"] = int(
            not np.isnan(power) and not np.isnan(cmc) and power > cmc
        )
        feat["efficient_creature"] = int(
            type_feats.get("is_creature", 0) == 1
            and not np.isnan(cmc) and cmc <= 2
        )

        records.append(feat)

    feat_df = pd.DataFrame(records, index=df.index)

    # Impute numeric columns with median (fit on the data we have)
    imputer = SimpleImputer(strategy="median")
    feat_df[:] = imputer.fit_transform(feat_df)

    # Rarity ordinal encode — join back by index
    rarity_series = df["rarity"].fillna("common").map(lambda r: str(r) if r else "common")
    enc = OrdinalEncoder(categories=RARITY_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)
    feat_df["rarity"] = enc.fit_transform(rarity_series.values.reshape(-1, 1))

    # Sentence-transformer embeddings for card_tags and keywords (separate)
    tags = df.get("card_tags", []).apply(lambda x: " ".join(str(p) for p in (x or [])))
    keywords = df.get("keywords", []).apply(lambda x: " ".join(str(p) for p in (x or [])))
    
    tags_embeddings = _embedder().encode(tags.tolist(), show_progress_bar=True, batch_size=64)
    keywords_embeddings = _embedder().encode(keywords.tolist(), show_progress_bar=True, batch_size=64)
    
    tags_df = pd.DataFrame(
        tags_embeddings,
        index=df.index,
        columns=[f"tags_emb_{i}" for i in range(tags_embeddings.shape[1])],
    )
    keywords_df = pd.DataFrame(
        keywords_embeddings,
        index=df.index,
        columns=[f"keywords_emb_{i}" for i in range(keywords_embeddings.shape[1])],
    )
    
    embed_df = pd.concat([tags_df, keywords_df], axis=1)

    return pd.concat([feat_df, embed_df], axis=1)
