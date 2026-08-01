"""Feature engineering for the CHL points prediction model.

Transforms the per-card aggregated dataframe into a numeric feature matrix
ready for model training or inference.
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder

# Sentence transformer model — downloaded once and cached locally by the library
_EMBEDDER: Optional[SentenceTransformer] = None
EMBED_MODEL = "all-MiniLM-L6-v2"

# TF-IDF over oracle text — "english" drops common stopwords (if, the, a, ...)
# before vectorising. Capped vocabulary keeps dimensionality sane relative to
# the (small) card dataset.
TFIDF_MAX_FEATURES = 200
TFIDF_MIN_DF = 2

COLORS = ["W", "U", "B", "R", "G"]
RARITY_ORDER = [["common", "uncommon", "rare", "mythic"]]

REL_COLUMNS = [
    "better-than", "colorshifted", "mirrors", "referenced-by",
    "references-to", "related-to", "similar-to", "with-body",
    "without-body", "worse-than",
]


@dataclass
class FeaturePreprocessors:
    """Transformers fit once on the training set and reused at inference time.

    Fitting these fresh on every build_feature_matrix() call (the original
    behaviour) silently breaks on small batches: e.g. a single inference row
    with a missing value has no other rows to compute a median from, so the
    value stays NaN and crashes any model that can't handle NaN natively.
    """
    tfidf_vectorizer: TfidfVectorizer
    imputer: SimpleImputer
    rarity_encoder: OrdinalEncoder
    keyword_binarizer: MultiLabelBinarizer


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


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug or "unknown"


def _tag_keyword_string(row: pd.Series) -> str:
    tags = row.get("card_tags", []) or []
    keywords = row.get("keywords", []) or []
    parts = list(tags) + list(keywords)
    return " ".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    preprocessors: Optional[FeaturePreprocessors] = None,
) -> tuple[pd.DataFrame, FeaturePreprocessors]:
    """Transforms the per-card dataframe into a numeric feature matrix.

    Args:
        df: Per-card aggregated dataframe from pipeline.aggregate_per_card().
        preprocessors: Transformers already fit at training time (TF-IDF
            vectorizer, median imputer, rarity encoder). Pass this in at
            inference time — including for small batches / single-card
            lookups — so vocabulary/statistics match training rather than
            being refit (and potentially undefined) on whatever happens to
            be in `df`. If None, fresh transformers are fit on `df`.

    Returns:
        Tuple of (feature matrix, the fitted preprocessors). The feature
        matrix has one row per card, all-numeric columns, row order matching
        df.index.
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

    imputer = preprocessors.imputer if preprocessors else SimpleImputer(strategy="median")
    rarity_encoder = preprocessors.rarity_encoder if preprocessors else OrdinalEncoder(
        categories=RARITY_ORDER, handle_unknown="use_encoded_value", unknown_value=-1
    )

    # Impute numeric columns with median (fit at training time, reused thereafter)
    if preprocessors is None:
        feat_df[:] = imputer.fit_transform(feat_df)
    else:
        feat_df[:] = imputer.transform(feat_df)

    # Rarity ordinal encode — join back by index
    # .to_numpy(dtype=object), not .values: pandas' pyarrow-backed string
    # arrays (default in pandas >=3.0, and more likely to be active once
    # pyarrow is already imported elsewhere in-process, e.g. by Streamlit)
    # don't support .reshape().
    rarity_series = df.get("rarity", pd.Series("common", index=df.index))
    rarity_series = rarity_series.fillna("common").map(lambda r: str(r) if r else "common")
    rarity_values = rarity_series.to_numpy(dtype=object).reshape(-1, 1)
    if preprocessors is None:
        feat_df["rarity"] = rarity_encoder.fit_transform(rarity_values)
    else:
        feat_df["rarity"] = rarity_encoder.transform(rarity_values)

    # card_tags stay as sentence-transformer embeddings: ~1,560 distinct
    # values with heavy near-synonym overlap ("removal" / "removal-creature"
    # / "spot removal"), which is exactly what a dense embedding can partially
    # capture (and a flat per-tag encoding can't).
    tags = df.get("card_tags", pd.Series([[]] * len(df), index=df.index))
    tags = tags.apply(lambda x: " ".join(str(p) for p in (x or [])))
    tags_embeddings = _embedder().encode(tags.tolist(), show_progress_bar=True, batch_size=64)
    tags_df = pd.DataFrame(
        tags_embeddings,
        index=df.index,
        columns=[f"tags_emb_{i}" for i in range(tags_embeddings.shape[1])],
    )

    # keywords, in contrast, are a small (~164 distinct values) fixed official
    # vocabulary with no synonym structure to exploit and under 1 keyword/card
    # on average — one-hot per keyword is smaller *and* directly
    # interpretable (SHAP names the actual keyword), unlike a dense embedding.
    keyword_lists = df.get("keywords", pd.Series([[]] * len(df), index=df.index))
    keyword_lists = keyword_lists.apply(lambda x: list(x) if isinstance(x, list) else [])
    keyword_binarizer = preprocessors.keyword_binarizer if preprocessors else MultiLabelBinarizer()
    if preprocessors is None:
        keyword_matrix = keyword_binarizer.fit_transform(keyword_lists)
    else:
        keyword_matrix = keyword_binarizer.transform(keyword_lists)
    keywords_df = pd.DataFrame(
        keyword_matrix,
        index=df.index,
        columns=[f"keyword_{_slugify(c)}" for c in keyword_binarizer.classes_],
    )

    embed_df = pd.concat([tags_df, keywords_df], axis=1)

    # TF-IDF over oracle (rules) text — stopwords pruned before vectorising.
    oracle_text = df.get("oracle_text", pd.Series("", index=df.index))
    oracle_text = oracle_text.fillna("").map(str)

    tfidf_vectorizer = preprocessors.tfidf_vectorizer if preprocessors else TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
    )
    if preprocessors is None:
        tfidf_matrix = tfidf_vectorizer.fit_transform(oracle_text)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(oracle_text)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=df.index,
        columns=[f"oracle_tfidf_{term}" for term in tfidf_vectorizer.get_feature_names_out()],
    )

    if preprocessors is None:
        preprocessors = FeaturePreprocessors(
            tfidf_vectorizer=tfidf_vectorizer,
            imputer=imputer,
            rarity_encoder=rarity_encoder,
            keyword_binarizer=keyword_binarizer,
        )

    return pd.concat([feat_df, embed_df, tfidf_df], axis=1), preprocessors
