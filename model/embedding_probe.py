"""Empirical probe for what a tags_emb_*/keywords_emb_* dimension "means".

Individual dimensions of a dense sentence-transformer embedding don't
correspond to one human concept by construction (unlike the TF-IDF oracle
text features, where a dimension literally *is* a word) — there's no lookup
table for "dimension 364 = ramp". This instead answers the question
empirically: for a given dimension, which of the real, human-written tags
on this dataset's cards have the highest/lowest average value on it?

That's a correlational hunch for building intuition, not a rigorous claim.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from model.features import _embedder

_MIN_TAG_SUPPORT = 3  # a tag must appear on at least this many cards to be reported
_TOP_N_PER_SIDE = 4

_EMB_COL_RE = re.compile(r"^(tags|keywords)_emb_(\d+)$")

# Embeddings are expensive to compute (a sentence-transformer forward pass
# over every card) but cheap to slice once computed — cache per (per_card
# dataframe identity, source column) so probing several dimensions in one
# report build only pays the encode() cost once per source column.
_embedding_cache: dict[tuple[int, str], np.ndarray] = {}


def _source_column(feature_name: str) -> Optional[str]:
    match = _EMB_COL_RE.match(feature_name)
    if not match:
        return None
    return "card_tags" if match.group(1) == "tags" else "keywords"


def _dimension_index(feature_name: str) -> int:
    match = _EMB_COL_RE.match(feature_name)
    return int(match.group(2))


def _tag_embeddings(per_card: pd.DataFrame, source_col: str) -> np.ndarray:
    cache_key = (id(per_card), source_col)
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    texts = per_card[source_col].apply(lambda x: " ".join(str(t) for t in (x or []))).tolist()
    embeddings = np.asarray(_embedder().encode(texts, batch_size=64, show_progress_bar=False))
    _embedding_cache[cache_key] = embeddings
    return embeddings


def probe_dimension(per_card: pd.DataFrame, feature_name: str) -> Optional[dict]:
    """Returns {"positive": [(tag, delta), ...], "negative": [...]} — the
    real tags whose cards average highest/lowest on this one embedding
    dimension, sorted by how far each pulls from the dataset-wide mean.
    Returns None if `feature_name` isn't a tags_emb_*/keywords_emb_* column,
    the source list column isn't present, or no tag clears the minimum
    support threshold.
    """
    source_col = _source_column(feature_name)
    if source_col is None or source_col not in per_card.columns:
        return None

    dim = _dimension_index(feature_name)
    embeddings = _tag_embeddings(per_card, source_col)
    if dim >= embeddings.shape[1]:
        return None
    values = embeddings[:, dim]
    overall_mean = float(values.mean())

    tag_values: dict[str, list[float]] = {}
    for row_tags, val in zip(per_card[source_col], values):
        for tag in set(row_tags or []):
            tag_values.setdefault(tag, []).append(float(val))

    scored = [
        (tag, (sum(vals) / len(vals)) - overall_mean)
        for tag, vals in tag_values.items()
        if len(vals) >= _MIN_TAG_SUPPORT
    ]
    if not scored:
        return None

    scored.sort(key=lambda t: t[1], reverse=True)
    positive = [(tag, delta) for tag, delta in scored[:_TOP_N_PER_SIDE] if delta > 0]
    negative = [(tag, delta) for tag, delta in scored[-_TOP_N_PER_SIDE:] if delta < 0]
    negative.reverse()  # most negative first

    if not positive and not negative:
        return None
    return {"positive": positive, "negative": negative}
