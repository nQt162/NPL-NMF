"""Build a sentence-by-term, nonnegative TF-IDF matrix for one article."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_sentences(processed_sentences: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return V and its column terms; empty vocabularies become n-by-0 V."""
    if not processed_sentences:
        return np.zeros((0, 0), dtype=float), []
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        norm="l2",
        min_df=1,
    )
    try:
        matrix = vectorizer.fit_transform(processed_sentences).toarray()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return np.zeros((len(processed_sentences), 0), dtype=float), []
    return matrix, vectorizer.get_feature_names_out().tolist()
