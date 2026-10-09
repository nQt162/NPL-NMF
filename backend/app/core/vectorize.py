"""Build a sentence-by-term, nonnegative TF-IDF matrix for one article."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_sentence_vectorizer(
    processed_sentences: list[str],
    *,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = None,
) -> tuple[np.ndarray, list[str], TfidfVectorizer | None]:
    """Fit local TF-IDF and return its matrix, feature names, and fitted model."""
    if not processed_sentences:
        return np.zeros((0, 0), dtype=float), [], None
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        norm="l2",
        min_df=1,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(processed_sentences).toarray()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return np.zeros((len(processed_sentences), 0), dtype=float), [], None
    return matrix, vectorizer.get_feature_names_out().tolist(), vectorizer


def vectorize_sentences(
    processed_sentences: list[str],
    *,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return V and its feature names; empty vocabularies become n-by-0 V."""
    matrix, terms, _ = fit_sentence_vectorizer(
        processed_sentences,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    return matrix, terms
