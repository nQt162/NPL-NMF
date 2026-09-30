"""Load and apply a corpus-trained NMF model when one is available."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning


DEFAULT_MODEL_PATH = Path(__file__).with_name("nmf_corpus.joblib")


@dataclass(frozen=True)
class CorpusNMFModel:
    vectorizer: object
    nmf: object
    terms: list[str]
    config: dict
    path: Path

    def project(
        self,
        processed_sentences: list[str],
        requested_topics: int,
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
        """Transform sentences with the saved vectorizer and NMF topic space."""
        matrix = self.vectorizer.transform(processed_sentences)
        V = matrix.toarray()
        W_full = np.asarray(self.nmf.transform(matrix), dtype=float)
        H_full = np.asarray(self.nmf.components_, dtype=float)
        if W_full.ndim != 2 or H_full.ndim != 2 or W_full.shape[1] != H_full.shape[0]:
            raise ValueError("Corpus NMF model has incompatible W/H shapes")
        topic_count = min(max(requested_topics, 1), W_full.shape[1])
        salience = W_full.sum(axis=0)
        topic_indices = np.argsort(-salience, kind="stable")[:topic_count]
        return V, self.terms, W_full[:, topic_indices], H_full[topic_indices]


def _model_path() -> Path:
    configured = os.getenv("TOPICSUM_NMF_CORPUS_MODEL")
    return Path(configured) if configured else DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def load_corpus_nmf_model() -> CorpusNMFModel | None:
    """Return the saved corpus NMF model, or None when no model file exists."""
    path = _model_path()
    if not path.is_file():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Corpus NMF model must be a joblib dict bundle")
    vectorizer = bundle.get("vectorizer")
    nmf = bundle.get("nmf")
    if vectorizer is None or nmf is None:
        raise ValueError("Corpus NMF model must contain 'vectorizer' and 'nmf'")
    if not hasattr(vectorizer, "transform"):
        raise ValueError("Corpus NMF vectorizer does not support transform()")
    if not hasattr(nmf, "transform") or not hasattr(nmf, "components_"):
        raise ValueError("Corpus NMF estimator does not support transform() and components_")
    terms = bundle.get("terms")
    if not terms:
        terms = vectorizer.get_feature_names_out().tolist()
    return CorpusNMFModel(
        vectorizer=vectorizer,
        nmf=nmf,
        terms=list(terms),
        config=dict(bundle.get("config") or {}),
        path=path,
    )
