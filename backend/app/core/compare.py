"""Global NMF helper retained for offline batch benchmark baselines."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np

try:  # scikit-learn exposes this warning in recent versions.
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover - older sklearn fallback.
    InconsistentVersionWarning = UserWarning

from .preprocess import preprocess_sentences
from .sentences import split_sentences


GLOBAL_MODEL_PATH = Path(__file__).with_name("nmf_corpus.joblib")
GLOBAL_TRAINED_DOCUMENTS = 3260
GLOBAL_MODEL_ID = "global_nmf_corpus_3260"


@lru_cache(maxsize=1)
def _load_global_bundle() -> dict[str, Any]:
    if not GLOBAL_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Global NMF model not found: {GLOBAL_MODEL_PATH}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        bundle = joblib.load(GLOBAL_MODEL_PATH)
    required = {"vectorizer", "nmf", "terms", "config"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Global NMF joblib must contain vectorizer, nmf, terms and config")
    return bundle


def _top_terms(H: np.ndarray, terms: list[str], topic_indices: list[int], top_n: int = 5) -> list[dict]:
    topics: list[dict] = []
    for topic_index in topic_indices:
        row = H[int(topic_index)]
        term_indices = np.argsort(-row, kind="stable")[:top_n]
        topics.append({
            "topic_index": int(topic_index),
            "top_terms": [terms[int(i)] for i in term_indices],
        })
    return topics


def _active_feature_count(matrix: Any) -> int:
    summed = np.asarray(matrix.sum(axis=0)).ravel()
    return int(np.count_nonzero(summed > 0))


def _global_summary(
    sentences: list[str],
    processed: list[str],
    budget: int,
    topic_count: int,
) -> dict[str, Any]:
    bundle = _load_global_bundle()
    vectorizer = bundle["vectorizer"]
    nmf = bundle["nmf"]
    terms = list(bundle["terms"])
    config = dict(bundle["config"])

    X = vectorizer.transform(processed)
    W = nmf.transform(X)
    H = np.asarray(nmf.components_, dtype=float)
    sentence_scores = np.asarray(W.sum(axis=1), dtype=float)
    ranked_sentences = sorted(range(len(sentences)), key=lambda index: (-sentence_scores[index], index))
    selected = sorted(ranked_sentences[:budget])
    selected_set = set(selected)

    topic_salience = np.asarray(W.sum(axis=0), dtype=float)
    if np.any(topic_salience > 0):
        topic_indices = np.argsort(-topic_salience, kind="stable")[:topic_count].tolist()
    else:
        topic_indices = list(range(min(topic_count, H.shape[0])))

    max_score = float(np.max(sentence_scores)) if len(sentence_scores) else 0.0
    analysis = []
    for index, sentence in enumerate(sentences):
        score = float(sentence_scores[index])
        analysis.append({
            "index": index,
            "text": sentence,
            "selected": index in selected_set,
            "score": score,
            "normalized_score": score / max_score if max_score > 0 else 0.0,
            "dominant_topic": int(np.argmax(W[index])) if W.shape[1] and np.any(W[index] > 0) else None,
        })

    input_terms = sorted({token for sentence in processed for token in sentence.split()})
    vocabulary = set(getattr(vectorizer, "vocabulary_", {}))
    matched_terms = [term for term in input_terms if term in vocabulary]
    oov_terms = [term for term in input_terms if term not in vocabulary]
    oov_rate = len(oov_terms) / len(input_terms) if input_terms else 0.0

    return {
        "method_id": GLOBAL_MODEL_ID,
        "label": "Global NMF cũ",
        "scope": "Train sẵn trên 3260 bài báo",
        "summary": " ".join(sentences[index] for index in selected),
        "selected_indices": selected,
        "topics": _top_terms(H, terms, topic_indices),
        "sentence_analysis": analysis,
        "metadata": {
            "model_file": str(GLOBAL_MODEL_PATH),
            "trained_documents": GLOBAL_TRAINED_DOCUMENTS,
            "n_components": int(getattr(nmf, "n_components", H.shape[0])),
            "loss_name": str(nmf.get_params().get("beta_loss", "frobenius")),
            "solver": str(nmf.get_params().get("solver", "unknown")),
            "init": str(nmf.get_params().get("init", "unknown")),
            "max_iter": int(nmf.get_params().get("max_iter", 0) or 0),
            "vocabulary_size": len(vocabulary),
            "active_terms": _active_feature_count(X),
            "input_terms": len(input_terms),
            "matched_input_terms": len(matched_terms),
            "oov_terms": len(oov_terms),
            "oov_rate": oov_rate,
            "oov_examples": oov_terms[:12],
            "config": config,
        },
    }


def summarize_global_text(
    text: str,
    *,
    summary_sentences: int = 3,
    topic_count: int = 3,
) -> dict[str, Any]:
    """Apply the archived corpus model to one document for batch baselines."""
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("text must contain at least one sentence")
    processed = preprocess_sentences(sentences)
    result = _global_summary(
        sentences,
        processed,
        min(summary_sentences, len(sentences)),
        max(1, topic_count),
    )
    return {
        "method": GLOBAL_MODEL_ID,
        "summary": result["summary"],
        "selected_indices": result["selected_indices"],
        "fallback_reason": None,
    }
