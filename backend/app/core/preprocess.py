"""Vietnamese tokenization for TF-IDF and accent-preserving ROUGE."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from underthesea import word_tokenize


_WORD = re.compile(r"\w+", re.UNICODE)
_STOPWORDS_PATH = Path(__file__).with_name("stopwords_vi.txt")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    return frozenset(
        line.strip().lower()
        for line in _STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def tokenize_words(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Keep Vietnamese diacritics and underthesea's multiword tokens."""
    segmented = word_tokenize(text.lower(), format="text")
    words = [token for token in segmented.split() if _WORD.fullmatch(token)]
    if remove_stopwords:
        words = [token for token in words if token not in _stopwords()]
    return words


def preprocess_sentences(sentences: list[str]) -> list[str]:
    """Return space-tokenized copies; the caller retains original sentences."""
    return [" ".join(tokenize_words(sentence)) for sentence in sentences]
