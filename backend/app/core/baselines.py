"""Simple extractive reference methods."""

from __future__ import annotations

from .sentences import split_sentences


def lead_n(text_or_sentences: str | list[str], n: int = 3) -> tuple[str, list[int]]:
    """Return the first n original sentences and their zero-based indices."""
    if n < 1:
        raise ValueError("n phải lớn hơn 0")
    sentences = split_sentences(text_or_sentences) if isinstance(text_or_sentences, str) else text_or_sentences
    indices = list(range(min(n, len(sentences))))
    return " ".join(sentences[:n]), indices
