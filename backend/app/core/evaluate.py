"""Vietnamese-aware ROUGE metrics shared by API and batch experiments."""

from __future__ import annotations

from rouge_score import rouge_scorer

from .preprocess import tokenize_words


class VietnameseTokenizer:
    """Preserve diacritics and the same word segmentation for every method."""

    def tokenize(self, text: str) -> list[str]:
        return tokenize_words(text, remove_stopwords=False)


_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=VietnameseTokenizer(),
)


def rouge_f1(reference: str, candidate: str) -> dict[str, float]:
    """Return F1 values on the 0..1 scale; do not remove Vietnamese accents."""
    scores = _SCORER.score(reference, candidate)
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }
