from app.core.baselines import lead_n
from app.core.evaluate import rouge_f1
from app.core.nmf import (
    SPARSE_KL_ALPHA_H,
    SPARSE_KL_ALPHA_W,
    SPARSE_KL_L1_RATIO,
    SPARSE_KL_LOSS_NAME,
    SPARSE_KL_MAX_ITER,
    SPARSE_KL_SOLVER,
)
from app.core.sentences import split_sentences
import app.core.summarize as summarize_module
from app.core.summarize import METHOD_FROBENIUS, METHOD_LOCAL_KL, summarize
import numpy as np
import pytest


TEXT = (
    "Thành phố mở tuyến xe buýt điện mới. Tuyến xe nối ga trung tâm và bệnh viện. "
    "Giá vé xe buýt bằng tuyến thường. Nông dân trồng rau sạch trong nhà kính. "
    "Họ dùng nước mưa để tưới rau. Vụ thu hoạch rau bắt đầu vào tháng sau."
)


def test_summary_preserves_original_sentences_and_budget() -> None:
    result = summarize(TEXT, k=2, summary_sentences=3)
    sentences = split_sentences(TEXT)
    indices = result["selected_indices"]
    assert len(indices) == 3
    assert indices == sorted(set(indices))
    assert result["summary"] == " ".join(sentences[i] for i in indices)
    assert result["fallback_reason"] is None
    assert result["k"] == 2
    assert result["loss_name"] == SPARSE_KL_LOSS_NAME
    assert result["solver"] == SPARSE_KL_SOLVER
    assert result["selection_method"] == "maximum_marginal_relevance"
    assert result["mmr_lambda"] == 0.7
    assert len(result["topics"]) == 2
    assert sum(row["selected"] for row in result["sentence_analysis"]) == 3
    assert all("relevance" in row and "redundancy" in row for row in result["sentence_analysis"])


def test_fallback_and_lead_n() -> None:
    result = summarize("Một câu duy nhất.", summary_sentences=3)
    assert result["summary"] == "Một câu duy nhất."
    assert result["selected_indices"] == [0]
    assert result["fallback_reason"] is None
    assert result["k"] == 1
    assert lead_n(TEXT, 2) == (" ".join(split_sentences(TEXT)[:2]), [0, 1])
    empty_vocabulary = summarize("Và là. Của các.", summary_sentences=2)
    assert empty_vocabulary["selected_indices"] == [0, 1]
    assert empty_vocabulary["fallback_reason"]


def test_summary_auto_k_uses_local_sentence_count() -> None:
    result = summarize(TEXT, summary_sentences=2)
    assert 1 <= result["k"] <= len(split_sentences(TEXT))
    assert result["k_selection_method"] == "positive_entry_masked_kl_cv"
    assert result["k_candidates"]
    assert len(result["topics"]) == result["k"]
    assert len(result["selected_indices"]) == 2


def test_summary_accepts_standard_mmr_lambda() -> None:
    result = summarize(TEXT, k=2, summary_sentences=2, mmr_lambda=0.35)
    assert result["selection_method"] == "maximum_marginal_relevance"
    assert result["mmr_lambda"] == 0.35
    assert all("coverage_gain" not in row for row in result["sentence_analysis"])


def test_summary_supports_local_frobenius_with_mmr() -> None:
    result = summarize(
        TEXT,
        k=2,
        summary_sentences=2,
        method=METHOD_FROBENIUS,
        mmr_lambda=0.6,
    )

    assert result["method"] == METHOD_FROBENIUS
    assert result["loss_name"] == "frobenius"
    assert result["objective"] == "local_frobenius_plus_l1"
    assert result["selection_method"] == "maximum_marginal_relevance"
    assert result["mmr_lambda"] == 0.6
    assert len(result["selected_indices"]) == 2
    assert result["regularization_loss"] > 0
    assert result["alpha_W"] == SPARSE_KL_ALPHA_W
    assert result["alpha_H"] == SPARSE_KL_ALPHA_H
    assert result["l1_ratio"] == SPARSE_KL_L1_RATIO


@pytest.mark.parametrize("method", [METHOD_LOCAL_KL, METHOD_FROBENIUS])
def test_query_prf_steers_selection_without_changing_nmf(method: str) -> None:
    options = {"k": 2, "summary_sentences": 1, "method": method}
    baseline = summarize(TEXT, **options)
    result = summarize(TEXT, query="rau", query_weight=1.0, **options)

    assert result["fallback_reason"] is None
    assert result["query_feedback_applied"] is True
    assert result["query_feedback_reason"] is None
    assert result["query_feedback_method"] == "pseudo_relevance_feedback"
    assert result["selection_method"] == "query_prf_mmr"
    assert result["pseudo_relevant_indices"]
    assert len(result["selected_indices"]) == 1
    assert "rau" in result["summary"].lower()
    assert max(row["query_relevance"] for row in result["sentence_analysis"]) == pytest.approx(1.0)
    for row in result["sentence_analysis"]:
        assert row["relevance"] == pytest.approx(row["query_relevance"])
    for key in ("k", "k_candidates", "topics", "loss_value", "data_loss", "regularization_loss"):
        assert result[key] == baseline[key]


@pytest.mark.parametrize("method", [METHOD_LOCAL_KL, METHOD_FROBENIUS])
def test_query_weight_controls_relevance_blend(method: str) -> None:
    options = {"k": 2, "summary_sentences": 2, "method": method}
    baseline = summarize(TEXT, **options)
    ignored = summarize(TEXT, query="rau", query_weight=0.0, **options)
    blended = summarize(TEXT, query="rau", query_weight=0.25, **options)

    assert ignored["selected_indices"] == baseline["selected_indices"]
    assert ignored["selection_trace"] == baseline["selection_trace"]
    for original, updated in zip(baseline["sentence_analysis"], blended["sentence_analysis"]):
        assert updated["relevance"] == pytest.approx(
            0.75 * original["relevance"] + 0.25 * updated["query_relevance"]
        )


@pytest.mark.parametrize("method", [METHOD_LOCAL_KL, METHOD_FROBENIUS])
@pytest.mark.parametrize("query", [None, "", "   ", "và là", "qzxv987654321"])
def test_unusable_query_preserves_standard_summary(method: str, query: str | None) -> None:
    options = {"k": 2, "summary_sentences": 2, "method": method}
    baseline = summarize(TEXT, **options)
    result = summarize(TEXT, query=query, **options)

    assert result["fallback_reason"] is None
    assert result["query_feedback_applied"] is False
    assert result["pseudo_relevant_indices"] == []
    assert result["selection_method"] == "maximum_marginal_relevance"
    assert result["summary"] == baseline["summary"]
    assert result["selected_indices"] == baseline["selected_indices"]
    assert result["sentence_analysis"] == baseline["sentence_analysis"]
    assert result["loss_value"] == baseline["loss_value"]
    if query:
        assert result["query_feedback_reason"]
    else:
        assert result["query_feedback_reason"] is None


def test_summary_uses_kl_mu_sparse_nmf(monkeypatch) -> None:
    captured = {}

    class FakeNMF:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.components_ = None

        def fit_transform(self, V):
            self.components_ = np.ones((captured["n_components"], V.shape[1]))
            return np.ones((V.shape[0], captured["n_components"]))

    monkeypatch.setattr(summarize_module, "SklearnNMF", FakeNMF)
    summarize(TEXT, k=2, summary_sentences=2)

    assert captured["solver"] == SPARSE_KL_SOLVER
    assert captured["beta_loss"] == SPARSE_KL_LOSS_NAME
    assert captured["alpha_W"] == SPARSE_KL_ALPHA_W
    assert captured["alpha_H"] == SPARSE_KL_ALPHA_H
    assert captured["l1_ratio"] == SPARSE_KL_L1_RATIO
    assert captured["max_iter"] == SPARSE_KL_MAX_ITER
    assert captured["init"] == "nndsvdar"
    assert captured["random_state"] == 42


def test_both_losses_use_same_l1_and_seeded_stochastic_initialization(monkeypatch) -> None:
    calls = []

    class FakeNMF:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.components_ = None

        def fit_transform(self, V):
            self.components_ = np.ones((calls[-1]["n_components"], V.shape[1]))
            return np.ones((V.shape[0], calls[-1]["n_components"]))

    monkeypatch.setattr(summarize_module, "SklearnNMF", FakeNMF)
    summarize(TEXT, k=2, summary_sentences=2, seed=17)
    summarize(TEXT, k=2, summary_sentences=2, seed=23, method=METHOD_FROBENIUS)

    assert calls[0]["alpha_W"] == calls[1]["alpha_W"] == SPARSE_KL_ALPHA_W
    assert calls[0]["alpha_H"] == calls[1]["alpha_H"] == SPARSE_KL_ALPHA_H
    assert calls[0]["l1_ratio"] == calls[1]["l1_ratio"] == SPARSE_KL_L1_RATIO
    assert calls[0]["init"] == calls[1]["init"] == "nndsvdar"
    assert calls[0]["random_state"] == 17
    assert calls[1]["random_state"] == 23


def test_mmr_scores_are_traced_per_greedy_round() -> None:
    result = summarize(TEXT, k=2, summary_sentences=3)
    trace = result["selection_trace"]
    assert len(trace) == 3
    assert [entry["round"] for entry in trace] == [1, 2, 3]
    assert {entry["selected_index"] for entry in trace} == set(result["selected_indices"])
    assert all(any(candidate["index"] == entry["selected_index"] for candidate in entry["candidates"]) for entry in trace)


def test_topic_terms_include_representative_sentences() -> None:
    result = summarize(TEXT, k=2, summary_sentences=2)
    assert all(topic["top_terms"] for topic in result["topics"])
    assert all(topic["representative_sentences"] for topic in result["topics"])


def test_large_l1_that_collapses_factors_uses_explained_fallback() -> None:
    result = summarize(
        TEXT,
        k=2,
        summary_sentences=2,
        method=METHOD_FROBENIUS,
        alpha_w=0.1,
        alpha_h=0.1,
    )
    assert len(result["selected_indices"]) == 2
    assert result["fallback_reason"]
    assert "collapsed to zero" in result["fallback_reason"]


def test_rouge_preserves_vietnamese_diacritics() -> None:
    identical = rouge_f1("Thành phố mở tuyến xe.", "Thành phố mở tuyến xe.")
    different = rouge_f1("Thành phố mở tuyến xe.", "Thanh pho mo tuyen xe.")
    assert identical == {"rouge1_f1": 1.0, "rouge2_f1": 1.0, "rougeL_f1": 1.0}
    assert different["rouge1_f1"] < 1.0
