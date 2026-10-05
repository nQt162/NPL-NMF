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
from app.core.summarize import summarize
import numpy as np


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
    assert len(result["topics"]) == 2
    assert sum(row["selected"] for row in result["sentence_analysis"]) == 3


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
    assert result["k_selection_method"] == "masked_kl_imputation"
    assert result["k_candidates"]
    assert len(result["topics"]) == result["k"]
    assert len(result["selected_indices"]) == 2


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
    assert captured["init"] == "nndsvda"


def test_rouge_preserves_vietnamese_diacritics() -> None:
    identical = rouge_f1("Thành phố mở tuyến xe.", "Thành phố mở tuyến xe.")
    different = rouge_f1("Thành phố mở tuyến xe.", "Thanh pho mo tuyen xe.")
    assert identical == {"rouge1_f1": 1.0, "rouge2_f1": 1.0, "rougeL_f1": 1.0}
    assert different["rouge1_f1"] < 1.0
