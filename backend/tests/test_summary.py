from app.core.baselines import lead_n
from app.core.evaluate import rouge_f1
from app.core.sentences import split_sentences
from app.core.summarize import summarize


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
    assert len(result["topics"]) == 2
    assert sum(row["selected"] for row in result["sentence_analysis"]) == 3


def test_fallback_and_lead_n() -> None:
    result = summarize("Một câu duy nhất.", summary_sentences=3)
    assert result["summary"] == "Một câu duy nhất."
    assert result["selected_indices"] == [0]
    assert result["fallback_reason"]
    assert lead_n(TEXT, 2) == (" ".join(split_sentences(TEXT)[:2]), [0, 1])
    empty_vocabulary = summarize("Và là. Của các.", summary_sentences=2)
    assert empty_vocabulary["selected_indices"] == [0, 1]
    assert empty_vocabulary["fallback_reason"]


def test_rouge_preserves_vietnamese_diacritics() -> None:
    identical = rouge_f1("Thành phố mở tuyến xe.", "Thành phố mở tuyến xe.")
    different = rouge_f1("Thành phố mở tuyến xe.", "Thanh pho mo tuyen xe.")
    assert identical == {"rouge1_f1": 1.0, "rouge2_f1": 1.0, "rougeL_f1": 1.0}
    assert different["rouge1_f1"] < 1.0
