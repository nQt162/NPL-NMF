from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)
TEXT = (
    "Thành phố mở tuyến xe buýt điện mới. Tuyến xe nối ga trung tâm và bệnh viện. "
    "Giá vé xe buýt bằng tuyến thường. Nông dân trồng rau sạch trong nhà kính. "
    "Họ dùng nước mưa để tưới rau. Vụ thu hoạch rau bắt đầu vào tháng sau."
)


def test_global_local_comparison_endpoint_is_removed() -> None:
    assert "/api/compare-global-local" not in app.openapi()["paths"]


def test_health_and_simulation_snapshots() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/api/simulate", json={"text": TEXT, "k": 2, "iterations": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["k"] == 2
    assert data["loss_name"] == "kullback-leibler"
    assert data["solver"] == "mu"
    assert data["alpha_W"] == 0.001
    assert data["alpha_H"] == 0.001
    assert data["l1_ratio"] == 1.0
    assert len(data["sentences"]) == 6
    assert len(data["V"]) == 6
    assert data["snapshots"][0]["iteration"] == 0
    assert data["snapshots"][-1]["iteration"] == 5
    assert data["snapshots"][-1]["loss"] == (
        data["snapshots"][-1]["data_loss"]
        + data["snapshots"][-1]["regularization_loss"]
    )


def test_simulation_supports_frobenius_with_shared_l1() -> None:
    response = client.post("/api/simulate", json={
        "text": TEXT,
        "k": 2,
        "iterations": 5,
        "method": "local_frobenius_mmr",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "local_frobenius_mmr"
    assert data["loss_name"] == "frobenius"
    assert data["alpha_W"] == 0.001
    assert data["alpha_H"] == 0.001
    assert data["l1_ratio"] == 1.0
    assert data["implementation"] == "educational_custom_mu"
    assert data["snapshots"][-1]["regularization_loss"] > 0
    assert data["snapshots"][-1]["loss"] == (
        data["snapshots"][-1]["data_loss"]
        + data["snapshots"][-1]["regularization_loss"]
    )


def test_removed_methods_are_rejected() -> None:
    for endpoint, payload in (
        ("/api/simulate", {"text": TEXT, "method": "snmf"}),
        ("/api/summarize", {"text": TEXT, "method": "nmfts_pairwise"}),
    ):
        assert client.post(endpoint, json=payload).status_code == 422


def test_simulation_and_summarizer_share_auto_k() -> None:
    simulation = client.post("/api/simulate", json={"text": TEXT, "iterations": 1})
    summary = client.post("/api/summarize", json={"text": TEXT, "summary_sentences": 2})

    assert simulation.status_code == 200
    assert summary.status_code == 200
    assert simulation.json()["k"] == summary.json()["k"]
    assert simulation.json()["k_candidates"] == summary.json()["k_candidates"]
    assert simulation.json()["k_selection_method"] == summary.json()["k_selection_method"]


def test_summarize_evaluate_and_validation() -> None:
    result = client.post("/api/summarize", json={"text": TEXT, "summary_sentences": 2})
    assert result.status_code == 200
    assert len(result.json()["selected_indices"]) == 2
    assert result.json()["k_selection_method"] == "positive_entry_masked_kl_cv"
    assert result.json()["k_candidates"]
    assert result.json()["loss_name"] == "kullback-leibler"
    assert result.json()["selection_method"] == "maximum_marginal_relevance"
    assert result.json()["mmr_lambda"] == 0.7
    assert result.json()["selection_trace"]
    assert result.json()["topics"][0]["representative_sentences"]
    assert all(item["score_round"] is not None for item in result.json()["sentence_analysis"])
    assert "factor_sparsity" not in result.json()

    evaluation = client.post("/api/evaluate", json={
        "text": TEXT,
        "reference_summary": "Thành phố mở tuyến xe buýt điện mới.",
        "k": 2,
        "summary_sentences": 2,
    })
    assert evaluation.status_code == 200
    evaluation_data = evaluation.json()
    assert evaluation_data["k"] == 2
    assert evaluation_data["kl_l1_mmr"]["summary"]
    assert evaluation_data["frobenius"]["summary"]
    for variant in (evaluation_data["kl_l1_mmr"], evaluation_data["frobenius"]):
        assert set(("rouge1_f1", "rouge2_f1", "rougeL_f1")) <= variant.keys()
        assert all(0 <= variant[metric] <= 1 for metric in ("rouge1_f1", "rouge2_f1", "rougeL_f1"))

    assert client.post("/api/summarize", json={"text": "   "}).status_code == 422
    assert client.post("/api/evaluate", json={
        "text": TEXT,
        "reference_summary": "   ",
    }).status_code == 422
    assert client.post("/api/simulate", json={"text": TEXT, "iterations": 21}).status_code == 422
    assert client.post("/api/simulate", json={"text": "Một câu."}).status_code == 422


@pytest.mark.parametrize("method", ["local_kl_mmr", "local_frobenius_mmr"])
def test_summarize_accepts_query_prf_for_both_losses(method: str) -> None:
    response = client.post("/api/summarize", json={
        "text": TEXT,
        "k": 2,
        "summary_sentences": 2,
        "method": method,
        "query": "rau",
        "query_weight": 0.8,
    })

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == method
    assert data["loss_name"] == ("frobenius" if method == "local_frobenius_mmr" else "kullback-leibler")
    assert data["query_feedback_applied"] is True
    assert data["query_feedback_reason"] is None
    assert data["query_weight"] == 0.8
    assert data["selection_method"] == "query_prf_mmr"
    assert data["pseudo_relevant_indices"]
    assert len(data["selected_indices"]) == 2
    assert data["selected_indices"] == sorted(set(data["selected_indices"]))
    assert any(row["query_relevance"] > 0 for row in data["sentence_analysis"])


@pytest.mark.parametrize("method", ["local_kl_mmr", "local_frobenius_mmr"])
def test_summarize_reports_oov_query_without_rejecting_request(method: str) -> None:
    response = client.post("/api/summarize", json={
        "text": TEXT,
        "k": 2,
        "summary_sentences": 2,
        "method": method,
        "query": "qzxv987654321",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["query_feedback_applied"] is False
    assert data["query_feedback_reason"]
    assert data["selection_method"] == "maximum_marginal_relevance"
    assert data["fallback_reason"] is None
    assert len(data["selected_indices"]) == 2
