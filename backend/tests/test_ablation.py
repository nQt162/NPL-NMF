import json

import numpy as np
from fastapi.testclient import TestClient

from app.core import ablation
from app.core.ablation import _configuration_params, parse_articles_jsonl, run_l1_ablation
from app.core.summarize import _factor_sparsity_stats
from app.main import app


TEXT = (
    "Thành phố mở tuyến xe buýt điện mới. Tuyến xe kết nối ga trung tâm và bệnh viện. "
    "Giá vé được trợ giá bằng ngân sách. Người dân có thể tra cứu lịch chạy. "
    "Đơn vị vận hành theo dõi mức độ hài lòng. Thành phố sẽ đánh giá tuyến xe."
)
ARTICLE = {
    "id": "article-1",
    "article": TEXT,
    "reference_summary": "Thành phố mở tuyến xe buýt điện mới và theo dõi hiệu quả.",
}


def test_configurations_only_penalize_requested_factors() -> None:
    assert _configuration_params(0.1) == {
        "kl_no_l1": (0.0, 0.0, 1.0),
        "kl_l1_w_only": (0.1, 0.0, 1.0),
        "kl_l1_h_only": (0.0, 0.1, 1.0),
        "kl_l1_w_h": (0.1, 0.1, 1.0),
    }


def test_factor_sparsity_statistics_use_explicit_absolute_threshold() -> None:
    W = np.array([[0.0, 0.4], [1e-9, 0.6]])
    H = np.array([[0.0, 0.3, 1e-10], [0.2, 0.0, 0.4]])
    stats = _factor_sparsity_stats(W, H, 1e-8)
    assert stats == {
        "zero_threshold": 1e-8,
        "w_zero_fraction": 0.5,
        "h_zero_fraction": 0.5,
        "active_topics_per_sentence": 1.0,
        "active_terms_per_topic": 1.5,
    }


def test_jsonl_parser_reports_invalid_and_duplicate_rows() -> None:
    content = "\n".join([
        json.dumps(ARTICLE, ensure_ascii=False),
        "not json",
        json.dumps(ARTICLE, ensure_ascii=False),
    ])
    rows, errors = parse_articles_jsonl(content)
    assert [row["id"] for row in rows] == ["article-1"]
    assert len(errors) == 2
    assert errors[0]["line"] == 2
    assert "Duplicate id" in errors[1]["reason"]


def test_ablation_holds_k_and_seed_and_writes_paired_artifacts(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_summarize(**kwargs):
        calls.append(kwargs)
        return {
            "k": kwargs["k"],
            "summary": ARTICLE["reference_summary"],
            "selected_indices": [0, 1],
            "fallback_reason": None,
            "factor_sparsity": {
                "w_zero_fraction": 0.2,
                "h_zero_fraction": 0.3,
                "active_topics_per_sentence": 1.2,
                "active_terms_per_topic": 2.5,
            },
        }

    monkeypatch.setattr(ablation, "summarize", fake_summarize)
    report = run_l1_ablation(
        [ARTICLE],
        tmp_path,
        split="validation",
        k=2,
        summary_sentences=2,
        alpha=0.1,
        seeds=[7, 11],
    )

    assert len(calls) == 8
    assert {call["k"] for call in calls} == {2}
    assert {call["seed"] for call in calls} == {7, 11}
    assert all(call["query"] is None and call["include_factor_stats"] for call in calls)
    assert [(call["alpha_w"], call["alpha_h"], call["l1_ratio"]) for call in calls[:4]] == [
        (0.0, 0.0, 1.0),
        (0.1, 0.0, 1.0),
        (0.0, 0.1, 1.0),
        (0.1, 0.1, 1.0),
    ]
    assert report["paired_deltas_vs_kl_no_l1"]["kl_l1_w_h"]["paired_count"] == 2
    assert report["paired_deltas_vs_kl_no_l1"]["kl_l1_w_h"]["rouge"]["rouge1_f1"]["mean"] == 0
    assert (tmp_path / "ablation_metrics.csv").is_file()
    assert (tmp_path / "ablation_predictions.jsonl").is_file()
    assert (tmp_path / "ablation_report.json").is_file()


def test_ablation_api_queues_job_and_exposes_artifacts(tmp_path, monkeypatch) -> None:
    import app.core.ablation_jobs as jobs
    import app.main as main_module

    monkeypatch.setattr(jobs, "ABLATION_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "ABLATION_OUTPUT_ROOT", tmp_path)
    received_options = {}

    def fake_run(articles, output_dir, *, progress_callback, **kwargs):
        received_options.update(kwargs)
        progress_callback(1, 1)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in jobs.ABLATION_ARTIFACTS.values():
            (output_dir / name).write_text("artifact", encoding="utf-8")
        return {"valid_articles": len(articles), "methods": {}}

    monkeypatch.setattr(jobs, "run_l1_ablation", fake_run)
    client = TestClient(app)
    response = client.post("/api/ablation", json={
        "article_id": ARTICLE["id"],
        "article": ARTICLE["article"],
        "reference_summary": ARTICLE["reference_summary"],
        "k": 2,
        "seeds": [42],
    })
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    status = client.get(f"/api/ablation/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "complete"
    assert status.json()["report"]["valid_articles"] == 1
    assert received_options["split"] == "pilot"
    assert received_options["dataset_name"] == ARTICLE["id"]
    assert client.get(f"/api/ablation/{run_id}/files/metrics").text == "artifact"
    assert client.get(f"/api/ablation/{run_id}/files/unknown").status_code == 404


def test_ablation_api_rejects_blank_reference_summary() -> None:
    client = TestClient(app)
    response = client.post("/api/ablation", json={
        "article": ARTICLE["article"],
        "reference_summary": "   ",
    })
    assert response.status_code == 422
