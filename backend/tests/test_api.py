from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
TEXT = (
    "Thành phố mở tuyến xe buýt điện mới. Tuyến xe nối ga trung tâm và bệnh viện. "
    "Giá vé xe buýt bằng tuyến thường. Nông dân trồng rau sạch trong nhà kính. "
    "Họ dùng nước mưa để tưới rau. Vụ thu hoạch rau bắt đầu vào tháng sau."
)


def test_health_and_simulation_snapshots() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/api/simulate", json={"text": TEXT, "k": 2, "iterations": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data["sentences"]) == 6
    assert len(data["V"]) == 6
    assert data["snapshots"][0]["iteration"] == 0
    assert data["snapshots"][-1]["iteration"] == 5


def test_summarize_evaluate_and_validation() -> None:
    result = client.post("/api/summarize", json={"text": TEXT, "summary_sentences": 2})
    assert result.status_code == 200
    assert len(result.json()["selected_indices"]) == 2

    evaluation = client.post("/api/evaluate", json={
        "text": TEXT,
        "reference_summary": "Thành phố mở tuyến xe buýt điện mới.",
    })
    assert evaluation.status_code == 200
    assert 0 <= evaluation.json()["rouge1_f1"] <= 1

    assert client.post("/api/summarize", json={"text": "   "}).status_code == 422
    assert client.post("/api/simulate", json={"text": TEXT, "iterations": 21}).status_code == 422
    assert client.post("/api/simulate", json={"text": "Một câu."}).status_code == 422
