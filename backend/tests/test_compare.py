from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
TEXT = (
    "Thanh pho mo tuyen xe buyt dien moi. Tuyen xe noi ga trung tam va benh vien. "
    "Gia ve xe buyt bang tuyen thuong. Nong dan trong rau sach trong nha kinh. "
    "Ho dung nuoc mua de tuoi rau. Vu thu hoach rau bat dau vao thang sau."
)


def test_compare_global_local_endpoint() -> None:
    response = client.post("/api/compare-global-local", json={"text": TEXT, "summary_sentences": 2})
    assert response.status_code == 200
    data = response.json()
    assert len(data["sentences"]) == 6
    assert data["global_nmf"]["method_id"] == "global_nmf_corpus_3260"
    assert data["local"]["method_id"] == "local_kl_nmf_mmr"
    assert len(data["global_nmf"]["selected_indices"]) == 2
    assert len(data["local"]["selected_indices"]) == 2
    assert data["global_nmf"]["metadata"]["trained_documents"] == 3260
    assert data["local"]["metadata"]["loss_name"] == "kullback-leibler"
    assert "selection_jaccard" in data["differences"]
