# TopicSum (NPL-NMF)

Bộ khung dự án cho ứng dụng tóm tắt **trích xuất** văn bản tiếng Việt bằng NMF. Yêu cầu chi tiết và tiêu chí hoàn thành nằm trong [TopicSum_AGENT.md](TopicSum_AGENT.md).

## Trạng thái hiện tại

Backend đã có xử lý tiếng Việt, NMF tự cài đặt, API, script chuẩn bị dữ liệu/thí nghiệm và kiểm thử. Frontend vẫn là khung TODO. `results/metrics.csv` và `results/predictions.jsonl` hiện rỗng vì chưa có tập đánh giá hợp lệ.

## Cấu trúc và việc cần làm

| Khu vực | Việc chính |
| --- | --- |
| `backend/app/core/` | Tách câu, tiền xử lý tiếng Việt, TF-IDF, NMF tự cài đặt, chọn câu, baseline, ROUGE. Dùng **một** hàm `fit_nmf` cho mô phỏng và tóm tắt. |
| `backend/app/main.py`, `schemas.py` | FastAPI, ba endpoint chính, health check và validation. |
| `frontend/src/` | Ba màn hình Visualizer, Summarizer, Evaluation; gọi dữ liệu thật từ API. |
| `data/` | Ví dụ tự viết, dữ liệu có quyền sử dụng và chia train/val/test theo bài. |
| `scripts/` | Chuẩn bị dữ liệu và chạy so sánh NMF với Lead-N. |
| `backend/tests/` | Kiểm tra NMF, thứ tự câu, giới hạn số câu và fallback. |
| `results/` | Nơi xuất kết quả thí nghiệm sau khi có dữ liệu hợp lệ. |

## Cài đặt và chạy backend (PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

API tại `http://127.0.0.1:8000`, tài liệu tương tác tại `/docs`. Mở một terminal khác trong `backend` để chạy kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Các endpoint: `GET /health`, `POST /api/simulate`, `POST /api/summarize`, `POST /api/evaluate`. `/api/evaluate` chỉ chấm một mẫu; batch chạy bằng CLI. Ví dụ request tóm tắt:

```json
{"text":"Thành phố mở tuyến xe buýt điện. Tuyến xe nối ga và bệnh viện. Giá vé bằng tuyến thường.","k":2,"summary_sentences":2}
```

## Chuẩn bị dữ liệu và chạy thí nghiệm

Điền `data/raw/articles.jsonl` bằng các bài có quyền sử dụng; mỗi dòng cần `id`, `article`, `reference_summary`, `source`, nên có thêm `license`. Từ thư mục `backend` đã cài package:

```powershell
.\.venv\Scripts\python.exe ..\scripts\prepare_data.py --input ..\data\raw\articles.jsonl --output ..\data\splits
.\.venv\Scripts\python.exe ..\scripts\run_experiment.py --input ..\data\splits\val.jsonl --output ..\results
```

Chọn `k`, ngân sách và các hệ số trên validation. Sau khi chốt cấu hình, chạy test đúng một lần bằng `--input ..\data\splits\test.jsonl`; CLI nhận `--k`, `--summary-sentences`, `--alpha`, `--beta`, `--gamma`, `--seed`. Kết quả gồm `metrics.csv`, `predictions.jsonl`, `report.json`. Script chia tập ghi thêm `manifest.json`; manifest nhắc kiểm tra thủ công quyền sử dụng, câu quảng cáo và tham chiếu lấy từ sapo/lead.

ROUGE-1/2/L F1 tính trên token do `underthesea.word_tokenize(format="text")` tạo ra, lowercase nhưng **giữ dấu**, không bỏ stopword và không dùng English stemming. Cùng một cách tách từ được áp dụng cho Lead-N và NMF. Các điểm F1 nằm trong khoảng 0–1. `sentence_analysis` lưu điểm ở lúc chọn đối với câu được chọn; các câu khác giữ điểm ở vòng đầu.

## Quy tắc dữ liệu và đánh giá

- Không đưa bài báo có bản quyền chưa được phép sử dụng hoặc khóa API vào repo.
- NMF fit riêng trên từng bài; chọn cấu hình bằng tập validation và chỉ đánh giá test sau khi chốt cấu hình.
- Nếu chỉ có dữ liệu nhỏ tự viết, ghi rõ đó là pilot và không suy rộng kết quả.
- Không tự điền số liệu vào `results/metrics.csv` hay `results/predictions.jsonl`.
