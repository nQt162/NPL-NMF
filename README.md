# TopicSum (NPL-NMF)

L1 regularization ablation workflow: [docs/ABLATION_L1.md](docs/ABLATION_L1.md) (web page 04 and `scripts/run_ablation.py`).

Bộ khung dự án cho ứng dụng tóm tắt **trích xuất** văn bản tiếng Việt bằng NMF. Yêu cầu chi tiết và tiêu chí hoàn thành nằm trong [TopicSum_AGENT.md](TopicSum_AGENT.md).

## Trạng thái hiện tại

Backend đã có xử lý tiếng Việt, NMF tự cài đặt, API, script chuẩn bị dữ liệu/thí nghiệm và kiểm thử. Frontend React/Vite có bốn màn hình dùng API thật. `results/metrics.csv` và `results/predictions.jsonl` hiện rỗng vì chưa có tập đánh giá hợp lệ.

## Cấu trúc và việc cần làm

| Khu vực | Việc chính |
| --- | --- |
| `backend/app/core/` | Tiền xử lý, TF-IDF, masked-KL chọn k, NMF scikit-learn cho tóm tắt, MU tự cài đặt cho mô phỏng, MMR và ROUGE. |
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

## Cài đặt và chạy frontend (PowerShell)

Cần Node.js 20.19+ hoặc 22.12+. Chạy backend ở cổng 8000 trước, sau đó mở terminal khác:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Mở `http://localhost:5173`. Vite chuyển `/api` và `/health` đến `http://127.0.0.1:8000`. Nếu frontend được phục vụ riêng, đặt `VITE_API_URL` thành địa chỉ backend trước khi build và cấu hình CORS backend cho origin tương ứng. Kiểm tra frontend bằng `npm.cmd test` và `npm.cmd run build`.

- **Mô phỏng NMF:** nạp văn bản tự viết từ `data/demo/toy.json`, gọi `/api/simulate` và hiển thị V, W, H, WH, loss ở từng vòng. Đây là MU tự cài đặt để minh họa, không phải trace nội bộ của scikit-learn dùng ở trang tóm tắt.
- **Tóm tắt:** gọi `/api/summarize`, hiển thị câu gốc, chủ đề, điểm thành phần và lý do fallback.
- **Đánh giá:** nhập văn bản nguồn và bản tham chiếu; trang tạo hai bản tóm tắt với KL+L1+MMR và Frobenius+L1+MMR, dùng chung k, alpha/L1, seed, ngân sách câu và MMR rồi so sánh ROUGE-1/2/L qua `/api/evaluate`.

Hai nhánh sản xuất đều dùng `sklearn.decomposition.NMF` với `solver="mu"`, `init="nndsvdar"`, và cùng penalty mặc định `alpha_W=alpha_H=0.001`, `l1_ratio=1.0`; chỉ khác `beta_loss`. Mức 0.001 là mặc định bảo thủ để tránh penalty làm factor sụp về 0 trên TF-IDF cục bộ; đây chưa phải giá trị tối ưu thực nghiệm vì split dữ liệu hiện rỗng. Khi không nhập k, hệ thống thử các k khả thi từ 1 đến 10 bằng 10 lượt giữ lại ngẫu nhiên các phần tử TF-IDF dương, chấm KL trên phần tử bị giữ lại. Nếu dữ liệu dương quá ít, API trả nhãn heuristic và không bịa điểm số. Điểm k là reconstruction validation, không phải ROUGE/chất lượng tóm tắt.

Lưu ý: giá trị objective KL và Frobenius không cùng thang đo nên không so sánh trực tiếp. Dùng ROUGE với cùng reference để so đầu ra; split dữ liệu trong repo hiện rỗng nên chưa có kết luận thực nghiệm tổng quát.

## Chuẩn bị dữ liệu và chạy thí nghiệm

Điền `data/raw/articles.jsonl` bằng các bài có quyền sử dụng; mỗi dòng cần `id`, `article`, `reference_summary`, `source`, nên có thêm `license`. Từ thư mục `backend` đã cài package:

```powershell
.\.venv\Scripts\python.exe ..\scripts\prepare_data.py --input ..\data\raw\articles.jsonl --output ..\data\splits
.\.venv\Scripts\python.exe ..\scripts\run_experiment.py --input ..\data\splits\val.jsonl --output ..\results
```

Chọn `k`, ngân sách, `--alpha-w`, `--alpha-h`, `--l1-ratio` và `--mmr-lambda` trên validation. Sau khi chốt cấu hình, chạy test đúng một lần bằng `--input ..\data\splits\test.jsonl --output ..\results`. Kết quả gồm `metrics.csv`, `predictions.jsonl`, `report.json`. Script chia tập ghi thêm `manifest.json`; manifest nhắc kiểm tra thủ công quyền sử dụng, câu quảng cáo và tham chiếu lấy từ sapo/lead.

ROUGE-1/2/L F1 tính trên token do `underthesea.word_tokenize(format="text")` tạo ra, lowercase nhưng **giữ dấu**, không bỏ stopword và không dùng English stemming. Cùng một cách tách từ được áp dụng cho Lead-N và NMF. Các điểm F1 nằm trong khoảng 0–1. `sentence_analysis` lưu điểm ở lúc chọn đối với câu được chọn; các câu khác giữ điểm ở vòng đầu.

## Quy tắc dữ liệu và đánh giá

- Không đưa bài báo có bản quyền chưa được phép sử dụng hoặc khóa API vào repo.
- NMF fit riêng trên từng bài; chọn cấu hình bằng tập validation và chỉ đánh giá test sau khi chốt cấu hình.
- Nếu chỉ có dữ liệu nhỏ tự viết, ghi rõ đó là pilot và không suy rộng kết quả.
- Không tự điền số liệu vào `results/metrics.csv` hay `results/predictions.jsonl`.
