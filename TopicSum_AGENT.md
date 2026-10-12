# TopicSum – đặc tả ban đầu và hướng dẫn coding agent

> **Trạng thái hiện tại:** Tài liệu này bắt đầu như bản đặc tả/roadmap ban đầu; một số mô tả thuật toán cũ bên dưới không còn là trạng thái code. Hệ thống hiện dùng scikit-learn NMF cho tóm tắt/đánh giá, còn `/api/simulate` dùng MU tự cài đặt để minh họa. Hai loss trong luồng tóm tắt dùng cùng L1; chi tiết thực tế nằm trong [docs/NMF_IMPROVEMENTS.md](docs/NMF_IMPROVEMENTS.md) và [docs/PAPER_METHODS_AND_EXPERIMENTS.md](docs/PAPER_METHODS_AND_EXPERIMENTS.md).

## 0. Mục tiêu và phạm vi
Xây ứng dụng web tóm tắt trích xuất một văn bản tiếng Việt bằng phân tích chủ đề NMF. Ba phần gồm: (1) minh họa phân rã theo vòng lặp, (2) tóm tắt câu theo chủ đề, (3) đánh giá với bản tóm tắt tham chiếu. Không dùng LLM tạo bản tóm tắt; giữ nguyên câu gốc. Kiến trúc hiện tại: React + Vite, FastAPI, NumPy/scikit-learn, underthesea và ROUGE.

## 1. Dữ liệu và quy tắc chia
- `data/demo/toy.json`: 1 văn bản tự viết, 5–8 câu, hai chủ đề rõ; dùng cho mô phỏng dễ nhìn. Có thể thêm ma trận V tự định nghĩa 4×5 để kiểm tra phép nhân bằng tay.
- `data/raw/articles.jsonl`: khoảng 100–300 bản tin tiếng Việt có quyền sử dụng, mỗi dòng `{ "id": "001", "article": "...", "reference_summary": "...", "source": "..." }`. Đơn vị mẫu là **một bài**, không tách câu bài này sang tập khác. Dữ liệu mô phỏng tách riêng tập đánh giá.
- `data/splits/{train,val,test}.jsonl`: chia cố định theo bài 60/20/20, seed=42, ghi manifest nguồn và license. `train` chỉ để khảo sát/thiết kế; NMF được fit **riêng trên từng tài liệu** lúc suy luận, không học tham số chung trên train. Chọn k, tỉ lệ tóm tắt, hệ số đa dạng trên val; test chỉ chạy một lần sau khi chốt cấu hình.
- Bài có ít nhất 6 câu đủ nội dung; lọc trùng bài, câu quảng cáo, bản tin thiếu reference. Không lấy chính đoạn mở đầu của bài làm tham chiếu nếu muốn đánh giá độc lập với Lead-3. Ghi rõ nếu tham chiếu vốn là sapo/lead vì sẽ thiên lệch về Lead-3.
- Không bịa kết quả. Nếu chưa có dữ liệu hợp lệ, dùng 10–20 bài tự viết hoặc được phép dùng để chạy demo, đánh dấu kết quả pilot; không kết luận tổng quát.

Ví dụ một dòng JSONL (minh họa, không phải dữ liệu đánh giá):
```json
{"id":"demo-01","article":"Thành phố mở thêm tuyến xe buýt điện vào sáng thứ Hai. Tuyến mới kết nối ga trung tâm với bệnh viện và hai trường đại học. Đơn vị vận hành bố trí mười xe, mỗi xe có sức chứa bốn mươi người. Giá vé được giữ bằng tuyến buýt thường. Người dân có thể tra cứu lịch chạy trên ứng dụng của thành phố. Tháng tới, đơn vị vận hành sẽ khảo sát mức độ hài lòng của hành khách.","reference_summary":"Thành phố mở tuyến xe buýt điện nối ga trung tâm với bệnh viện và hai trường đại học; giá vé bằng tuyến thường.","source":"synthetic"}
```

## 2. Cây dự án
```text
topicsum/
├── README.md
├── .gitignore
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py                 # FastAPI, CORS cho localhost
│   │   ├── schemas.py              # request/response Pydantic
│   │   └── core/
│   │       ├── sentences.py        # tách câu; giữ nguyên text, index
│   │       ├── preprocess.py       # word tokenize, chuẩn hóa cho TF-IDF
│   │       ├── vectorize.py        # TF-IDF câu × từ
│   │       ├── nmf.py              # multiplicative updates + snapshots
│   │       ├── summarize.py        # chấm điểm, greedy coverage/diversity
│   │       ├── baselines.py        # Lead-N; TextRank chỉ nếu còn thời gian
│   │       └── evaluate.py         # ROUGE và thống kê
│   └── tests/
│       ├── test_nmf.py
│       └── test_summary.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── pages/
│           ├── Visualizer.jsx
│           ├── Summarizer.jsx
│           └── Evaluation.jsx
├── data/
│   ├── demo/toy.json
│   ├── raw/articles.jsonl
│   └── splits/
├── scripts/
│   ├── prepare_data.py
│   └── run_experiment.py
└── results/
    ├── metrics.csv
    └── predictions.jsonl
```

## 3. Thuật toán dùng chung
Với một tài liệu: tách câu gốc `s_i`, tạo bản tiền xử lý `p_i`, TF-IDF `V` kích thước `n_sentences × n_terms`, `V >= 0`. `TfidfVectorizer` nhận các câu đã được tách từ bằng dấu cách; dùng `tokenizer=str.split`, `preprocessor=None`, `token_pattern=None`, `lowercase=False`, `norm='l2'`. Tiền xử lý tiếng Việt: lowercase, underthesea `word_tokenize(..., format='text')`, stopword list nhỏ lưu trong repo; không làm biến đổi câu gốc xuất ra. Không dùng `min_df=2` cho tài liệu ngắn; nếu V rỗng, fallback Lead-N và báo lý do.

Phần mô tả MU bên dưới chỉ áp dụng cho **trang mô phỏng**. Tóm tắt chạy scikit-learn NMF cục bộ; cả hai loss dùng cùng `alpha_W`, `alpha_H`, `l1_ratio` và seed. Khởi tạo sản xuất là `nndsvdar`, vì vậy seed được truyền vào thuật toán và có thể làm thay đổi khởi tạo.

Trong mô phỏng, `k = min(requested_k, n_sentences, n_terms)`; nếu không nhập k, API dùng chung bộ chọn positive-entry masked-KL với trang tóm tắt. Bộ chọn thử k khả thi từ 1 đến 10, dùng 10 lượt giữ lại ngẫu nhiên 20% TF-IDF dương và tính KL trên vị trí giữ lại; dữ liệu quá ít trả heuristic không kèm điểm. `W: n_sentences × k`, `H: k × n_terms`. Mô phỏng có thể hiển thị Frobenius hoặc KL, cùng L1 penalty:
```text
H <- H * (W.T @ V) / (W.T @ W @ H + alpha_H * n_sentences + eps)
W <- W * (V @ H.T) / (W @ H @ H.T + alpha_W * n_terms + eps)
loss = 0.5 * ||V - W @ H||_F^2 + alpha_W * n_terms * ||W||_1 + alpha_H * n_sentences * ||H||_1
```
`eps=1e-9`, tối đa 20 vòng trên API mô phỏng. `fit_nmf(V,k,seed,max_iter,trace=False)` trả W,H, loss và snapshots. Trace này là thuật toán MU tự cài đặt, không phải trace của estimator scikit-learn ở luồng tóm tắt; không so trực tiếp từng hệ số W/H giữa hai cách cài đặt.

Từ khóa topic được xếp hạng theo trọng số H đã chuẩn hóa nhân với độ độc quyền của từ giữa các topic; mỗi topic còn hiển thị hai câu đại diện lấy từ W. Topic là latent, không có nhãn ngữ nghĩa tự động; người đọc diễn giải qua từ khóa và câu đại diện. `W`/`H` không mặc nhiên là xác suất hoặc tỉ lệ phần trăm.

Chọn câu dùng greedy Maximum Marginal Relevance (MMR). Tại mỗi vòng, với mỗi câu chưa chọn `i`:
```text
relevance_i = sum_t W[i,t] * salience[t] / (sum_t salience[t] + eps)
coverage_gain_i = sum_t salience[t] * max(0, W[i,t] - coverage[t]) / (sum_t salience[t] + eps)
redundancy_i = max(cosine(V[i], V[j]) for j in selected), or 0 for the first pick
MMR_i = lambda * relevance_i - (1 - lambda) * redundancy_i
```
Relevance lấy từ phân phối topic của câu và độ phổ biến topic; redundancy là cosine TF-IDF lớn nhất với các câu đã chọn. Chọn đúng ngân sách nếu đủ câu, rồi sắp xếp lại theo thứ tự gốc trước khi ghép. Điểm MMR chỉ so sánh trong cùng vòng vì redundancy đổi sau mỗi câu được chọn; API trả `selection_trace` theo vòng.

## 4. API
- `GET /health` → `{ "status": "ok" }`.
- `POST /api/simulate` body `{ "text": "...", "k": 2, "iterations": 10, "seed": 42 }`; response gồm `sentences`, `terms`, `V`, `snapshots:[{iteration,W,H,WH,loss}]`, metadata về loss/L1 và `implementation="educational_custom_mu"`. Dùng `fit_nmf(trace=True)`.
- `POST /api/summarize` nhận `text`, `k`, `summary_sentences`, `mmr_lambda`, `seed`, `method`, `alpha_w`, `alpha_h`, `l1_ratio`, cùng query tùy chọn trên nhánh KL. Hai method là `local_kl_mmr` và `local_frobenius_mmr`; cùng L1/MMR/k, chỉ khác `beta_loss`. Response có objective breakdown, candidate k, từ khóa/câu đại diện topic, phân tích câu theo `score_round` và `selection_trace`.
- `POST /api/evaluate` chỉ dùng cho một mẫu demo `{ "text":"...", "reference_summary":"...", "summary_sentences":3 }`; trả ROUGE-1/2/L F1 và summary. Batch đánh giá chạy từ CLI, không qua API.
- Validation Pydantic: giới hạn kích thước văn bản ~20.000 ký tự, k 1..10, số câu 1..5, iterations 1..20 cho simulate; 422 cho input thiếu/hỏng. Frontend hiển thị lỗi.

## 5. Ba màn hình
1. **Visualizer**: nạp ví dụ nhỏ, hiện V, W, H, WH, loss; nút trước/sau hoặc slider từ bước 0 đến N. Các số lấy từ API thực, không hardcode.
2. **Summarizer**: nhập văn bản; chọn k và số câu; xem bản tóm tắt, top terms của topic, câu đã chọn và thành phần điểm.
3. **Evaluation**: nhập văn bản nguồn và reference summary; API tạo KL+L1+MMR và Frobenius+L1+MMR trên cùng cấu hình rồi hiển thị ROUGE-1/2/L F1. Đây không phải benchmark batch; benchmark Lead-N/Global NMF chạy riêng bằng CLI.

## 6. Đánh giá và đối chứng
`run_experiment.py --input data/splits/test.jsonl --output results/` chạy Lead-N và NMF trên cùng bài và cùng số câu. Tính ROUGE-1/2/L F1 trên câu gốc và reference, báo cáo mean trên bài hợp lệ, thời gian trung bình, số bài lỗi/fallback. Dùng cách tokenize ROUGE hỗ trợ tiếng Việt thống nhất cho mọi mô hình (tách từ trước khi tính metric, không bỏ dấu); ghi lựa chọn đó trong README. Xuất `metrics.csv` theo từng bài và phương pháp, `predictions.jsonl`; seed=42, config lưu k, ngân sách, alpha/beta/gamma. Báo cáo 3 ca tốt/3 ca thất bại và ảnh hưởng k=1,2,3 trên val; không tune bằng test. Lead-N lấy đúng N câu đầu. TextRank chỉ thêm sau khi ba phần chính chạy ổn.

## 7. Tiến độ 14 ngày / tiêu chí hoàn thành
- Ngày 1–2: dữ liệu demo, schema JSONL, tách câu và TF-IDF; chốt 20–30 bài pilot có reference.
- Ngày 3–4: NMF tự cài đặt + kiểm tra loss, trace, lỗi đầu vào.
- Ngày 5–6: chọn câu, Lead-N, API summarize/simulate.
- Ngày 7–9: frontend 3 màn hình, gọi API thật, xử lý trường hợp lỗi.
- Ngày 10–11: hoàn thiện dữ liệu thử nghiệm, chia cố định, chạy ROUGE.
- Ngày 12–13: phân tích ca lỗi, điều chỉnh trên val, viết README/báo cáo.
- Ngày 14: chạy lại toàn bộ và tập dượt demo.

Definition of done ban đầu: mô phỏng có V≈WH và loss thực; ứng dụng trả về câu nguyên văn; phép đánh giá dùng ROUGE trên test độc lập. Kiến trúc hiện hành không dùng cùng implementation cho mô phỏng và tóm tắt. Khi chưa có dataset đầy đủ, chỉ báo cáo pilot và không khẳng định hiệu quả tổng quát.

## 8. Lệnh chạy dự kiến
```bash
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -e . && uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
cd backend && python ../scripts/run_experiment.py --input ../data/splits/test.jsonl --output ../results
cd backend && pytest
```
Agent cần cập nhật các lệnh nếu packaging thực tế khác; README là nguồn sự thật cho cách chạy. Không đưa dữ liệu bản quyền hoặc khóa API vào repo. Lưu file dữ liệu thử nghiệm kèm nguồn rõ ràng.
