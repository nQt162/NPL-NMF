# TopicSum — đặc tả triển khai cho coding agent

## 0. Mục tiêu và phạm vi
Xây ứng dụng web tóm tắt trích xuất một văn bản tiếng Việt bằng phân tích chủ đề NMF, trong 14 ngày. Ba phần dùng **cùng một hàm NMF tự cài đặt**: (1) mô phỏng từng vòng lặp trên ví dụ nhỏ, (2) tóm tắt câu theo chủ đề, (3) thử nghiệm với bản tóm tắt tham chiếu. Không dùng LLM tạo bản tóm tắt; giữ nguyên câu gốc. Không xây đăng nhập, cơ sở dữ liệu, PDF, triển khai cloud trong MVP. Web đơn giản: React + Vite, FastAPI, NumPy/scikit-learn, underthesea, rouge-score. Nếu nguồn lực ít, ưu tiên hoàn thiện backend và ba màn hình tối thiểu.

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

`k = min(requested_k, n_sentences, n_terms)`, mặc định 2; nếu < 2 câu/từ hữu ích, fallback Lead-N. NMF khởi tạo W,H dương bằng `numpy.random.default_rng(seed)`; `W: n_sentences × k`, `H: k × n_terms`. Dùng cập nhật nhân và Frobenius loss:
```text
H <- H * (W.T @ V) / (W.T @ W @ H + eps)
W <- W * (V @ H.T) / (W @ H @ H.T + eps)
loss = 0.5 * sum((V - W @ H)^2)
```
`eps=1e-9`, tối đa 100 vòng, dừng nếu mức giảm tương đối < 1e-5 sau 5 vòng; không giữ snapshot cho tất cả bài đánh giá. `fit_nmf(V,k,seed,max_iter,trace=False)` trả về W,H, losses và optional snapshots (bản sao của W,H,WH mỗi vòng). Snapshot 0 là trạng thái khởi tạo; chụp sau khi cập nhật cả H và W. Chỉ trace với ma trận nhỏ hoặc tối đa 20 vòng để tránh payload lớn. Loss nên không tăng đáng kể ngoài sai số số học; kiểm tra nonnegative, shapes, reproducibility. NMF có thể khác `sklearn` do khởi tạo và solver khác; so sánh reconstruction error, không so trực tiếp từng hệ số W/H.

Nhãn topic là top 5 từ của từng hàng H (nhãn mô tả, không coi là nhãn thật). `W` được chuẩn hóa theo tổng từng hàng để hiển thị mức liên quan của câu; topic salience từ tổng W theo cột. Tránh ngộ nhận giá trị W và H là xác suất hoặc tỉ lệ phần trăm nếu chưa chuẩn hóa.

MVP chọn câu theo greedy topic coverage: tại mỗi bước, với câu chưa chọn `i`:
```text
relevance_i = sum_t W[i,t] * salience[t] / (sum_t salience[t] + eps)
coverage_gain_i = sum_t salience[t] * max(0, W[i,t] - coverage[t]) / (sum_t salience[t] + eps)
redundancy_i = max(cosine(V[i], V[j]) for j in selected), hoặc 0 nếu chưa chọn
score_i = alpha*relevance_i + beta*coverage_gain_i - gamma*redundancy_i
```
Trước khi tính, chuẩn hóa mỗi hàng W về tổng 1; salience tính từ **W gốc**, chuẩn hóa tổng 1. Mỗi lần chọn, cập nhật `coverage[t] = max(coverage[t], W_normalized[i,t])`. Mặc định alpha=1, beta=1, gamma=0.5; tinh chỉnh trên val. Ngân sách `summary_sentences` mặc định 3, giới hạn 1..min(5,n_sentences); luôn cùng ngân sách cho baseline. Nếu tất cả điểm thấp vẫn lấy đủ ngân sách. Tie-break theo index; sắp xếp các câu được chọn về thứ tự gốc trước khi nối. Trả về index, câu gốc, điểm từng thành phần và topic trội.

## 4. API
- `GET /health` → `{ "status": "ok" }`.
- `POST /api/simulate` body `{ "text": "...", "k": 2, "iterations": 10, "seed": 42 }`; chỉ nhận tối đa 10 câu / 50 từ vựng; response gồm `sentences`, `terms`, `V`, `snapshots:[{iteration,W,H,WH,loss}]`. Ma trận lớn trả 422 với thông báo rõ. Dùng `fit_nmf(trace=True)`.
- `POST /api/summarize` body `{ "text": "...", "k": 2, "summary_sentences": 3, "alpha": 1, "beta": 1, "gamma": 0.5 }`; response `{ "summary": "...", "selected_indices": [0,2,5], "topics": [{"top_terms": [...] }], "sentence_analysis": [{"index":0,"text":"...","selected":true,"relevance":0.0,"coverage_gain":0.0,"redundancy":0.0,"score":0.0,"dominant_topic":0}], "fallback_reason": null }`. `sentence_analysis` là điểm tại lúc xét/chọn hoặc điểm đầu vòng được mô tả rõ trong code; không giả định score cố định khi đã có chọn câu.
- `POST /api/evaluate` chỉ dùng cho một mẫu demo `{ "text":"...", "reference_summary":"...", "summary_sentences":3 }`; trả ROUGE-1/2/L F1 và summary. Batch đánh giá chạy từ CLI, không qua API.
- Validation Pydantic: giới hạn kích thước văn bản ~20.000 ký tự, k 1..10, số câu 1..5, iterations 1..20 cho simulate; 422 cho input thiếu/hỏng. Frontend hiển thị lỗi.

## 5. Ba màn hình
1. **Visualizer**: nạp ví dụ nhỏ, hiện V, W, H, WH, loss; nút trước/sau hoặc slider từ bước 0 đến N. Các số lấy từ API thực, không hardcode.
2. **Summarizer**: nhập văn bản; chọn k và số câu; xem bản tóm tắt, top terms của topic, câu đã chọn và thành phần điểm.
3. **Evaluation**: đọc `results/metrics.csv` có sẵn hoặc chạy một mẫu qua API; bảng Lead-N vs NMF, ROUGE-1/2/L F1, runtime, số câu. Không cần chức năng upload hay tài khoản.

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

Definition of done: mô phỏng tiến từng vòng với V≈WH và loss thực; ứng dụng trả về câu nguyên văn; cùng `fit_nmf` được gọi ở hai phần; thí nghiệm có Lead-N, NMF và ROUGE trên test độc lập; README có lệnh cài đặt/chạy rõ ràng; test trọng yếu kiểm tra shape/nonnegative, loss hội tụ, thứ tự câu và giới hạn ngân sách. Khi chưa có dataset đầy đủ, báo cáo rõ số mẫu pilot.

## 8. Lệnh chạy dự kiến
```bash
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -e . && uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
cd backend && python ../scripts/run_experiment.py --input ../data/splits/test.jsonl --output ../results
cd backend && pytest
```
Agent cần cập nhật các lệnh nếu packaging thực tế khác; README là nguồn sự thật cho cách chạy. Không đưa dữ liệu bản quyền hoặc khóa API vào repo. Lưu file dữ liệu thử nghiệm kèm nguồn rõ ràng.
