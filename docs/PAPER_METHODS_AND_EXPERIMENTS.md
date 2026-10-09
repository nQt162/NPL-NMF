# Các phương pháp NMF theo paper và quy trình đánh giá

Tài liệu này ghi lại các nhánh thuật toán vừa được bổ sung, mức độ tương ứng với các paper trong thư mục dự án và cách đánh giá mà không tạo số liệu giả.

## Phương pháp đã tích hợp

| Nhánh | Objective / chọn câu | Liên hệ paper và giới hạn |
| --- | --- | --- |
| Local KL-NMF + L1 + MMR | KL divergence với L1 trên W/H; MMR chọn câu | Nhánh mặc định hiện có. `alpha_w`, `alpha_h`, `l1_ratio` giờ cấu hình được; mặc định vẫn 0.1, 0.1, 1.0. |
| NMFTS-inspired pairwise | Frobenius reconstruction + symmetric-KL penalty giữa các hàng W, trọng số bởi đồ thị câu; sau đó chọn câu bằng MMR | Bổ sung theo hướng Aghdam và cộng sự. Đồ thị hiện dùng TF-IDF cosine kNN; đây là bản thích nghi/paper-inspired, không tuyên bố tái hiện nguyên trạng semantic representation và toàn bộ thực nghiệm của paper. |
| SNMF đồ thị câu | Phân rã đối xứng `A ≈ G Gᵀ`; xếp hạng câu trong cụm bằng độ tương tự nội cụm và centrality | Bổ sung theo hướng Wang và cộng sự. Đồ thị TF-IDF cosine là thay thế thực dụng cho biểu diễn quan hệ ngữ nghĩa/semantic-role của paper, không tương đương hoàn toàn. |
| Query PRF | Mở rộng query bằng centroid của tối đa ba câu pseudo-relevant, kết hợp query relevance với topic relevance rồi chọn bằng MMR | Bổ sung theo hướng Park và An. Chỉ hỗ trợ trên nhánh Local KL-NMF; query phải có từ trùng vocabulary cục bộ. |

Các nhánh có thể chọn tại `/api/summarize` và màn hình Tóm tắt. Với NMFTS/SNMF, tối đa 500 câu mỗi request vì đồ thị câu được lưu dạng ma trận dày.

## Quy tắc chọn k

Khi không truyền `k`, hệ thống tiếp tục dùng masked KL imputation. Quy tắc này được dùng chung giữa Local KL, NMFTS và SNMF để so sánh các nhánh trên cùng chính sách chọn k; nó không phải tiêu chí tối ưu riêng objective NMFTS hay SNMF. Muốn so sánh có kiểm soát hơn, truyền cùng một `k` cho tất cả phương pháp. Không nên diễn giải điểm imputation là chất lượng tóm tắt.

## Vị trí trên web và tác dụng

Các trang là các mục điều hướng ở sidebar; số trang dưới đây là số thứ tự hiển thị trong app.

| Trang / vùng | Phần mới hiển thị | Người dùng quan sát được gì |
| --- | --- | --- |
| **02 / Tóm tắt văn bản** → form bên trái, menu **Phương pháp** | Local KL-NMF + L1 + MMR, NMFTS-inspired pairwise KL, SNMF đồ thị câu | Chọn nhánh trên cùng input. Panel **Phương pháp đang chọn** bên phải tóm tắt objective và cách chọn câu của nhánh đó. |
| **02 / Tóm tắt văn bản** → form, truy vấn và **Tham số thuật toán** | Query PRF, alpha W/H, L1 ratio, MMR λ, pairwise λ | Query và query weight chỉ hiện với Local KL; pairwise λ chỉ hiện với NMFTS; alpha/L1 chỉ hiện với Local KL. Tác dụng: điều chỉnh đúng tham số của từng phương pháp mà không gán chúng nhầm sang nhánh khác. |
| **02 / Tóm tắt văn bản** → kết quả sau khi chạy | Panel **Objective**, **Ứng viên k**, từ khóa chủ đề và **Phân tích từng câu** | Xem loss breakdown đúng objective, candidate k khi auto-k, câu/từ khóa được chọn, relevance/redundancy và query relevance nếu PRF hoạt động. Với query không khớp, cảnh báo nêu lý do. |
| **03 / Đánh giá** → **Thí nghiệm batch** | Bảng trung bình theo phương pháp và bảng từng bài | Sau khi chạy batch và nhấn **Làm mới dữ liệu**, đối chiếu ROUGE-1/2/L, thời gian, số câu, số bài và fallback cho Lead-N, Global NMF, Local KL, NMFTS, SNMF; Query PRF chỉ xuất hiện khi split có query đầy đủ. |
| **03 / Đánh giá** → **Đánh giá một mẫu** | Chọn method/k, ngân sách câu, MMR/pairwise và query PRF tùy chọn | Xem nhanh ROUGE cùng summary, objective, k và loss của một văn bản/reference cụ thể. Đây là kiểm tra một mẫu, không thay thế batch benchmark. |
| **04 / So sánh NMF** | Global NMF cũ vs Local KL-NMF | Trang này vẫn chỉ so sánh hai luồng Global/Local; NMFTS, SNMF và Query PRF không nằm trong màn hình này, mà được chọn ở trang 02 và benchmark ở trang 03. |

Tuning không có màn hình web riêng: `scripts/tune_params.py` lưu best config và grid vào `results/tuning/tuning.json`/`tuning.csv`. Dùng cấu hình đã chọn khi chạy batch trên test, sau đó trang 03 đọc metrics qua backend API.

Các hiệu quả được mô tả ở đây là **tác dụng dự kiến của thuật toán**, chưa phải kết quả đã chứng minh trên dữ liệu project. Vì các split hiện rỗng, chưa có điểm ROUGE để khẳng định phương pháp nào tốt hơn.

## Benchmark và tuning

`scripts/run_experiment.py` chạy cùng bài và cùng ngân sách câu qua Lead-N, Global NMF cũ, Local KL+L1+MMR, NMFTS pairwise và SNMF. Query PRF chỉ được thêm khi mọi bài trong split đều có `query`. Lỗi từng phương pháp được ghi vào `report.json`, không bị đổi thành ROUGE bằng 0 trong `metrics.csv`.

`scripts/tune_params.py` tìm cấu hình trên validation: k (0 nghĩa tự chọn), số câu, alpha W/H, L1 ratio, MMR lambda, pairwise lambda và trọng số query. Một cấu hình chỉ được xếp hạng khi chạy thành công trên toàn bộ bài validation, để tránh so sánh trên các tập con khác nhau. Không tune trên test.

Từ thư mục `backend`, sau khi đã chuẩn bị dữ liệu:

```powershell
.\.venv\Scripts\python.exe ..\scripts\tune_params.py `
  --input ..\data\splits\val.jsonl `
  --output ..\results\tuning `
  --methods local_kl_mmr nmfts_pairwise snmf

.\.venv\Scripts\python.exe ..\scripts\run_experiment.py `
  --input ..\data\splits\test.jsonl `
  --output ..\results
```

Mỗi dòng JSONL cần có `id`, `article`, `reference_summary`; `query` là trường bắt buộc nếu muốn đưa Query PRF vào batch. Bài viết cần được chia theo bài vào train/validation/test, không chia các câu của cùng bài qua nhiều split. Ghi nguồn và quyền sử dụng dữ liệu trong manifest.

## Đọc kết quả

- `results/metrics.csv`: điểm ROUGE-1/2/L F1, thời gian, số câu và fallback theo bài/phương pháp.
- `results/predictions.jsonl`: summary và chỉ số câu được chọn để kiểm tra định tính.
- `results/report.json`: cấu hình, trung bình theo phương pháp và lỗi phát sinh.
- Web đọc batch metrics qua `GET /api/experiment/metrics`; ROUGE một mẫu được tính qua `POST /api/evaluate`.
- Loss hiển thị trong màn hình tóm tắt là objective của nhánh đang chạy: KL+L1, Frobenius+pairwise symmetric KL hoặc symmetric Frobenius. Không so sánh trực tiếp các trị số loss giữa những objective khác nhau.

## Việc còn cần cho kết luận học thuật

Các file `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl` hiện rỗng và `results/metrics.csv` chưa có số liệu. Vì vậy giao diện chưa hiển thị kết quả thực nghiệm; cần tập tham chiếu hợp pháp trước khi kết luận phương pháp nào tốt hơn.

Các bước tiếp theo có giá trị cao:

1. Chuẩn bị tập dữ liệu tiếng Việt có reference summary, ghi rõ nguồn/quyền và split theo bài; dùng validation để tuning, chỉ mở test một lần cho báo cáo cuối.
2. Thêm đánh giá định tính mù bởi người đọc (coverage, coherence, redundancy, factuality) bên cạnh ROUGE; với dữ liệu đủ lớn, bổ sung BERTScore và khoảng tin cậy/bootstrap theo bài.
3. Làm ablation: KL vs Frobenius; có/không L1; có/không pairwise; MMR vs xếp hạng relevance; query PRF bật/tắt. Giữ nguyên corpus, k, số câu và seed trong mỗi cặp so sánh.
4. Nếu muốn tuyên bố tái hiện paper, thay đồ thị TF-IDF cosine bằng đúng semantic-role/semantic similarity được mô tả trong paper, khớp tiền xử lý và siêu tham số, rồi đánh giá trên dataset paper dùng.

## Mã nguồn liên quan

- `backend/app/core/paper_methods.py`: đồ thị câu, objective/updates NMFTS-inspired và symmetric NMF.
- `backend/app/core/summarize.py`: ba nhánh tóm tắt, PRF, loss breakdown và metadata.
- `backend/app/schemas.py`, `backend/app/main.py`: request/response API và batch metrics endpoint.
- `scripts/run_experiment.py`: so sánh baseline và các nhánh NMF.
- `scripts/tune_params.py`: tìm cấu hình chỉ trên validation.
- `frontend/src/pages/Summarizer.jsx`: chọn thuật toán, query và tham số.
- `frontend/src/pages/Evaluation.jsx`: ROUGE một mẫu và kết quả batch.
