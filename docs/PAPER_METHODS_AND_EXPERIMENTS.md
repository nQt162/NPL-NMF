# Các phương pháp NMF theo paper và quy trình đánh giá

Tài liệu này ghi lại các nhánh thuật toán vừa được bổ sung, mức độ tương ứng với các paper trong thư mục dự án và cách đánh giá mà không tạo số liệu giả.

**Trạng thái hiện tại của web:** so sánh Local KL-NMF + L1 + MMR và Local Frobenius-NMF + L1 + MMR. Cùng alpha/L1, k, seed, ngân sách câu và MMR; thay đổi chính là hàm mất mát. Trang mô phỏng dùng MU tự cài đặt để minh họa, còn tóm tắt chạy scikit-learn. NMFTS pairwise và SNMF đồ thị câu đã được rút khỏi API/giao diện. Query PRF là tùy chọn trên cả KL và Frobenius ở trang tóm tắt; đánh giá và ablation vẫn tắt query.

## Phương pháp đã tích hợp

**Cập nhật 15-10-2026:** web dùng `test_nmf_01` gồm 12 câu AI, reference trích xuất S1/S5/S8/S10 và mặc định 4 câu trên trang 02/03. Đây là nội dung minh họa do người dùng cung cấp, không phải dataset của paper hay benchmark độc lập.

| Nhánh | Objective / chọn câu | Liên hệ paper và giới hạn |
| --- | --- | --- |
| Local KL-NMF + L1 + MMR | KL divergence với L1 trên W/H; MMR chọn câu | Phương pháp đang hoạt động. `alpha_w`, `alpha_h`, `l1_ratio` cấu hình được; mặc định 0.001, 0.001, 1.0 (chưa tối ưu trên corpus). |
| Local Frobenius-NMF + L1 + MMR | Nửa bình phương sai số Frobenius + cùng L1 trên W/H; MMR chọn câu | Phương pháp đang hoạt động; đối chiếu loss dưới penalty và chính sách trích câu như nhau. |
| NMFTS-inspired pairwise | Frobenius reconstruction + symmetric-KL penalty giữa các hàng W, trọng số bởi đồ thị câu; sau đó chọn câu bằng MMR | Nhánh nghiên cứu cũ, đã rút khỏi API/web. Trước đây là bản paper-inspired theo Aghdam và cộng sự, dùng TF-IDF cosine kNN thay cho toàn bộ semantic representation của paper. |
| SNMF đồ thị câu | Phân rã đối xứng `A ≈ G Gᵀ`; xếp hạng câu trong cụm bằng độ tương tự nội cụm và centrality | Nhánh nghiên cứu cũ, đã rút khỏi API/web. Trước đây là bản paper-inspired theo Wang và cộng sự, không tương đương hoàn toàn biểu diễn semantic-role của paper. |
| Query PRF | Mở rộng query bằng centroid của tối đa ba câu pseudo-relevant, kết hợp query relevance với topic relevance rồi chọn bằng MMR | Bổ sung theo hướng Park và An, hỗ trợ cả Local KL-NMF và Local Frobenius-NMF; query phải có từ trùng vocabulary cục bộ. |

Hai method đang được `/api/summarize` và giao diện chấp nhận là Local KL+L1 và Local Frobenius+L1. Cả hai dùng scikit-learn `NMF`, `solver="mu"`, `init="nndsvdar"`, cùng penalty; `beta_loss` lần lượt là `kullback-leibler` và `frobenius`. NMFTS/SNMF không còn được expose qua API/web.

## Quy tắc chọn k

Khi không truyền `k`, cùng bộ chọn chạy masked positive-entry KL validation cho mọi trang/method: thử `k=1..min(10,n_sentences,n_terms)`, 10 lần giữ lại 20% phần tử TF-IDF dương, rồi chọn k có KL validation thấp nhất. Fit chọn k không áp L1 để giữ tiêu chí cố định. Nếu số phần tử dương quá ít, trả heuristic với score rỗng; không giả score bằng 0. Đây là tiêu chí khôi phục TF-IDF, không phải ROUGE/chất lượng tóm tắt hay tối ưu riêng Frobenius.

## Vị trí trên web và tác dụng

Các trang là các mục điều hướng ở sidebar; số trang dưới đây là số thứ tự hiển thị trong app.

| Trang / vùng | Phần mới hiển thị | Người dùng quan sát được gì |
| --- | --- | --- |
| **01 / Mô phỏng NMF** → menu phương pháp | Local KL-NMF + L1 hoặc Local Frobenius-NMF + L1 | Quan sát snapshot từ MU tự cài đặt, không phải trace sklearn; chọn k, số vòng lặp và seed. Loss khác thang đo nên không so trực tiếp giữa hai kiểu. |
| **02 / Tóm tắt văn bản** → form bên trái, menu **Phương pháp** | Local KL-NMF + L1 + MMR hoặc Local Frobenius-NMF + L1 + MMR | Hai method dùng cùng L1, MMR, chính sách k và thứ tự câu gốc; khác loss. |
| **02 / Tóm tắt văn bản** → form, truy vấn và **Tham số thuật toán** | Query PRF, alpha W/H, L1 ratio và MMR λ | Alpha/L1, MMR và PRF dùng chung hai nhánh. Query giữ nguyên khi đổi phương pháp. Không còn pairwise λ vì NMFTS đã rút khỏi web. |
| **02 / Tóm tắt văn bản** → kết quả sau khi chạy | Panel **Objective**, **Ứng viên k**, từ khóa đặc trưng/câu đại diện và **Phân tích từng câu** | Xem objective breakdown, k validation/heuristic, topic latent và candidate MMR theo từng vòng. Chỉ so MMR score trong cùng vòng; PRF relevance hiện khi hoạt động. |
| **03 / Đánh giá** | Văn bản nguồn, bản tham chiếu, k/ngân sách câu, seed và tham số chung | Sinh hai summary với KL+L1+MMR và Frobenius+L1+MMR trên cùng cấu hình; báo ROUGE-1/2/L. |
| **04 / Ablation L1** | KL-NMF không L1, L1 trên W, L1 trên H và L1 trên cả W/H | Web nhập một bài viết/reference trực tiếp (pilot); CLI nhận JSONL. So sánh ghép cặp ROUGE và thống kê W/H dưới cùng seed/k/MMR. |

Tuning không có màn hình web riêng: `scripts/tune_params.py` lưu best config/grid vào `results/tuning/tuning.json`/`tuning.csv`. Dùng cấu hình đã chọn khi chạy batch test và đọc artifact riêng; trang 03 chỉ sinh/chấm hai outputs của một bài, không đọc bảng batch.

Các hiệu quả được mô tả ở đây là **tác dụng dự kiến của thuật toán**, chưa phải kết quả đã chứng minh trên dữ liệu project. Vì các split hiện rỗng, chưa có điểm ROUGE để khẳng định phương pháp nào tốt hơn.

## Benchmark và tuning

`scripts/run_experiment.py` là công cụ benchmark batch ngoại tuyến, tách biệt với trang **03 / Đánh giá**. Trang web sinh hai summary từ cùng văn bản đầu vào, dùng chung k, số câu, seed và MMR rồi chấm ROUGE với cùng reference; benchmark batch không được hiển thị trên trang.

`scripts/tune_params.py` tìm cấu hình cho hai phương pháp: k (0 nghĩa tự chọn), số câu, alpha W/H, L1 ratio và MMR lambda. Cùng grid penalty được dùng cho cả hai loss để cô lập tác động hàm mất mát. Một cấu hình chỉ được xếp hạng khi chạy thành công trên toàn bộ bài validation; không tune trên test.

Từ thư mục `backend`, sau khi đã chuẩn bị dữ liệu:

```powershell
.\.venv\Scripts\python.exe ..\scripts\tune_params.py `
  --input ..\data\splits\val.jsonl `
  --output ..\results\tuning `
  --methods local_kl_mmr local_frobenius_mmr

.\.venv\Scripts\python.exe ..\scripts\run_experiment.py `
  --input ..\data\splits\test.jsonl `
  --output ..\results
```

Mỗi dòng JSONL cần có `id`, `article`, `reference_summary`. Bài viết cần được chia theo bài vào train/validation/test, không chia các câu của cùng bài qua nhiều split. Ghi nguồn và quyền sử dụng dữ liệu trong manifest.

## Đọc kết quả

- `results/metrics.csv`: điểm ROUGE-1/2/L F1, thời gian, số câu và fallback theo bài/phương pháp.
- `results/predictions.jsonl`: summary và chỉ số câu được chọn để kiểm tra định tính.
- `results/report.json`: cấu hình, trung bình theo phương pháp và lỗi phát sinh.
- Backend có `GET /api/experiment/metrics` để đọc artifact ngoại tuyến; giao diện hiện tại không hiển thị batch. ROUGE một mẫu trên trang 03 dùng `POST /api/evaluate`.
- Loss trong mô phỏng là objective của custom MU; tóm tắt là objective sklearn. Cả hai nhánh dùng KL+L1 hoặc Frobenius+L1. Không so trị số loss trực tiếp giữa hai objective/implementation; dùng ROUGE trên cùng reference.

## Việc còn cần cho kết luận học thuật

Các file `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl` và `results/metrics.csv` còn rỗng ngày kiểm tra. Trang 03 vẫn có thể hiển thị ROUGE của bài mẫu, trang 04 vẫn chạy pilot; điều chưa có là benchmark corpus, không phải mọi kết quả trên giao diện.

Với mẫu AI, auto-k=1/seed 42/ngân sách 4/alpha 0.001/lambda 0.7, KL đạt R1/R2/L **0.3897/0.2694/0.3282**, Frobenius **0.3979/0.2646/0.3246**. Một nhánh nhỉnh R1, nhánh kia nhỉnh R2/L, không có kết luận thắng tuyệt đối. k=1 là kết quả masked positive-entry KL CV, không xác nhận bài chỉ có một chủ đề. Xem [kịch bản](DEMO_PRESENTATION_SCRIPT.md) và [cẩm nang](NMF_SYSTEM_GUIDE.md#chuong-17).

Các bước tiếp theo có giá trị cao:

1. Chuẩn bị tập dữ liệu tiếng Việt có reference summary, ghi rõ nguồn/quyền và split theo bài; dùng validation để tuning, chỉ mở test một lần cho báo cáo cuối.
2. Thêm đánh giá định tính mù bởi người đọc (coverage, coherence, redundancy, factuality) bên cạnh ROUGE; với dữ liệu đủ lớn, bổ sung BERTScore và khoảng tin cậy/bootstrap theo bài.
3. Làm ablation: KL vs Frobenius, có/không L1 và tác động của MMR; giữ nguyên corpus, k, số câu và seed trong mỗi cặp so sánh.
4. Nếu muốn tuyên bố tái hiện paper, thay đồ thị TF-IDF cosine bằng đúng semantic-role/semantic similarity được mô tả trong paper, khớp tiền xử lý và siêu tham số, rồi đánh giá trên dataset paper dùng.

## Mã nguồn liên quan

- `backend/app/core/paper_methods.py`: mã nghiên cứu NMFTS/SNMF cũ, không còn được gọi trong luồng web.
- `backend/app/core/summarize.py`: hai nhánh tóm tắt đang hoạt động, PRF tùy chọn trên cả hai nhánh, loss breakdown và metadata.
- `backend/app/schemas.py`, `backend/app/main.py`: request/response API và batch metrics endpoint.
- `scripts/run_experiment.py`: so sánh baseline và các nhánh NMF.
- `scripts/tune_params.py`: tìm cấu hình chỉ trên validation.
- `frontend/src/pages/Summarizer.jsx`: chọn thuật toán, query và tham số.
- `frontend/src/pages/Evaluation.jsx`: nguồn/reference sẵn của mẫu mới và ROUGE hai output trên một bài.
