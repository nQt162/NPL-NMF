# Tổng hợp các cải tiến NMF

Tài liệu này tổng hợp các thay đổi đã thực hiện cho project NMF/TopicSum, gồm: phần cũ hoạt động như thế nào, đã cải tiến ra sao, hiệu quả khác gì trước, và các file/đoạn code liên quan.

## 1. Tổng quan trước và sau

| Hạng mục | Trước khi sửa | Sau khi sửa | Hiệu quả |
| --- | --- | --- | --- |
| Luồng tóm tắt | Dùng global NMF model từ `nmf_corpus.joblib` đã train sẵn trên corpus lớn | Fit NMF cục bộ trên chính văn bản người dùng nhập | Tóm tắt bám nội dung cục bộ, không bị lẫn từ vựng của corpus cũ |
| Đơn vị tài liệu | Văn bản người dùng bị đưa vào model toàn cục | Mỗi câu trong văn bản là một document riêng | Phù hợp hơn với extractive summarization cho văn bản ngắn |
| TF-IDF | Phụ thuộc vocabulary/model đã train từ trước | Tạo vectorizer mới và `fit_transform` trên các câu đầu vào | Loại bỏ lỗi out-of-vocabulary do vocabulary cũ |
| Loss NMF | Frobenius/MSE mặc định hoặc objective cũ | KL divergence + L1 regularization | Phù hợp hơn với dữ liệu văn bản thưa, topic sắc nét hơn |
| Solver | Mặc định của sklearn | `solver="mu"` | Đúng yêu cầu của sklearn khi dùng `beta_loss="kullback-leibler"` |
| Sparsity | Không có L1 penalty rõ ràng | `alpha_W=0.1`, `alpha_H=0.1`, `l1_ratio=1.0` | Ép W và H thưa hơn, giảm từ khóa nhiễu và topic chồng chéo |
| Chọn k | Heuristic/fixed k | Masked KL imputation nếu người dùng để trống k | Chọn k dựa trên khả năng tái tạo dữ liệu bị che, thực tế hơn heuristic cứng |
| Chọn câu | Dễ phụ thuộc điểm đơn giản từ W | Maximum Marginal Relevance chuẩn: `λ * relevance - (1 - λ) * redundancy` | Giảm trùng lặp, vẫn ưu tiên câu liên quan nhất, giữ thứ tự gốc khi xuất summary |
| Hiển thị frontend | Loss/k dễ gây hiểu nhầm với cấu hình cũ | Hiển thị KL + L1, KL data, L1 penalty, candidate k, topic/câu | Người dùng thấy rõ backend đang dùng objective mới |

## Cập nhật mới nhất: chuẩn hóa MMR

Sau lần sửa mới nhất, phần chọn câu tóm tắt đã được chuyển từ scoring mở rộng tự thiết kế sang Maximum Marginal Relevance chuẩn.

Trước đó, hệ thống dùng công thức kết hợp nhiều thành phần:

```text
score = alpha * relevance + beta * coverage_gain + position_weight * position_prior + length_weight * length_quality - gamma * redundancy
```

Sau khi chuẩn hóa, hệ thống dùng đúng công thức MMR:

```text
MMR(s) = λ * relevance(s) - (1 - λ) * max_similarity(s, selected)
```

Ý nghĩa thay đổi:

- `relevance` được lấy từ topic distribution của NMF, thể hiện mức đại diện của câu với các chủ đề chính.
- `redundancy` được tính bằng cosine similarity giữa câu ứng viên và các câu đã chọn.
- `mmr_lambda` mặc định là `0.7`; giá trị này càng gần 1 thì càng ưu tiên câu liên quan, càng gần 0 thì càng phạt trùng lặp mạnh.
- Các thành phần `coverage_gain`, `position_prior`, `length_quality`, `alpha`, `beta`, `gamma` không còn nằm trong công thức chọn câu chính.
- Frontend đã đổi sang một control duy nhất là `MMR λ`, đồng thời hiển thị `Relevance`, `Redundancy` và `MMR score` cho từng câu.

File đã cập nhật:

- `backend/app/core/summarize.py`: triển khai MMR chuẩn trong `_select_sentences(...)`.
- `backend/app/schemas.py`: thêm `mmr_lambda`, `selection_method`; bỏ các trường scoring cũ khỏi sentence analysis.
- `frontend/src/pages/Summarizer.jsx`: đổi UI sang tham số `MMR λ`.
- `backend/tests/test_summary.py` và `backend/tests/test_api.py`: thêm kiểm tra MMR.

## 2. Bỏ global model `nmf_corpus.joblib`

### Vấn đề cũ

Phiên bản cũ load một NMF model toàn cục từ file `nmf_corpus.joblib`. Model này đã học vocabulary và ma trận chủ đề từ corpus lớn, nên khi người dùng nhập văn bản ngắn, topic và từ khóa có thể đến từ corpus cũ thay vì từ chính đoạn văn hiện tại.

### Cải tiến đã làm

- Xóa luồng phụ thuộc vào `joblib.load()` và global corpus model trong flow tóm tắt.
- Xóa file code wrapper global model `backend/app/core/corpus_model.py`.
- Gỡ bỏ package-data liên quan đến `nmf_corpus.joblib` trong `backend/pyproject.toml`.
- API tóm tắt gọi trực tiếp local summarizer.

### Hiệu quả

- Không còn dùng topic/vocabulary của 3260 bài báo cũ để tóm tắt văn bản mới.
- Tránh lỗi lấy từ khóa ngoài ngữ cảnh.
- Luồng tóm tắt đúng bản chất hơn: học chủ đề từ chính văn bản đầu vào.

### File liên quan

- `backend/app/core/corpus_model.py`: đã xóa.
- `backend/pyproject.toml`: bỏ dòng đóng gói `nmf_corpus.joblib`.
- `backend/app/main.py:114`: endpoint `/api/summarize` gọi hàm `summarize(...)`.
- `backend/app/core/summarize.py:215`: hàm `summarize(...)` là luồng tóm tắt mới.

## 3. Local NMF on-the-fly cho tóm tắt

### Vấn đề cũ

Global model không phản ánh các ý chính cục bộ của đoạn văn ngắn. Khi vocabulary của người dùng khác vocabulary corpus train, kết quả transform dễ lệch.

### Cải tiến đã làm

Luồng mới trong `backend/app/core/summarize.py`:

1. Tách câu từ text đầu vào.
2. Tiền xử lý từng câu.
3. Tạo ma trận TF-IDF cục bộ trên chính danh sách câu.
4. Chọn hoặc suy ra `k`.
5. Fit NMF mới trên ma trận cục bộ.
6. Lấy topic terms từ ma trận H mới.
7. Chọn câu và ghép lại theo thứ tự gốc.

### Hiệu quả

- Topic và từ khóa đến từ văn bản hiện tại.
- Tóm tắt hợp lý hơn với văn bản ngắn.
- Không còn hiện tượng lấy từ khóa ngoài ngữ cảnh từ corpus cũ.

### File liên quan

- `backend/app/core/summarize.py:215`: hàm tổng `summarize(...)`.
- `backend/app/core/summarize.py:251`: chọn hoặc resolve `k`.
- `backend/app/core/summarize.py:263`: fit local KL-NMF.
- `backend/app/core/summarize.py:264`: chọn câu bằng MMR chuẩn.
- `backend/app/core/summarize.py:279`: ghép summary theo thứ tự câu gốc.

## 4. Đổi objective sang KL divergence + L1 sparsity

### Vấn đề cũ

Frobenius/MSE xem sai số tái tạo như bình phương khoảng cách Euclidean. Với dữ liệu text sparse như TF-IDF, cách này thường tạo topic kém sắc nét hơn và dễ có từ khóa nhiễu.

### Cải tiến đã làm

Cấu hình NMF mới:

```python
NMF(
    n_components=k,
    init="nndsvda",
    solver="mu",
    beta_loss="kullback-leibler",
    alpha_W=0.1,
    alpha_H=0.1,
    l1_ratio=1.0,
    max_iter=500,
    random_state=seed,
)
```

### Hiệu quả

- KL divergence phù hợp hơn với dữ liệu không âm, thưa và có tính phân phối như tần suất/TF-IDF.
- `solver="mu"` là solver phù hợp khi dùng KL trong sklearn.
- L1 trên W giúp mỗi câu tập trung vào ít chủ đề hơn.
- L1 trên H giúp mỗi topic có bộ từ khóa đặc trưng hơn.
- `max_iter=500` giúp MU solver có nhiều vòng lặp hơn để hội tụ.

### File liên quan

- `backend/app/core/nmf.py:14`: khai báo `SPARSE_KL_LOSS_NAME = "kullback-leibler"`.
- `backend/app/core/nmf.py:15`: khai báo `SPARSE_KL_SOLVER = "mu"`.
- `backend/app/core/nmf.py:16`: khai báo `SPARSE_KL_ALPHA_W = 0.1`.
- `backend/app/core/nmf.py:17`: khai báo `SPARSE_KL_ALPHA_H = 0.1`.
- `backend/app/core/nmf.py:18`: khai báo `SPARSE_KL_L1_RATIO = 1.0`.
- `backend/app/core/nmf.py:19`: khai báo `SPARSE_KL_MAX_ITER = 500`.
- `backend/app/core/summarize.py:109`: khởi tạo sklearn `NMF`.
- `backend/app/core/summarize.py:112`: gán `solver=SPARSE_KL_SOLVER`.
- `backend/app/core/summarize.py:113`: gán `beta_loss=SPARSE_KL_LOSS_NAME`.
- `backend/app/core/summarize.py:114`: gán `alpha_W=SPARSE_KL_ALPHA_W`.
- `backend/app/core/summarize.py:115`: gán `alpha_H=SPARSE_KL_ALPHA_H`.
- `backend/app/core/summarize.py:116`: gán `l1_ratio=SPARSE_KL_L1_RATIO`.
- `backend/app/core/summarize.py:117`: gán `max_iter=SPARSE_KL_MAX_ITER`.

## 5. Loss mới trong mô phỏng NMF

### Vấn đề cũ

Frontend vẫn hiển thị loss cũ hoặc chỉ một số loss chung, làm người dùng tưởng backend chưa đổi objective.

### Cải tiến đã làm

- Backend trả về `loss_name`, `solver`, `alpha_W`, `alpha_H`, `l1_ratio`.
- Snapshot trả thêm:
  - `loss`: total loss.
  - `data_loss`: phần KL data loss.
  - `regularization_loss`: phần L1 penalty.
- Frontend hiển thị rõ `KL + L1 loss`, `KL data`, `L1 penalty`.

### Hiệu quả

- Người dùng đọc đúng bản chất loss hiện tại.
- Có thể tách được phần lỗi tái tạo dữ liệu và phần penalty điều chuẩn.
- Giá trị loss là số thập phân bình thường, nhưng không nên so trực tiếp với Frobenius loss cũ vì objective đã khác.

### File liên quan

- `backend/app/core/nmf.py:44`: `Snapshot` có `data_loss` và `regularization_loss`.
- `backend/app/core/nmf.py:76`: `objective_breakdown(...)` tính data loss và regularization.
- `backend/app/main.py:89`: API trả `loss_name`.
- `backend/app/main.py:90`: API trả `solver`.
- `backend/app/main.py:91`: API trả `alpha_W`.
- `backend/app/main.py:92`: API trả `alpha_H`.
- `backend/app/main.py:93`: API trả `l1_ratio`.
- `backend/app/main.py:104`: snapshot trả `data_loss`.
- `backend/app/main.py:105`: snapshot trả `regularization_loss`.
- `backend/app/schemas.py:25`: schema `SnapshotResponse`.
- `frontend/src/pages/Visualizer.jsx:141`: hiển thị tên loss.
- `frontend/src/pages/Visualizer.jsx:143`: block breakdown loss.
- `frontend/src/pages/Visualizer.jsx:144`: hiển thị `KL data`.
- `frontend/src/pages/Visualizer.jsx:145`: hiển thị `L1 penalty`.

## 6. Cải tiến cách tự động tìm k

### Vấn đề cũ

Công thức `k = max(1, len(sentences) // 2)` đơn giản, dễ sai với văn bản có ít/nhiều chủ đề thật. Fixed k cũng không linh hoạt.

### Cải tiến đã làm

Nếu người dùng không nhập k:

1. Lấy các ô TF-IDF dương.
2. Che một phần các ô này làm validation.
3. Fit NMF với nhiều ứng viên k.
4. Tái tạo các ô bị che.
5. Tính KL imputation error.
6. Chọn k có điểm thấp nhất.

### Hiệu quả

- K được chọn dựa trên dữ liệu đầu vào, không chỉ dựa vào số câu.
- Giảm nguy cơ chọn k quá lớn cho văn bản ngắn.
- Giảm nguy cơ ép tất cả văn bản vào 1 topic khi có nhiều cụm ý rõ.

### File liên quan

- `backend/app/core/nmf.py:20`: `K_SELECTION_MAX_K = 6`.
- `backend/app/core/nmf.py:21`: `K_SELECTION_TRIALS = 3`.
- `backend/app/core/nmf.py:22`: `K_SELECTION_VALIDATION_FRACTION = 0.2`.
- `backend/app/core/nmf.py:305`: hàm `select_k_by_imputation(...)`.
- `backend/app/core/nmf.py:326`: giới hạn ứng viên k.
- `backend/app/core/nmf.py:327`: đặt lower bound k cho văn bản có đủ dữ liệu.
- `backend/app/core/nmf.py:341`: tạo danh sách ứng viên k.
- `backend/app/core/nmf.py:359`: tính KL error trên phần bị che.
- `backend/app/core/nmf.py:368`: sắp xếp và chọn k tốt nhất.
- `backend/app/core/summarize.py:85`: tóm tắt gọi `select_k_by_imputation(...)`.
- `backend/app/main.py:61`: simulate gọi `select_k_by_imputation(...)`.
- `frontend/src/pages/Visualizer.jsx:156`: hiển thị các candidate k.
- `frontend/src/pages/Summarizer.jsx:125`: hiển thị các candidate k trong trang tóm tắt.

## 7. Cải tiến cách chọn câu tóm tắt bằng MMR chuẩn

### Vấn đề cũ

Nếu chỉ lấy câu có tổng trọng số W cao nhất, summary có thể bị trùng lặp ý vì các câu cùng nói về một nội dung vẫn đều có điểm cao.

### Cải tiến đã làm

Hàm chọn câu mới dùng Maximum Marginal Relevance chuẩn:

```text
MMR(s) = λ * relevance(s) - (1 - λ) * max_similarity(s, selected)
```

Trong đó:

- `relevance`: độ liên quan của câu với các topic chính từ NMF.
- `redundancy`: cosine similarity lớn nhất giữa câu ứng viên và các câu đã chọn.
- `mmr_lambda`: tham số cân bằng giữa liên quan và chống trùng lặp, mặc định `0.7`.

Sau khi chọn xong, các câu được sắp xếp lại theo index gốc trước khi ghép summary.

### Hiệu quả

- Summary bớt lặp ý nhờ hình phạt redundancy chuẩn MMR.
- Vẫn ưu tiên câu đại diện tốt cho chủ đề chính nhờ relevance từ ma trận W.
- Văn bản tóm tắt đọc trôi chảy hơn vì giữ thứ tự xuất hiện ban đầu.

### File liên quan

- `backend/app/core/summarize.py:130`: hàm `_topic_relevance(...)` tính relevance từ NMF.
- `backend/app/core/summarize.py:146`: hàm `_max_similarity_to_selected(...)` tính redundancy.
- `backend/app/core/summarize.py:164`: hàm `_select_sentences(...)`.
- `backend/app/core/summarize.py:187`: công thức MMR chuẩn.
- `backend/app/core/summarize.py:199`: chọn câu có MMR score cao nhất.
- `backend/app/core/summarize.py:202`: sắp xếp câu đã chọn theo thứ tự gốc.
- `backend/app/core/summarize.py:279`: ghép summary từ các câu theo thứ tự gốc.

## 8. Mở rộng API contract

### Cải tiến đã làm

Schema API được mở rộng để frontend/doc/debug biết rõ backend đang dùng cấu hình nào:

- `k`.
- `loss_name`.
- `solver`.
- `alpha_W`.
- `alpha_H`.
- `l1_ratio`.
- `k_selection_method`.
- `k_candidates`.
- `selection_method`.
- `mmr_lambda`.
- `data_loss`.
- `regularization_loss`.

### Hiệu quả

- Frontend không cần đoán cấu hình backend.
- Dễ debug khi thấy loss/k khác kỳ vọng.
- Phù hợp hơn với yêu cầu giải thích quá trình NMF.

### File liên quan

- `backend/app/schemas.py:25`: `SnapshotResponse`.
- `backend/app/schemas.py:36`: `KCandidateResponse`.
- `backend/app/schemas.py:40`: `SimulateResponse`.
- `backend/app/schemas.py:82`: `SummarizeResponse`.
- `backend/app/main.py:88`: response `/api/simulate`.
- `backend/app/core/summarize.py:272`: response `/api/summarize`.

## 9. Cải tiến frontend

### Cải tiến đã làm

- Trang mô phỏng ghi rõ NMF đang dùng KL + L1.
- Trang tóm tắt để trống k mặc định để backend tự chọn.
- Trang tóm tắt có tham số `MMR λ` để cân bằng relevance và redundancy.
- Hiển thị candidate k và điểm từng candidate.
- Hiển thị topic terms từ H cục bộ.
- Hiển thị phân tích từng câu: topic chính, relevance, redundancy và MMR score.
- Hiển thị loss breakdown thay vì chỉ một số loss chung.

### Hiệu quả

- Giao diện trung thực hơn với backend mới.
- Người dùng nhìn được vì sao hệ thống chọn k và chọn câu.
- Tốt hơn cho demo, báo cáo và giải thích thuật toán.

### File liên quan

- `frontend/src/App.jsx:58`: mô tả luồng TF-IDF cục bộ, KL divergence và L1 sparsity.
- `frontend/src/pages/Visualizer.jsx:91`: mô tả masked imputation và KL/L1.
- `frontend/src/pages/Visualizer.jsx:141`: hiển thị loss.
- `frontend/src/pages/Visualizer.jsx:143`: hiển thị breakdown.
- `frontend/src/pages/Visualizer.jsx:156`: hiển thị candidate k.
- `frontend/src/pages/Summarizer.jsx:45`: gửi `mmr_lambda` lên backend.
- `frontend/src/pages/Summarizer.jsx:88`: input tham số `MMR λ`.
- `frontend/src/pages/Summarizer.jsx:120`: hiển thị `MMR λ` trong kết quả.
- `frontend/src/pages/Summarizer.jsx:134`: hiển thị relevance, redundancy và MMR score.
- `frontend/src/styles.css:139`: style loss breakdown.
- `frontend/src/styles.css:150`: style candidate k.

## 10. Kiểm thử đã chạy

Đã chạy các kiểm thử chính:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests
npm run build
npm test
```

Kết quả:

- Backend tests: 14 passed.
- Frontend tests: 2 passed.
- Frontend build: thành công.

## 11. Đánh giá mức độ đúng với paper và thực tế

### Điểm đúng hướng

- KL divergence là một mở rộng NMF phù hợp cho dữ liệu không âm và có tính phân phối.
- Multiplicative Update là solver đúng khi dùng KL divergence trong sklearn.
- L1 regularization trên W/H đúng mục tiêu tạo sparsity.
- Local NMF đúng bản chất extractive summarization cho văn bản ngắn.
- Chọn k bằng masked imputation thực tế hơn heuristic cứng.
- Chọn câu bằng MMR chuẩn đúng mục tiêu cân bằng giữa relevance và chống trùng lặp.

### Giới hạn còn lại

- TF-IDF + KL là cách làm thực dụng, nhưng nếu muốn theo xác suất chặt hơn có thể thử count matrix hoặc normalized term-frequency.
- `alpha_W=0.1` và `alpha_H=0.1` là tham số khởi đầu hợp lý, chưa phải kết quả tuning trên dataset lớn.
- Điểm summary cần benchmark bằng ROUGE/BERTScore trên tập test để kết luận chất lượng khách quan.
- Mô phỏng W/H trong frontend phù hợp để giải thích thuật toán; sklearn NMF mới là solver chính cho tóm tắt.

## 12. Tóm lại

Project đã chuyển từ mô hình global NMF sang local KL-NMF có L1 sparsity và chọn câu bằng MMR chuẩn. Đây là thay đổi quan trọng về bản chất: hệ thống không còn dùng topic/vocabulary của corpus cũ để tóm tắt văn bản mới, mà học topic trực tiếp từ chính đoạn văn người dùng nhập. Kết quả kỳ vọng là topic sát nội dung hơn, từ khóa gọn hơn, cách chọn k minh bạch hơn và summary ít trùng lặp hơn nhờ công thức `λ * relevance - (1 - λ) * redundancy`.
