# Thí nghiệm Ablation L1

## Mục tiêu

Thí nghiệm cô lập tác động của L1 trong Local KL-NMF + MMR. Cả bốn nhánh dùng chung KL divergence, solver MU, TF-IDF tiếng Việt, quy tắc chọn câu MMR, ngân sách câu, corpus, seed và `l1_ratio=1.0`. Query PRF được tắt ở tất cả nhánh.

| Cấu hình | `alpha_w` | `alpha_h` | Diễn giải |
| --- | ---: | ---: | --- |
| `kl_no_l1` | 0 | 0 | Baseline KL không có penalty L1 |
| `kl_l1_w_only` | α | 0 | Chỉ phạt W, biểu diễn câu-chủ đề |
| `kl_l1_h_only` | 0 | α | Chỉ phạt H, biểu diễn chủ đề-từ |
| `kl_l1_w_h` | α | α | Phạt đồng thời W và H |

Mặc định `α = 0.001`, có thể đổi để khảo sát trên validation. Đây là giá trị bảo thủ, chưa được chứng minh tối ưu do split dữ liệu trong repo đang rỗng. Không đặt `l1_ratio=0` cho baseline: cấu hình đó sẽ đổi penalty sang L2 thay vì loại bỏ regularization. Baseline dùng `alpha_w=alpha_h=0`.

## Ghép cặp và chọn k

Mỗi cặp `id + seed` có một giá trị k dùng chung. Khi không truyền k, k được chọn một lần bằng positive-entry masked KL validation trên cấu hình không L1, sau đó truyền cố định cho cả bốn lần gọi `summarize()`. Khi dữ liệu quá thưa, phương pháp chọn k ghi nhãn heuristic; cùng k heuristic vẫn được dùng cho cả bốn nhánh. Khi truyền k, cùng giá trị được dùng cho mọi nhánh.

Mỗi nhánh nhận cùng bài viết, reference summary, số câu tóm tắt, MMR λ và seed. Seed đi vào khởi tạo `nndsvdar`, nên các seed khác nhau thực sự tạo khởi tạo khác nhau; cùng seed ghép cặp giữa bốn cấu hình. CLI đọc JSONL với các trường chuỗi `id`, `article`, `reference_summary`; dòng không hợp lệ và ID lặp được ghi nhận, không âm thầm đưa vào so sánh.

Chọn α, k, số câu và MMR λ trên validation. Sau khi khóa cấu hình, mới chạy test để báo cáo cuối cùng; không chỉnh tham số dựa trên test. Split được ghi vào report để kết quả có thể truy nguyên. Nếu dùng pilot/tự viết, cần ghi rõ đây không phải đánh giá trên corpus đại diện.

## Chỉ số

- ROUGE-1/2/L F1 được tính giữa summary sinh ra và reference summary bằng tokenizer tiếng Việt dùng chung trong dự án.
- ROUGE được tổng hợp dưới dạng mean ± độ lệch chuẩn dân số trên các lượt NMF thành công. Fallback và lỗi được đếm riêng; không tính fallback như một lượt NMF thành công.
- Δ ROUGE được tính theo từng cặp cùng `id + seed` so với `kl_no_l1`, rồi mới lấy trung bình. Δ dương nghĩa là ROUGE cao hơn trên các cặp đang xét; không chứng minh cải thiện tổng quát.
- Độ thưa W/H là tỷ lệ phần tử có giá trị tuyệt đối ≤ `1e-8` (ngưỡng tuyệt đối). `active_topics_per_sentence` đếm số phần tử W vượt ngưỡng, trung bình theo câu; `active_terms_per_topic` đếm số phần tử H vượt ngưỡng, trung bình theo topic.
- Runtime là thời gian gọi `summarize()` với k đã cố định. Thời gian chọn k được ghi riêng để không cộng riêng chi phí chọn k cho baseline.

Objective tổng không dùng để xếp hạng giữa các cấu hình: penalty khác nhau làm các giá trị objective không còn là phép so sánh trực tiếp. Kết luận cần dựa trên ROUGE ghép cặp, sparsity, độ ổn định qua seeds và, nếu có, đánh giá định tính.

## Chạy bằng CLI

Chạy lệnh từ thư mục `backend` sau khi cài môi trường dự án:

```powershell
.\.venv\Scripts\python.exe ..\scripts\run_ablation.py `
  --input ..\data\splits\val.jsonl `
  --output ..\results\ablation\validation `
  --split validation `
  --summary-sentences 3 `
  --mmr-lambda 0.7 `
  --alpha 0.001 `
  --seeds 42 43 44
```

Thử nhanh một seed bằng `--seeds 42`. Khi chạy test, đổi sang `--input ..\data\splits\test.jsonl --split test` sau khi đã chốt tham số từ validation. Có thể truyền `--k 3` để cố định k; nếu bỏ, công cụ chọn k một lần cho mỗi bài-seed.

Ba artifact được tạo trong thư mục output:

- `ablation_metrics.csv`: một dòng cho mỗi bài, seed và cấu hình, gồm ROUGE, runtime, sparsity và fallback.
- `ablation_predictions.jsonl`: summary, câu đã chọn và metadata từng lượt.
- `ablation_report.json`: cấu hình, số lượt hợp lệ/lỗi/fallback, mean/std và chênh lệch ROUGE ghép cặp.

Kết quả ở `results/ablation/` được giữ tách biệt và không ghi đè `results/metrics.csv` hoặc `results/predictions.jsonl`.

## Trên web và dữ liệu hiện có

Mở trang **04 / Ablation L1** trong thanh điều hướng, nhập trực tiếp bài viết và reference summary, rồi chọn α, k tùy chọn, ngân sách câu, MMR λ và danh sách seed. Trang gọi backend chạy nền, hiển thị tiến độ, bảng tổng hợp cùng Δ ROUGE theo cặp và liên kết tải ba artifact. Web gửi đúng một bài viết mỗi lần nên kết quả là pilot minh họa, không phải đánh giá corpus. Các lần chạy qua web được lưu riêng theo mã chạy dưới `results/ablation/<run_id>/`.

Các file `data/splits/train.jsonl`, `val.jsonl` và `test.jsonl` hiện có trong repo nhưng đang rỗng. Vì vậy chưa có kết quả corpus để kết luận hiệu năng. Độ lệch qua seeds trên một bài chỉ đo độ nhạy khởi tạo cho bài đó, không phải sai số/độ không chắc chắn trên tập văn bản. Không dùng fixture kiểm thử làm bằng chứng về hiệu năng.
