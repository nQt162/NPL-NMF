# Thí nghiệm Ablation L1

> Đồng bộ web ngày **15-10-2026**. Trang 04 nhập văn bản/reference trực tiếp, không upload JSONL. Mẫu AI mới và reference có sẵn ở trang 03 nhưng chưa tự chuyển sang trang 04.

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
  --summary-sentences 4 `
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

## Thử cùng mẫu AI mới

Dùng [toy.json](../data/demo/toy.json), mã `test_nmf_01`: 12 câu nguồn và reference trích xuất 4 câu S1/S5/S8/S10, index `[0,4,7,9]`. Trên trang 04 dán `text` vào Văn bản nguồn, `extractive_summary` vào Reference summary; đặt ID tương ứng, **k=3 nhập tay**, **4 câu**, alpha 0.001, lambda 0.7, seeds `42,43,44`.

Trang 04 hiện để trống nguồn/reference, số câu mặc định vẫn 3 và seeds mặc định 42; không tự nhận mẫu và tham số từ trang khác. `toy.json` là JSON cho web, không phải JSONL của CLI; không đưa thẳng nó vào `--input`. CLI ví dụ phía trên cần JSONL hợp lệ, các split hiện rỗng nên chưa thể coi đó là lệnh benchmark chạy ngay.

**Phạm vi F1:** bảng này chấm ROUGE F1 từ summary/reference qua tokenizer, không dùng vector `ground_truth_labels` để tính F1 phân loại câu. Nhãn/index không ép câu được chọn. Reference được cung cấp trước, không sửa sau khi nhìn output để làm L1 thắng.

### Kết quả đã chạy lại ngày 15-10-2026

Cấu hình cố định như trên; mean ± population SD qua ba seed, paired delta so `kl_no_l1` cùng bài/seed:

| Cấu hình | ROUGE-1 | ROUGE-2 | ROUGE-L | Delta R1 | Delta R2 | Delta RL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KL không L1 | 0.612277 ± 0.004395 | 0.518893 ± 0.001072 | 0.567988 ± 0.005494 | 0 | 0 | 0 |
| L1 trên W | 0.209424 ± 0 | 0.010582 ± 0 | 0.115183 ± 0 | -0.402853 | -0.508311 | -0.452804 |
| L1 trên H | 0.606061 ± 0 | 0.520408 ± 0 | 0.575758 ± 0 | -0.006216 | +0.001516 | +0.007770 |
| L1 trên W và H | 0.209424 ± 0 | 0.010582 ± 0 | 0.115183 ± 0 | -0.402853 | -0.508311 | -0.452804 |

Mỗi cấu hình có 3 lượt thành công, không lỗi/fallback, paired_count=3. Cả bốn đều có W gần 0 trung bình 0.666667, H gần 0 trung bình 0.658359, active topic/câu 1 và feature/topic 109.666667, với ngưỡng tuyệt đối 1e-8.

**Đọc đúng:** phạt H tăng nhẹ R2/L nhưng giảm R1; phạt W hoặc W+H giảm mạnh cả ba trong cấu hình này. Cùng tỷ lệ gần 0 không có nghĩa ma trận hoặc summary giống nhau. Không có bảo đảm L1 làm tăng ROUGE/độ thưa, và các weights vẫn có thể đổi dù thống kê threshold không đổi. Đọc thêm scale ambiguity và scoring, không suy nguyên nhân chỉ từ một cột.

k=3 ở thí nghiệm này là chọn tay, không phải auto-k mặc định 1 của bài. Không so trực tiếp bảng này với trang 03 k=1 để quy khác biệt cho L1; so sánh ablation chỉ dùng các cặp trong cùng bảng. Không có tuning để chọn một kết quả thắng. Ba seed của một bài không phải ba bài độc lập hay khoảng tin cậy corpus.

Lượt đo dùng core runner và thư mục tạm để không ghi số vào `results/metrics.csv`. Khi chạy lại qua web sẽ có run_id/artifact mới; không giả định đã có sẵn report của lượt đo tài liệu trong thư mục kết quả.

### Chạy lại mà không cần tạo JSONL

Chạy Python sau từ thư mục `backend` với venv dự án, hoặc làm cùng thao tác trong trang 04:

```python
import json
from pathlib import Path
from app.core.ablation import run_l1_ablation

demo = json.loads(Path("../data/demo/toy.json").read_text(encoding="utf-8"))
report = run_l1_ablation(
    [{"id": demo["doc_id"], "article": demo["text"],
      "reference_summary": demo["extractive_summary"]}],
    Path("../results/ablation/pilot_ai_3seeds"),
    split="pilot", dataset_name=demo["doc_id"],
    k=3, summary_sentences=4, mmr_lambda=0.7,
    alpha=0.001, seeds=[42, 43, 44],
)
print(report["methods"])
print(report["paired_deltas_vs_kl_no_l1"])
```

Các số gắn với scikit-learn 1.7.2, Underthesea 9.5.0, NumPy 2.2.6 và rouge-score 0.1.2 cùng text/stopwords/config hiện tại; thay chúng có thể đổi kết quả.

## Trên web và dữ liệu hiện có

Mở trang **04 / Ablation L1** trong thanh điều hướng, nhập trực tiếp bài viết và reference summary, rồi chọn α, k tùy chọn, ngân sách câu, MMR λ và danh sách seed. Trang gọi backend chạy nền, hiển thị tiến độ, bảng tổng hợp cùng Δ ROUGE theo cặp và liên kết tải ba artifact. Web gửi đúng một bài viết mỗi lần nên kết quả là pilot minh họa, không phải đánh giá corpus. Các lần chạy qua web được lưu riêng theo mã chạy dưới `results/ablation/<run_id>/`.

Các file `data/splits/train.jsonl`, `val.jsonl` và `test.jsonl` hiện có trong repo nhưng đang rỗng. Vì vậy chưa có kết quả corpus để kết luận hiệu năng. Độ lệch qua seeds trên một bài chỉ đo độ nhạy khởi tạo cho bài đó, không phải sai số/độ không chắc chắn trên tập văn bản. Không dùng fixture kiểm thử làm bằng chứng về hiệu năng.
