# TopicSum (NPL-NMF)

Bộ khung dự án cho ứng dụng tóm tắt **trích xuất** văn bản tiếng Việt bằng NMF. Yêu cầu chi tiết và tiêu chí hoàn thành nằm trong [TopicSum_AGENT.md](TopicSum_AGENT.md).

## Trạng thái hiện tại

Đây mới là **cấu trúc và TODO comment** để phân công công việc. Chưa cài thư viện, chưa triển khai API, giao diện, thuật toán, kiểm thử hay thí nghiệm. `results/metrics.csv` và `results/predictions.jsonl` hiện rỗng, không phải kết quả đánh giá.

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

Đọc TODO trong từng file để biết yêu cầu đầu vào/đầu ra và các trường hợp biên. `pyproject.toml` và `package.json` chỉ chứa metadata tối thiểu; cần bổ sung dependencies và lệnh chạy khi bắt đầu triển khai. Khi đó cập nhật README này bằng **lệnh cài đặt, chạy, kiểm thử thực tế**. Không dùng các lệnh dự kiến trong đặc tả như thể chúng đã hoạt động.

## Quy tắc dữ liệu và đánh giá

- Không đưa bài báo có bản quyền chưa được phép sử dụng hoặc khóa API vào repo.
- NMF fit riêng trên từng bài; chọn cấu hình bằng tập validation và chỉ đánh giá test sau khi chốt cấu hình.
- Nếu chỉ có dữ liệu nhỏ tự viết, ghi rõ đó là pilot và không suy rộng kết quả.
- Không tự điền số liệu vào `results/metrics.csv` hay `results/predictions.jsonl`.
