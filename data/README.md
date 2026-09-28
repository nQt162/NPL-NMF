# Dữ liệu cần chuẩn bị

- `demo/toy.json`: ví dụ **tự viết** để mô phỏng, tách biệt dữ liệu đánh giá.
- `../backend/app/core/stopwords_vi.txt`: danh sách stopword nhỏ đang dùng cho TF-IDF; có thể điều chỉnh dựa trên tập validation.
- `raw/articles.jsonl`: hiện rỗng. Khi có quyền sử dụng, thêm mỗi bài một dòng JSON với `id`, `article`, `reference_summary`, `source`. Mục tiêu khoảng 100–300 bài; ít nhất 6 câu đủ nội dung mỗi bài.
- `splits/train.jsonl`, `val.jsonl`, `test.jsonl`: hiện rỗng. Chia theo **bài** với tỷ lệ 60/20/20 và seed 42, không để trùng bài giữa các tập.
- Trước khi dùng dữ liệu thật, lập manifest ghi nguồn, quyền sử dụng và ghi chú nếu reference là sapo/lead. Loại bài trùng, quảng cáo, thiếu reference.

Không đưa dữ liệu không có quyền sử dụng vào repo. Nếu chỉ có 10–20 bài tự viết hoặc được phép dùng, ghi rõ đó là thí nghiệm pilot.
