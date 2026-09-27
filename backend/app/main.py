# TODO: Tạo ứng dụng FastAPI và chỉ mở CORS cho địa chỉ frontend localhost.
# TODO: GET /health trả về {"status": "ok"}.
# TODO: POST /api/simulate: giới hạn 10 câu/50 từ vựng, gọi fit_nmf(trace=True),
#       trả sentences, terms, V và snapshots; ma trận quá lớn trả 422 rõ lý do.
# TODO: POST /api/summarize: gọi pipeline tóm tắt, trả câu gốc, indices, topics,
#       điểm từng thành phần và fallback_reason.
# TODO: POST /api/evaluate chỉ đánh giá một mẫu demo bằng ROUGE-1/2/L F1;
#       chạy batch bằng scripts/run_experiment.py, không qua API.
# TODO: Dùng schema Pydantic để trả 422 cho input thiếu/sai và lỗi dễ hiểu.
