# TODO: Nhận --input và --output; mặc định seed=42, lưu cấu hình k,
#       summary_sentences, alpha/beta/gamma cùng kết quả.
# TODO: Chạy Lead-N và NMF trên CÙNG bài, CÙNG ngân sách; NMF fit riêng
#       trên từng bài và không giữ snapshots.
# TODO: Tính ROUGE-1/2/L F1, thời gian; xuất metrics.csv theo bài/phương pháp
#       và predictions.jsonl. Báo mean, số bài hợp lệ, lỗi và fallback.
# TODO: Chọn tham số trên val; test chạy một lần sau khi chốt cấu hình.
# TODO: Phân tích 3 ca tốt, 3 ca thất bại và k=1,2,3 trên val; không bịa số liệu.
