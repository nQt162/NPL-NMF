# TODO: Gọi sentences -> preprocess -> vectorize -> fit_nmf cho một bài.
# TODO: Nếu <2 câu/từ hữu ích hoặc V rỗng, dùng Lead-N và nêu fallback_reason.
# TODO: Tạo topic từ 5 từ cao nhất mỗi hàng H; salience từ tổng W GỐC
#       theo cột, chuẩn hóa tổng 1. Chuẩn hóa mỗi hàng W về tổng 1 để chấm câu.
# TODO: Greedy chọn đủ ngân sách 1..min(5,n_sentences) bằng:
#       relevance = tổng W_norm[i,t] * salience[t]
#       coverage_gain = tổng salience[t] * max(0, W_norm[i,t]-coverage[t])
#       redundancy = max cosine(V[i], V[j]) của câu đã chọn, hoặc 0.
#       score = alpha*relevance + beta*coverage_gain - gamma*redundancy.
# TODO: Sau mỗi lượt, cập nhật coverage theo max; hòa điểm chọn index nhỏ.
#       Sắp câu đã chọn theo thứ tự gốc trước khi ghép summary.
# TODO: Trả index/câu nguyên văn, dominant_topic và thành phần điểm tại
#       thời điểm xét/chọn; ghi rõ quy ước thời điểm trong code/API.
