# TODO: fit_nmf(V, k, seed=42, max_iter=100, trace=False) là hàm NMF
#       TỰ CÀI ĐẶT dùng chung cho mô phỏng và tóm tắt.
# TODO: Kiểm tra V không âm, shape hợp lệ; k=min(requested_k, n_sentences,
#       n_terms). Khởi tạo W,H dương với numpy.random.default_rng(seed).
# TODO: Cập nhật nhân theo đúng thứ tự H rồi W:
#       H *= (W.T @ V) / (W.T @ W @ H + eps)
#       W *= (V @ H.T) / (W @ H @ H.T + eps), eps=1e-9.
# TODO: Ghi loss = 0.5 * sum((V - W @ H)**2) mỗi vòng; dừng khi mức giảm
#       tương đối < 1e-5 sau 5 vòng hoặc đạt max_iter.
# TODO: Trả W,H,losses và snapshots tùy chọn. Snapshot 0 là khởi tạo;
#       các snapshot sau chụp bản sao W,H,WH,loss sau cả hai cập nhật.
# TODO: Chỉ trace ma trận nhỏ/tối đa 20 vòng; không giữ snapshots khi chạy batch.
