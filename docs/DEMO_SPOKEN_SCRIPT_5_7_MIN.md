# Trang 01: Văn bản mẫu, ma trận và loss

Em xin giới thiệu TopicSum, hệ thống tóm tắt trích xuất tiếng Việt bằng Local NMF và MMR. Hệ thống chọn câu có sẵn và giữ thứ tự nguồn, không dùng LLM để viết câu mới. Bài mẫu có 12 câu về AI tại Việt Nam, từ đầu tư công nghệ đến nông nghiệp, y tế và các thách thức về nhân lực, dữ liệu. Mục tiêu demo là giữ bốn câu và giải thích tại sao chúng được chọn.

Em để trống k, giữ seed 42 và chạy mười vòng mô phỏng KL. Mỗi câu là một document để fit TF-IDF cục bộ. Trong môi trường hiện tại, V có 12 hàng và 321 đặc trưng unigram, bigram, không phải số từ viết cách trong bài. Auto-k chọn một nên W là 12 nhân một, H là một nhân 321, còn WH xấp xỉ lại V.

Kéo từ vòng không đến vòng mười, objective giảm từ khoảng 1070,90 xuống 155,24. Nó gồm sai số tái tạo cộng penalty L1 trên W và H, không cần reference để tính. Mỗi vòng custom MU cập nhật H khi giữ W, rồi dùng H mới cập nhật W; web hiển thị sau cả hai bước. Đây là mô phỏng MU tự cài, không phải log của lần sklearn fit tạo summary ở trang hai.

# Trang 01: Cách chọn k

Bộ chọn thử các k khả thi từ một đến mười bằng mười lượt che khoảng 20% ô TF-IDF dương, loại chúng khỏi phần fit rồi so giá trị dự đoán với giá trị thật bằng KL. Với bài và seed này, k bằng một có điểm khoảng 3,073, thấp hơn các ứng viên còn lại.

Điều này chỉ nói rank một khôi phục các ô giữ lại tốt nhất theo protocol hiện tại, không nói bài chỉ có một chủ đề theo cách con người đọc. k cũng không phải số câu summary: em vẫn yêu cầu bốn câu. Có thể nhập k bằng ba để xem nhiều thành phần hơn, nhưng phải gọi đó là cấu hình chọn tay, không phải kết quả auto-k.

# Trang 02: Tóm tắt và quyết định MMR

Ở trang tóm tắt, bài nguồn mới đã có sẵn và ngân sách mặc định là bốn câu. Em để k trống, query trống, alpha W và H bằng 0,001, L1 ratio bằng một, MMR lambda bằng 0,7. Backend fit NMF cục bộ của scikit-learn, không load model Global cũ.

Nhánh KL giữ câu bốn, sáu, tám và mười một; Frobenius giữ câu hai, ba, bảy và mười. Đây là thuật toán tự chọn, không ép theo reference. Từ khóa và câu đại diện giúp diễn giải factors, nhưng không phải nhãn lĩnh vực đã xác nhận. Với k bằng một, relevance các câu còn gần bằng nhau, nên đây cũng là giới hạn cần nói rõ.

Em mở trace MMR của Frobenius ở vòng hai, sau khi đã giữ câu mười. Câu ba có relevance nhỉnh hơn câu bảy, nhưng câu bảy ít trùng câu đã chọn hơn. Sau công thức 0,7 nhân relevance trừ 0,3 nhân redundancy, điểm câu bảy khoảng 0,6932, cao hơn 0,6905 của câu ba nên nó được chọn.

Redundancy là cosine lớn nhất với tất cả câu đã giữ, không bắt buộc cùng chủ đề. MMR so ứng viên trong cùng vòng rồi tính lại khi tập đã chọn thay đổi. Greedy chọn mười, bảy, ba, hai; khi xuất summary, hệ thống sắp thành hai, ba, bảy, mười để giữ thứ tự nguồn.

# Trang 02: Query ung thư phổi

Em giữ Frobenius và nhập truy vấn “ung thư phổi”, trọng số 0,5. PRF lấy câu tám làm pseudo-relevant, mở rộng query từ nội dung câu đó rồi kết hợp với topic relevance trước MMR. Summary chuyển sang câu hai, ba, bốn và tám, đưa câu sàng lọc bệnh vào kết quả.

Objective vẫn khoảng 5,8091 vì query chỉ tác động chấm điểm sau phân rã, không thay W, H hay loss. Query dùng được cho cả KL và Frobenius. Chuyển sang KL, tập câu có thể không đổi vì câu tám đã được giữ từ trước; query không bắt buộc thay summary, và query ngoài vocabulary sẽ được báo lý do bỏ PRF.

# Trang 03: Hai output và reference

Trang đánh giá đã có sẵn bài nguồn và reference bốn câu do người dùng cung cấp, gồm câu một, năm, tám và mười. Reference là mục tiêu chấm, không phải output được hệ thống suy luận sẵn và không đưa vào fit NMF. Em giữ k trống, bốn câu, seed 42 và cùng tham số cho hai nhánh; trang này không dùng query.

Hệ thống tạo hai outputs rồi tính ROUGE-1, ROUGE-2 và ROUGE-L F1 với cùng reference. Frobenius có ROUGE-1 khoảng 0,3979, cao hơn 0,3897 của KL; nhưng KL cao hơn ROUGE-2 và ROUGE-L. Ngay trong mẫu này, kết luận còn phụ thuộc metric, không có một nhánh thắng mọi chỉ số.

ROUGE đo overlap token và chuỗi token với reference, không xác nhận toàn bộ sự thật hay độ mạch lạc. Điểm 0,3979 không có nghĩa summary đúng 39,79% nội dung. Loss phân rã, MMR chọn câu và ROUGE đánh giá là ba đại lượng khác vai trò.

# Trang 04: Ablation L1 và kết luận

Em dán cùng bài và reference, đổi ngân sách thành bốn, đặt k bằng ba cố định và chạy các seed 42, 43, 44. Rank ba ở đây do em nhập để khảo sát penalty, không phải auto-k đã chọn ba. Trong mỗi cặp bài và seed, bốn nhánh giữ cùng k, MMR và TF-IDF, chỉ đổi không L1, L1 trên W, trên H hoặc trên cả hai; query luôn tắt.

Trên cấu hình này, baseline có ROUGE-1 trung bình khoảng 0,6123. Chỉ phạt H làm ROUGE-2 và ROUGE-L tăng nhẹ nhưng ROUGE-1 giảm nhẹ; phạt W hoặc cả hai làm cả ba metric giảm mạnh. Tỷ lệ gần zero của W và H lại bằng nhau giữa bốn nhánh, nên không thể chỉ nhìn độ thưa để kết luận chất lượng. Kết quả cần được báo trung thực, không phải cứ thêm L1 là summary tốt hơn.

Ba seed vẫn là một bài, không phải ba bài độc lập; cũng không so thẳng bảng k bằng ba này với trang trước k bằng một để quy khác biệt cho L1. Report lưu cấu hình, metrics, predictions và fallback để kiểm tra lại.

Qua demo, hệ thống đã có bốn luồng hoạt động và có thể giải thích từ biểu diễn, tối ưu đến lựa chọn câu và đánh giá. Tuy nhiên corpus train, validation, test còn rỗng, nên chưa đủ căn cứ tuyên bố phương pháp nào tối ưu tổng quát; bước tiếp theo là dữ liệu độc lập, chọn cấu hình trên validation và đánh giá test. Em xin kết thúc demo và nhận câu hỏi.
