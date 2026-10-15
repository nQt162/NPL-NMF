# Bộ câu hỏi và trả lời bảo vệ dự án NMF

> Đối chiếu code/web ngày **15-10-2026**. Bộ này gồm **60 câu hỏi: 15 dễ, 25 trung bình, 20 khó**. Câu trả lời dùng cách xưng "em" để báo cáo; có thể đổi thành "nhóm em".
> Mẫu mới: 12 câu AI, reference S1/S5/S8/S10, ngân sách 4 trên trang 02/03; PRF hỗ trợ cả KL và Frobenius. Ví dụ phụ có cấu hình khác được ghi riêng.
>
> Phân biệt ba mức khẳng định: **kiến thức nền tảng**, **hành vi code hiện tại**, và **hiệu quả đã được kiểm chứng**. Có chức năng chạy được không đồng nghĩa đã chứng minh chức năng đó tối ưu trên dữ liệu thực tế. Các hướng cải tiến trong phần khó chưa được coi là đã triển khai.

## Cách dùng

Đọc mức dễ để nắm hệ thống, mức trung bình để giải thích công thức, mức khó để xử lý phản biện. Khi trả lời, nói kết luận trước, giải thích cơ chế, rồi nêu giới hạn hoặc bằng chứng. Không cần đọc nguyên văn toàn bộ đoạn dài.

| Mức | Câu | Trọng tâm |
| --- | --- | --- |
| Dễ | 01-15 | Mục tiêu, token, TF-IDF, V/W/H, Local NMF, giao diện |
| Trung bình | 16-40 | Loss, solver, L1/L2, chọn k, relevance, MMR, PRF, ROUGE |
| Khó | 41-60 | Hạn chế auto-k, scale của factors, công bằng thí nghiệm, ablation, paper, bằng chứng |

Ưu tiên ôn trước: **07, 15, 16, 22, 23, 27, 31, 32, 39, 41, 46, 47, 50, 51, 55, 58, 60**.

## Cấu hình đang thực sự dùng

| Thành phần | Code hiện tại |
| --- | --- |
| Mẫu web | `test_nmf_01`: 12 câu nguồn, reference 4, index `[0,4,7,9]`; không ép chọn theo nhãn |
| Ngân sách mặc định | Trang 02/03: 4 câu; trang 04/API core: 3, cần sửa tay khi dùng cùng mẫu |
| Biểu diễn | TF-IDF cục bộ, mỗi câu là một document; unigram + bigram; tối đa 800 đặc trưng; chuẩn hóa L2 từng hàng |
| Tóm tắt | scikit-learn NMF; solver MU; init nndsvdar; tối đa 500 vòng |
| Hai phương pháp | Local KL-NMF + L1 + MMR và Local Frobenius-NMF + L1 + MMR |
| Regularization mặc định | alpha_W = alpha_H = 0.001; l1_ratio = 1 |
| MMR mặc định | lambda = 0.7 |
| Auto-k | k = 1..min(10, số câu, số đặc trưng); 10 lượt giữ lại khoảng 20% ô dương; fit KL không L1, tối đa 120 vòng mỗi lần |
| Mô phỏng | MU tự viết, khởi tạo ngẫu nhiên; cập nhật H rồi W |
| Tóm tắt qua sklearn | Bản MU đang cài cập nhật W rồi H; không phải trace của trang mô phỏng |
| Query PRF | Tùy chọn trên cả KL và Frobenius; không được dùng trong phép đánh giá hai loss và ablation hiện tại |
| ROUGE | ROUGE-1/2/L F1, token tiếng Việt, giữ dấu, giữ stopword, không English stemming |
| Ablation | Bốn cấu hình L1; cùng bài/seed/k/ngân sách/MMR trong mỗi cặp |

Ngày đối chiếu, ba file train/val/test và results/metrics.csv vẫn rỗng. Không được diễn giải ví dụ demo thành benchmark đã hoàn thành.

## Mức dễ

### 01. Dự án giải quyết bài toán gì?

**Trả lời:** Em xây dựng hệ thống tóm tắt trích xuất văn bản tiếng Việt. Hệ thống chọn các câu từ văn bản nguồn, giữ nguyên nội dung câu và sắp xếp lại theo thứ tự xuất hiện. NMF dùng để biểu diễn cấu trúc câu - thành phần tiềm ẩn; MMR dùng để chọn câu; ROUGE và ablation dùng để quan sát chất lượng đầu ra và tác động của L1.

**Cần phân biệt:** NMF không tự sinh ra câu tóm tắt. Một pipeline tóm tắt cần cả biểu diễn, chấm điểm và chọn câu.

### 02. NMF là gì? Có phải mạng neural hay ChatGPT không?

**Trả lời:** NMF là phân rã ma trận không âm: tìm W và H không âm sao cho tích WH xấp xỉ ma trận dữ liệu V. Nó là phương pháp biểu diễn và giảm chiều, không phải LLM sinh văn bản. Trong dự án, mỗi câu được biểu diễn bằng tổ hợp các thành phần NMF, sau đó thuật toán chọn câu gốc.

**Ví dụ:** Thay vì biểu diễn một câu bằng 158 đặc trưng, W có thể biểu diễn câu đó bằng 3 trọng số thành phần. H mô tả mỗi thành phần liên hệ với 158 đặc trưng như thế nào.

### 03. Tại sao yêu cầu W và H không âm?

**Trả lời:** TF-IDF vốn không âm. Khi W/H không âm, các đóng góp được cộng vào nhau để tái tạo dữ liệu, không dùng đóng góp âm để triệt tiêu đóng góp dương. Điều này giúp diễn giải một câu như tổ hợp của các thành phần từ vựng. Tuy nhiên, không âm không tự bảo đảm các thành phần rõ nghĩa hoặc không chồng chéo.

**Không nên nói:** "Không âm nên tất cả chủ đề đều đúng."

### 04. Từ văn bản đầu vào đến bản tóm tắt trải qua những bước nào?

**Trả lời:** Em tách câu và lưu thứ tự gốc, tiền xử lý tiếng Việt, fit TF-IDF trên các câu, nhận k người dùng nhập hoặc chạy auto-k, fit NMF cục bộ, tính relevance từ W, rồi dùng MMR để chọn đủ ngân sách câu. Nếu có query hợp lệ ở nhánh KL hoặc Frobenius, PRF bổ sung query relevance trước MMR. Cuối cùng, các câu được chọn được sắp theo index gốc và ghép lại.

**Chỉ trên web:** Trang 01 thấy V/W/H/WH; trang 02 thấy summary, keywords và trace chọn câu; trang 03 chấm hai outputs bằng reference.

### 05. Ai tạo unigram và bigram? NMF có tự tìm ra cụm từ không?

**Trả lời:** Không phải NMF tạo token. Underthesea tách từ tiếng Việt; vectorizer tạo đặc trưng unigram và bigram từ chuỗi token đã tiền xử lý. NMF chỉ học trọng số trên các đặc trưng đã có trong V. Một token tiếng Việt có thể gồm nhiều âm tiết được nối bằng dấu gạch dưới; bigram là hai token liên tiếp, không nhất thiết là hai âm tiết.

**Lưu ý:** Bigram được tạo sau khi bỏ stopword, nên hai token cạnh nhau trong đặc trưng có thể không liền nhau trong câu gốc.

### 06. TF-IDF có ý nghĩa gì? Trọng số cao có chắc là ý quan trọng không?

**Trả lời:** TF-IDF kết hợp mức xuất hiện của một đặc trưng trong câu với mức phổ biến của nó trong tập câu. Đặc trưng xuất hiện ở ít câu có thể nhận IDF cao. Vì vậy, trọng số TF-IDF cao thể hiện tính phân biệt trong biểu diễn, không tự đồng nghĩa với ý chính của văn bản. Trong dự án, các hàng còn được chuẩn hóa L2, nên không thể chỉ nhìn tổng số từ trong câu để suy ra điểm.

**Ví dụ:** Một tên riêng chỉ xuất hiện một lần có thể có IDF cao nhưng vẫn là chi tiết phụ.

### 07. Ý nghĩa và kích thước của V, W, H là gì?

**Trả lời:** Với n câu, m đặc trưng và k thành phần:

\[
V\in\mathbb{R}_{+}^{n\times m},\qquad
W\in\mathbb{R}_{+}^{n\times k},\qquad
H\in\mathbb{R}_{+}^{k\times m}.
\]

V là câu - đặc trưng. W là câu - thành phần. H là thành phần - đặc trưng. k là số cột W và số hàng H.

**Mẫu web hiện tại:** V là 12×321; nhập tay k=3 thì W là 12×3, H là 3×321; auto-k/seed 42 hiện chọn 1 thì W là 12×1, H là 1×321. WH luôn 12×321 trong các lần fit đó.

**Nếu bị hỏi về rank:** rank(WH) ≤ k. Yêu cầu k thành phần không bảo đảm rank thực sự bằng k, vì có thể có thành phần không hoạt động hoặc các thành phần phụ thuộc nhau.

### 08. WH được tính thế nào? Có phải nhân từng ô W với H không?

**Trả lời:** WH là phép nhân ma trận, không phải nhân từng ô tương ứng:

\[
(WH)_{ij}=\sum_{t=1}^{k}W_{it}H_{tj}.
\]

Ô tái tạo tại câu i, đặc trưng j cộng đóng góp của tất cả k thành phần. Sau đó so sánh nó với đúng ô V_ij.

**Ví dụ tính tay:** Hàng W_i = [0.2, 0.5], cột H_j = [0.4, 0.8] thì ô WH_ij = 0.2×0.4 + 0.5×0.8 = 0.48.

### 09. W và H có phải ma trận xác suất không?

**Trả lời:** Không mặc nhiên. Chúng là trọng số không âm; tổng hàng hoặc cột không bắt buộc bằng 1. Code chuẩn hóa một số đại lượng để tính relevance và keywords, nhưng việc đó không biến toàn bộ NMF thành mô hình xác suất được hiệu chỉnh. Ví dụ, W_it = 0.8 không tự có nghĩa "80% xác suất câu i thuộc chủ đề t".

**Khi bị hỏi sâu:** Scale giữa W/H có thể bù trừ; xem câu 50.

### 10. k có phải số câu tóm tắt không?

**Trả lời:** Không. k là số thành phần biểu diễn, summary_sentences là ngân sách câu. Mẫu hiện tại auto-k=1 nhưng vẫn giữ 4 câu; có thể nhập k=3 và giữ 4 câu. MMR không bắt buộc một câu cho mỗi thành phần.

**Chỉ trên web:** Hai ô "Số chủ đề k" và "Số câu tóm tắt" điều khiển hai bước khác nhau.

### 11. Topic 1 có tên cố định không? Ai đặt tên chủ đề?

**Trả lời:** NMF học các thành phần không có nhãn chủ đề giám sát. T1/T2/T3 chỉ là index của factors. Người đọc diễn giải chúng qua từ khóa và câu đại diện. Topic 1 ở một lần fit không nhất thiết tương ứng Topic 1 ở lần fit khác.

**Lưu ý:** Nhãn "nông nghiệp" do em diễn giải sau khi xem nội dung, không phải nhãn ground truth mà NMF đã được huấn luyện để dự đoán.

### 12. Local NMF khác Global NMF cũ thế nào?

**Trả lời:** Local NMF fit TF-IDF và W/H riêng trên các câu của văn bản đang nhập, nên vocabulary và factors được học từ chính văn bản đó. Global NMF cũ dùng một vocabulary và topics đã học từ corpus khác; khi transform văn bản mới có thể gặp OOV hoặc topic không sát nội dung. Global NMF không sai về nguyên tắc: nó hữu ích nếu cần không gian chủ đề chung cho nhiều tài liệu. Em chọn Local vì mục tiêu hiện tại là phân tích và tóm tắt từng văn bản.

**Code hiện tại:** Luồng tóm tắt web không load nmf_corpus.joblib. Việc bỏ model toàn cục không có nghĩa Underthesea cũng không dùng tài nguyên đã huấn luyện.

### 13. Hai phương pháp đang so sánh trên web khác nhau ở đâu?

**Trả lời:** Một nhánh dùng generalized KL, nhánh còn lại dùng Frobenius. Cả hai đều có L1 và MMR trong cấu hình hiện tại. Tên ngắn "Frobenius" không có nghĩa nhánh đó bỏ regularization. Khi so sánh trên trang 03, hai nhánh dùng cùng k, seed, alpha/L1, số câu và MMR.

**Không nên nói:** "Em đang so KL+L1+MMR với Frobenius thuần không L1/MMR."

### 14. Bốn trang web có vai trò gì?

**Trả lời:** Trang 01 / Mô phỏng NMF minh họa các vòng MU và loss. Trang 02 / Tóm tắt tạo summary và giải thích câu được chọn. Trang 03 / Đánh giá tạo hai outputs từ cùng nguồn rồi chấm ROUGE với reference. Trang 04 / Ablation L1 chạy bốn cấu hình penalty để kiểm tra riêng L1.

**Lưu ý:** Trang so sánh Global vs Local cũ đã bị bỏ; trang 04 hiện là ablation, không phải trang so sánh đó.

**Thao tác mới:** Trang 01/02/03 có sẵn bài AI; trang 03 có reference và Nạp ví dụ khôi phục nguồn/reference/ngân sách 4. Trang 04 vẫn nhập hai đoạn bằng tay, không upload JSON và không tự đổi sang4 câu.

### 15. Không có nhãn tóm tắt thì loss lấy từ đâu?

**Trả lời:** Loss NMF dùng V làm mục tiêu tái tạo: đo sự khác biệt giữa V và WH, cộng penalty trên W/H. Nó không cần nhãn câu nào phải có trong summary. Reference chỉ được dùng khi chấm ROUGE, không tham gia fit NMF hoặc MMR của luồng hiện tại.

**Câu ngắn dễ nhớ:** "V là dữ liệu cần tái tạo, không phải bản tóm tắt đúng."

**Về JSON mẫu:** `extractive_summary` là reference người dùng đưa, không phải prediction sẵn. Index/labels không đi vào fit NMF hay quyết định MMR; frontend chỉ dùng số index để khởi tạo ngân sách 4, không ép chọn S1/S5/S8/S10. Reference là gold của phép đánh giá này, không phải đáp án duy nhất hay nguồn xác minh thông tin của bài.

## Mức trung bình

### 16. Hàm loss hiện tại gồm những phần nào?

**Trả lời:** Khi l1_ratio=1, objective gồm data loss và L1:

\[
L(W,H)=D(V,WH)
+\alpha_W m\sum_{i,t}W_{it}
+\alpha_H n\sum_{t,j}H_{tj}.
\]

Ở nhánh KL, với y=(WH)_ij:

\[
d(x,y)=x\log(x/y)-x+y,\qquad d(0,y)=y.
\]

Ở nhánh Frobenius:

\[
D_F(V,WH)=\frac12\sum_{i,j}(V_{ij}-(WH)_{ij})^2.
\]

Code dùng epsilon để tránh chia cho 0/log 0. "KL data" và "L1 penalty" là hai phần; "objective" là tổng. Khi l1_ratio nhỏ hơn 1, còn có phần L2.

**Cơ sở:** Công thức regularization và scaling được mô tả trong [API NMF 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.NMF.html). Công thức của dự án nằm trong objective_breakdown.

### 17. Vì sao dùng KL thay Frobenius? KL có luôn tốt hơn không?

**Trả lời:** Frobenius dùng sai số bình phương theo mức chênh lệch tuyệt đối. KL có thêm log của tỷ lệ thật/dự đoán, nên phản ứng khác khi một ô thật dương bị dự đoán quá thấp. Hai losses tạo ra hai cách fit dữ liệu, không có luật KL luôn cho summary tốt hơn. TF-IDF là dữ liệu đã được biến đổi và chuẩn hóa, không tự trở thành dữ liệu đếm Poisson chỉ vì dùng KL.

**Ví dụ tính tay:** Với x=0.5, y=0.2, KL khoảng 0.158145. Với y=0.001, KL khoảng 2.608304. Đây là minh họa độ nhạy, không chứng minh chất lượng tóm tắt.

**Nguồn nền tảng:** [Lee và Seung, hai thuật toán NMF với least squares và KL](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf).

### 18. Solver MU là gì? Loss có tự cập nhật W/H không?

**Trả lời:** Loss định nghĩa mục tiêu; solver là cách tìm W/H để tối ưu mục tiêu đó. MU là Multiplicative Update: trọng số hiện tại được nhân với một tỷ lệ tính từ dữ liệu, reconstruction và regularization. Tỷ lệ dưới 1 làm giảm trọng số; trên 1 làm tăng. Nhánh KL của sklearn cần dùng solver MU trong các solver NMF hiện có; MU cũng dùng được với Frobenius.

**Phân biệt:** "Dùng KL" là chọn objective; "dùng MU" là chọn cách tối ưu. Không gọi MU là hàm loss.

### 19. Cập nhật H rồi W có nghĩa tạo lại H từ đầu không?

**Trả lời:** Không. Trong custom MU, mỗi vòng giữ W hiện tại để cập nhật H một lượt, rồi dùng H mới để cập nhật W một lượt. V giữ nguyên. W/H chỉ được khởi tạo ở đầu một lần fit, không bị bốc ngẫu nhiên lại ở từng vòng. Cũng không tối ưu H đến hoàn chỉnh rồi mới bắt đầu W.

**Đúng với code hiện tại:** Custom MU ở trang 01 và bộ chọn k cập nhật H trước W. Sklearn MU đang cài, dùng trong tóm tắt, cập nhật W trước H. Cả hai đều luân phiên; không có yêu cầu mọi implementation phải cùng thứ tự.

### 20. Vì sao mô phỏng và tóm tắt không có cùng loss cuối?

**Trả lời:** Chúng dùng cùng biểu diễn và có thể cùng objective, nhưng không cùng đường tối ưu. Mô phỏng khởi tạo ngẫu nhiên và chạy custom MU với số vòng người dùng chọn. Tóm tắt dùng sklearn NMF với init nndsvdar, tối đa 500 vòng và điều kiện dừng riêng. Nndsvdar lấy cấu trúc từ NNDSVD rồi lấp các vị trí zero bằng giá trị ngẫu nhiên nhỏ. Vì vậy, khác loss cuối là có thể xảy ra.

**Cần nói rõ:** Trang mô phỏng là minh họa thuật toán, không phải log nội bộ của lần fit sklearn tạo summary. Khởi tạo có giá trị dương giúp tránh zero locking ngay từ đầu; không bảo đảm mọi hệ số luôn dương về sau.

### 21. W/H thưa có ý nghĩa gì? Càng thưa càng tốt không?

**Trả lời:** W thưa nghĩa là mỗi câu có ít thành phần đóng góp đáng kể. H thưa nghĩa là mỗi thành phần có ít đặc trưng hoạt động đáng kể. Biểu diễn có thể gọn và dễ đọc hơn, nhưng quá thưa sẽ mất thông tin, đặc biệt khi câu có nhiều nội dung hoặc các chủ đề có từ vựng chung. Độ thưa là thuộc tính của factors, không phải bằng chứng summary tốt.

**Chỉ trên web:** Trang 04 có tỷ lệ W/H gần 0, số thành phần hoạt động mỗi câu và số đặc trưng hoạt động mỗi thành phần. Code dùng ngưỡng tuyệt đối 1e-8; đây là quy ước thống kê, không phải nhãn semantic.

### 22. L1 lấy cơ sở gì để phạt? Nó biết từ nào nhiễu bằng cách nào?

**Trả lời:** L1 không tự biết từ nào nhiễu. Nó phạt tổng trị tuyệt đối của các hệ số; trong NMF không âm, đó là tổng trọng số. Solver cân bằng lợi ích tái tạo với chi phí sử dụng hệ số. Trọng số không đem lại đủ cải thiện về data loss có xu hướng bị co nhỏ; nhưng không có cơ chế nhìn nhãn "từ nhiễu" rồi xóa từ đó.

**Ví dụ tính tay:** Giả sử hệ số phạt hiệu dụng là 0.1. Bỏ một hệ số 0.02 làm penalty giảm 0.002. Nếu data loss chỉ tăng 0.001 thì tổng loss giảm 0.001. Nếu data loss tăng 0.01 thì bỏ hệ số đó lại không có lợi. Đây là so sánh objective giả định, không phải code thử xóa từng hệ số theo bảng.

### 23. Làm sao biết trọng số lớn và cần giảm? Có ngưỡng cố định không?

**Trả lời:** Không có ngưỡng "lớn hơn X thì phải giảm". Một cột W gồm nhiều trọng số của một thành phần trong các câu. Custom MU cập nhật từng hệ số dựa vào dữ liệu và penalty. Với KL và pure L1, công thức không mask, giản lược epsilon, là:

\[
W_{it}\leftarrow W_{it}
\frac{\sum_j H_{tj}V_{ij}/(WH)_{ij}}
{\sum_j H_{tj}+\alpha_W m}.
\]

Tỷ lệ nhỏ hơn 1 thì hệ số giảm, lớn hơn 1 thì tăng. L1 làm tăng mẫu số, tạo áp lực giảm; dữ liệu có thể tạo áp lực tăng.

**Ví dụ:** W_it=0.8, tỷ lệ 0.6 thì thành 0.48; cùng trọng số 0.8 nhưng tỷ lệ 1.2 thì thành 0.96. Vì vậy, "trọng số lớn" không đồng nghĩa "trọng số sai".

### 24. L1 và L2 khác nhau thế nào? L1 có bảo đảm chính xác bằng 0 không?

**Trả lời:** L1 phạt theo trị tuyệt đối, L2 phạt theo bình phương. Khi xét một hệ số dương x, độ dốc penalty L1 là hằng số theo x; độ dốc L2 tỷ lệ với x. L1 thường thuận lợi hơn cho nghiệm thưa; L2 thường tạo shrinkage phân tán hơn. Tuy nhiên, với MU, hệ số được nhân với tỷ lệ, nên có thể chỉ rất gần 0 thay vì chính xác bằng 0. Ràng buộc không âm và xử lý số học cũng ảnh hưởng kết quả.

**Phản ví dụ cần nhớ:** [1,0,0] và [0.5,0.3,0.2] đều có L1 bằng 1, dù số hệ số khác 0 khác nhau. L1 không trực tiếp đếm nonzero. Cũng không nên khẳng định L2 tuyệt đối không bao giờ tạo zero.

### 25. Vì sao chọn alpha=0.001 và max_iter=500? Đã tối ưu tham số chưa?

**Trả lời:** Đây là cấu hình mặc định bảo thủ để tránh penalty quá mạnh trên TF-IDF cục bộ. Max_iter=500 là giới hạn số vòng cho sklearn, không bắt buộc chạy đủ 500 vòng và không bảo đảm hội tụ tới nghiệm toàn cục. Hiện chưa có corpus validation hợp lệ trong các split để chứng minh 0.001 là alpha tối ưu.

**Không nên nói:** "Paper chứng minh alpha=0.001 tốt nhất cho tiếng Việt." Muốn khẳng định cần tuning trên validation, sau đó báo test độc lập.

### 26. Nếu tăng L1 mạnh thì chuyện gì xảy ra? Fallback là gì?

**Trả lời:** Tăng alpha làm chi phí giữ weights lớn hơn. Nó có thể làm factors gọn hơn, nhưng cũng có thể làm W hoặc H sụp về gần 0, mất khả năng chấm điểm hữu ích. Code có kiểm tra factors không còn giá trị vượt EPS và trả fallback lấy các câu đầu, kèm lý do. Đây là cơ chế dự phòng, không được gọi output đó là thành công của NMF+MMR.

**Chỉ trên web:** Xem fallback_reason/cảnh báo ở kết quả. Khi chạy ablation, phải báo cả số lần fallback chứ không chỉ ROUGE của những lần thành công.

### 27. Thuật toán tự chọn k hiện tại chính xác làm gì?

**Trả lời:** Khi để trống k, code thử các k từ 1 đến min(10,n,m). Mỗi lượt chọn ngẫu nhiên khoảng 20% vị trí có V>0, đặt training mask ở các vị trí đó bằng 0, rồi fit KL không regularization trên các ô còn lại. Sau fit, code tính mean KL giữa giá trị gốc và giá trị WH tại các ô giữ lại. Điểm của k là trung bình qua 10 lượt; k có điểm thấp nhất được chọn. Nếu điểm bằng nhau chính xác, ưu tiên k nhỏ hơn.

**Chi tiết có thể bị hỏi:** Mỗi fit tối đa 120 vòng; cùng mask cho mọi k trong một lượt; seed fit = seed + trial×997 + k. Bộ chọn không dùng reference, MMR hoặc ROUGE.

**Giới hạn tên gọi:** Đây là positive-entry masked KL CV của dự án, không phải tự động tìm số chủ đề ngữ nghĩa đúng.

### 28. Che ô có phải thay giá trị thật bằng 0 để huấn luyện không?

**Trả lời:** Không. Zero thật nghĩa là đặc trưng không xuất hiện trong câu; ô bị che nghĩa là tạm không được dùng khi fit. Code giữ V gốc nhưng dùng mask để loại ô bị che khỏi data objective và các biểu thức cập nhật có mask. Chỉ khi validation mới đọc lại giá trị thật tại đó để chấm prediction.

**Ví dụ:** V_ij=0.5 bị che thì mô hình không được thưởng vì dự đoán nó bằng 0. Nếu sửa thành 0 và vẫn tính ô đó trong training, ta đang huấn luyện trên nhãn sai của ô chứ không kiểm tra imputation.

### 29. Số vòng mô phỏng có thay đổi cách chọn k không?

**Trả lời:** Không trong code hiện tại. Bộ chọn k có cấu hình riêng: 10 trial và tối đa 120 vòng mỗi fit. Số vòng trên form mô phỏng, mặc định 10 và giới hạn API 20, điều khiển lần fit minh họa sau khi k đã được chọn. Tóm tắt dùng sklearn với max_iter=500. Vì vậy, không đọc slider "10 vòng" thành "auto-k chỉ học 10 vòng".

**Chỉ trên web:** Panel Ứng viên k thuộc lượt validation riêng; loss theo slider thuộc lần fit mô phỏng.

### 30. Tại sao bấm nhiều lần vẫn cùng loss và k? Có phải kết quả được làm sẵn?

**Trả lời:** Cùng text, cấu hình, seed và môi trường thường tạo kết quả tái lập. Trang tóm tắt dùng seed core mặc định 42; schema SummarizeRequest hiện không có trường seed. Mô phỏng, đánh giá và ablation có điều khiển seed theo schema riêng. Việc không đổi kết quả mỗi lần bấm không chứng minh có cache hoặc kết quả giả.

**Cách kiểm tra:** Đổi văn bản/k/alpha và xem request, response thật. Đổi seed ở trang có hỗ trợ để khảo sát độ nhạy, không chọn riêng seed cho kết quả đẹp.

### 31. Relevance của câu được tính từ W như thế nào?

**Trả lời:** Code tính mức phổ biến của thành phần bằng tổng cột W, rồi chuẩn hóa:

\[
s_t=\frac{\sum_i W_{it}}{\sum_{i,u}W_{iu}},\qquad
P_{it}=\frac{W_{it}}{\sum_u W_{iu}}.
\]

Relevance thô của câu i là:

\[
r_i=\sum_t P_{it}s_t.
\]

Sau đó chia cho relevance lớn nhất để chuẩn hóa về thang dùng cho MMR. Code có epsilon và nhánh dự phòng theo norm V nếu không tính được relevance hữu ích.

**Ví dụ tính tay:** W=[[4.5,0.5],[0.4,1.6]] cho s=[0.7,0.3]. Hai phân bố câu là [0.9,0.1] và [0.2,0.8], nên relevance thô là 0.66 và 0.38; sau chuẩn hóa là 1 và khoảng 0.575758.

**Giới hạn:** Đây là lựa chọn scoring của dự án, không phải công thức bắt buộc của mọi hệ thống NMF.

### 32. MMR chấm và phạt điểm câu như thế nào?

**Trả lời:** Ở mỗi vòng, với S là tập câu đã chọn:

\[
\mathrm{MMR}(i\mid S)
=\lambda r_i-(1-\lambda)\max_{j\in S}\cos(V_i,V_j).
\]

Code chọn câu chưa được chọn có điểm cao nhất. Nếu S rỗng, redundancy=0. Lambda=1 bỏ phạt trùng lặp; lambda=0 chỉ ưu tiên ít giống các câu đã chọn, và vòng đầu vẫn cần quy tắc phá hòa. Mặc định dự án là 0.7.

**Liên hệ nghiên cứu:** Công thức lựa chọn cân bằng relevance với novelty có nền tảng từ [Carbonell và Goldstein](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf); relevance cụ thể trong dự án lấy từ factors, không hoàn toàn là query similarity của paper gốc.

### 33. Redundancy chỉ so với câu vừa chọn hoặc câu cùng topic phải không?

**Trả lời:** Không. Nó lấy cosine lớn nhất với tất cả câu trong S, không lọc theo dominant topic và không chỉ so câu vừa được chọn. Nếu một câu giống mạnh với bất kỳ câu đã chọn nào thì có thể bị phạt. Hai câu cùng chủ đề nhưng cung cấp chi tiết khác nhau vẫn có thể cùng được giữ nếu similarity không quá cao và relevance đủ tốt.

**Ví dụ:** S gồm câu 1 và câu 4; câu 6 có cosine với chúng là 0.2 và 0.8 thì redundancy của câu 6 là 0.8.

### 34. Câu relevance cao nhất có luôn được chọn không? Cho ví dụ MMR.

**Trả lời:** Không sau khi đã có câu được chọn. Giả sử C đã nằm trong S, lambda=0.7. Hai ứng viên A và B có:

| Ứng viên | Relevance | Redundancy với S | MMR |
| --- | ---: | ---: | ---: |
| A | 0.9 | 0.9 | 0.7×0.9 - 0.3×0.9 = 0.36 |
| B | 0.8 | 0.2 | 0.7×0.8 - 0.3×0.2 = 0.50 |

B được chọn vì mang thông tin ít trùng hơn, dù relevance thấp hơn. Đây là ví dụ tính tay minh họa, không phải số chấm từ một văn bản thật.

### 35. Vì sao điểm MMR thay đổi theo vòng? Có thể so các số trên mọi câu trực tiếp không?

**Trả lời:** Tập S thay đổi sau mỗi lần chọn, nên redundancy và MMR của các câu còn lại được tính lại. Chỉ nên so điểm giữa các ứng viên trong cùng vòng. Trong sentence_analysis, câu đã chọn giữ điểm ở vòng nó được chọn; câu chưa chọn giữ điểm ở vòng cuối mà nó đã được xét. Vì vậy, một bảng cuối có thể chứa score_round khác nhau.

**Khi hòa điểm:** Code ưu tiên score cao, rồi relevance cao, rồi index gốc nhỏ. Các số bằng nhau sau làm tròn trên UI chưa chắc bằng nhau ở full precision. Muốn giải thích quyết định, xem selection_trace.

### 36. Nếu mọi điểm MMR âm thì có chọn câu không? Có lấy đúng top_n không?

**Trả lời:** Có. Code chọn ứng viên có điểm cao nhất trong các câu còn lại, kể cả khi điểm cao nhất là âm. Ngân sách là min(số câu yêu cầu, số câu nguồn); không có threshold cố định làm mất bớt câu trong luồng MMR hiện tại. Sau đó sort index gốc để summary giữ trình tự văn bản.

**Giới hạn:** Đủ số câu không có nghĩa mọi câu đều hữu ích. Ngân sách quá lớn có thể buộc giữ chi tiết phụ. MMR cũng không bảo đảm phân bố một câu cho mỗi chủ đề.

### 37. Query và PRF hoạt động thế nào?

**Trả lời:** Query được tiền xử lý rồi transform bằng chính vectorizer của bài. Code lấy tối đa ba câu có similarity dương với query, trong giới hạn ngân sách, làm tập pseudo-relevant. Centroid của các câu đó được trộn 0.5 với query gốc và chuẩn hóa để tính query relevance mới. Sau đó query_weight trộn query relevance với topic relevance trước MMR.

**Phân biệt hai hệ số:** Tỷ lệ 0.5 khi mở rộng query và query_weight mặc định 0.5 khi trộn relevance là hai thao tác khác nhau.

**Demo hiện tại:** Frobenius/auto-k=1/query `ung thư phổi`/weight 0.5 lấy S8 làm pseudo-relevant, đổi summary từ S2/S3/S7/S10 sang S2/S3/S4/S8, objective vẫn 5.809127. KL cùng query có thể giữ nguyên tập câu vì đã có S8; có PRF không bắt buộc đổi output.

**Phạm vi và giới hạn:** Query PRF hỗ trợ cả KL và Frobenius vì đây là bước bổ sung relevance sau phân rã, không phụ thuộc riêng một loss. Query OOV hoặc không còn token hữu ích được báo lý do và bỏ phần query, không tự tạo từ vựng mới.

### 38. ROUGE-1, ROUGE-2 và ROUGE-L đo gì?

**Trả lời:** ROUGE-1 đo overlap unigram, ROUGE-2 đo overlap bigram giữa candidate và reference, có xét số lần xuất hiện. ROUGE-L dùng độ dài longest common subsequence, không bắt buộc các token khớp phải liền nhau nhưng phải giữ thứ tự. Dự án trả F1 của cả ba, không phải chỉ recall.

**Triển khai tiếng Việt:** Underthesea tách từ, lowercase và giữ dấu; không bỏ stopword ở bước chấm ROUGE, không dùng English stemming. Bigram của ROUGE được tạo từ token đánh giá, không phải lấy trực tiếp cột bigram trong TF-IDF.

**Nguồn:** [Lin, ROUGE](https://aclanthology.org/W04-1013/). Hành vi F1/tokenizer cụ thể được kiểm tra tại evaluate.py.

### 39. Precision, recall và F1 được tính thế nào? Cho ví dụ.

**Trả lời:** Với overlap C, số đơn vị trong candidate M và trong reference N:

\[
P=C/M,\qquad R=C/N,\qquad F_1=\frac{2PR}{P+R}.
\]

Ví dụ tính tay trên token đã xác định: reference=[a,b,c,d], candidate=[a,b,e].

| Metric | Số khớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| ROUGE-1 | 2 | 2/3 | 2/4 | 4/7 ≈ 0.571429 |
| ROUGE-2 | 1 bigram: a b | 1/2 | 1/3 | 0.4 |
| ROUGE-L | LCS dài 2 | 2/3 | 2/4 | 4/7 ≈ 0.571429 |

Đây là ví dụ trên chuỗi token tính tay, không giả định Underthesea sẽ tách một câu tiếng Việt bất kỳ giống danh sách này. Điểm 0.57 là một số trên thang 0..1, không phải chứng nhận 57% câu đúng.

### 40. Reference là ground truth phải không? Có lấy văn bản nguồn làm reference được không?

**Trả lời:** Reference là mục tiêu tham chiếu của phép đánh giá, có thể coi là ground truth theo nghĩa nhãn đánh giá do người viết xác định. Nhưng tóm tắt có nhiều cách hợp lý; một reference không phải chân lý duy nhất. Văn bản nguồn chứa mọi chi tiết nên không thay thế tốt cho một reference ngắn xác định ý cần giữ. Code đánh giá web hiện dùng một reference cho mỗi bài.

**Nguyên tắc:** Viết reference trước khi xem outputs nếu có thể; dùng cùng reference để chấm mọi phương pháp. Reference demo tự viết chỉ hỗ trợ pilot, không tương đương bộ dữ liệu benchmark được gán nhãn độc lập.

## Mức khó

### 41. Nhìn văn bản rõ ràng có ba chủ đề, tại sao auto-k vẫn chọn 1? Thuật toán sai không?

**Trả lời:** Em phân biệt số mảng nội dung theo cách đọc với rank được chọn theo một tiêu chí. Code có thể chọn đúng argmin KL validation nhưng tiêu chí đó chưa phù hợp để xác định số chủ đề ngữ nghĩa. k=1 có thể trộn từ vựng của nhiều mảng vào một thành phần tổng quát. Đây là hạn chế của bộ chọn rank trên những văn bản đã thử, không phải bằng chứng văn bản chỉ có một ý.

**Ví dụ phụ đã kiểm tra, không phải mẫu AI hiện tại:** Mẫu ba mảng nông nghiệp/y tế/an ninh mạng có V là 12×158 và 131 đặc trưng chỉ xuất hiện trong một câu. Trong 10 trial, 278/390 ô giữ lại không còn quan sát dương trong cột sau khi che. Những ô không thể học được này đóng góp loss lớn nhưng ít thông tin phân biệt k. Với các ô còn lại, biểu diễn dàn trải của k=1 có thể dự đoán các giá trị dương tốt hơn; validation hiện không chấm ô zero để cân đối các dự đoán dàn trải.

**Điểm cần nói chính xác:** Các cột không còn dữ liệu dương không tự làm k=1 thắng; chúng chủ yếu làm phép kiểm tra kém thông tin. Không được gộp tất cả nguyên nhân thành "từ xuất hiện một lần nên 1 chắc chắn tốt nhất".

### 42. Có phải code đặt cứng k=1 hoặc luôn chọn mô hình đơn giản nhất?

**Trả lời:** Không. Code thử danh sách các k khả thi và sort theo score, chỉ ưu tiên k nhỏ khi score bằng nhau chính xác. Em đã kiểm tra bằng ma trận tạo từ W×H có ba thành phần: selector có thể chọn k=3. Ma trận một thành phần có thể chọn 1. Mẫu văn bản lặp rõ ba nhóm ở phụ lục cũng được API chọn 3. Những phép thử này bác bỏ việc đầu ra bị đặt cứng, nhưng chưa chứng minh chất lượng trên corpus văn bản tự nhiên.

**Nếu giảng viên nói văn bản dài vẫn chọn 1:** Độ dài không thay thế được mức lặp và cấu trúc của những đặc trưng có thể học khi bị che. Em cần khảo sát nhiều bài và kiểm tra protocol, không chỉ giải thích mọi kết quả 1 là đúng.

### 43. Có một mẫu chọn k=3 thì đã chứng minh auto-k tốt chưa?

**Trả lời:** Chưa. Mẫu được thiết kế có ba nhóm từ vựng lặp lại nhiều, giúp mô hình còn quan sát để khôi phục ô bị che. Nó là ca minh họa chức năng và kiểm tra rằng selector có thể trả 3, không đại diện cho phân bố bài báo thực tế. Em phải công bố cách tạo mẫu, không dùng nó để khẳng định độ chính xác của auto-k.

**Số liệu mẫu phụ lục:** k=3 có score 0.263495, nhưng k=7 có 0.264881, khá gần nhau. Code hiện chưa trả độ lệch chuẩn score của k, nên không được nói khoảng cách đó chứng minh chắc chắn đúng ba thành phần. Chọn riêng một mẫu hay seed để minh họa phải được ghi là lựa chọn có chủ đích.

### 44. Vậy nhập tay k=3 có hợp lệ không? Có phải sửa kết quả cho đẹp?

**Trả lời:** Hợp lệ nếu công bố mục đích. Auto-k=1 là kết quả theo protocol validation; k=3 nhập tay là cấu hình em chọn để khảo sát biểu diễn chi tiết hơn hoặc kiểm soát phép so sánh. Em không được đổi nhãn kết quả nhập tay thành "k tối ưu do hệ thống tìm được". Khi so hai losses với k=3, cả hai phải cùng dùng 3 và báo rõ đó là fixed-k.

**Câu trả lời nói được ngay:** "Em cố định k=3 để phân tích ba thành phần; đây không phải khẳng định auto-k đã xác định đúng ba chủ đề."

### 45. Tỷ lệ che 20% có phải chuẩn bắt buộc? Làm sao biết kết quả chọn k đáng tin?

**Trả lời:** Không có tỷ lệ 20% bắt buộc cho mọi bài toán NMF. Trong dự án, đó là hyperparameter của repeated holdout trên ô dương, không phải 20% câu và cũng không phải 20% toàn V. Để đánh giá độ tin cậy, em cần khảo sát tỷ lệ giữ lại, mức quan sát còn lại trong từng hàng/cột, nhiều initialization, biến động score và ảnh hưởng đến downstream summary.

**Hạn chế hiện tại:** 10 trial dùng 10 lần fit với initialization khác nhau cho mỗi k, nhưng không có nhiều restart để chọn nghiệm tốt trong từng mask; mỗi fit chỉ tối đa 120 vòng. Nhiều trial không chữa được một protocol lấy mẫu thiếu thông tin.

**Hướng cải tiến, chưa triển khai:** Giữ đủ quan sát training, công bố những ô không đủ điều kiện validation, thử kiểm tra cả zero với trọng số rõ ràng, chọn restart theo training objective, và trả mean/SD. Cách điều chỉnh này vẫn không tự đồng nhất với BCV của paper.

### 46. Vì sao chọn k bằng KL không L1, kể cả khi final model dùng Frobenius hoặc có L1?

**Trả lời:** Đây là chính sách chung để kiểm soát k giữa các nhánh trong phép so hiện tại. Bước chọn rank dùng một tiêu chí cố định, sau đó final model có thể dùng KL hoặc Frobenius và penalty người dùng cấu hình. Do đó, thay alpha trên form không trực tiếp đổi score auto-k nếu V và seed không đổi.

**Giới hạn:** k được chọn theo KL không phải k đã được tối ưu riêng cho Frobenius hay cho ROUGE. Nếu muốn đánh giá hiệu quả tốt nhất của từng phương pháp, cần một protocol tuning có ngân sách tương đương và test độc lập. Không được gọi chính sách chung này là tối ưu mọi mục tiêu cùng lúc.

### 47. Objective KL nhỏ hơn hay lớn hơn Frobenius thì nhánh nào tốt hơn?

**Trả lời:** Không thể kết luận từ trị số raw. Hai hàm đo khác nhau, không cùng đơn vị hoặc scale. Chia cả hai cho số ô chỉ đổi tổng thành trung bình, không biến chúng thành cùng một thước đo. Cũng không được so loss của hai văn bản khác nhau rồi suy ra văn bản nào được tóm tắt tốt hơn.

**Cách so đúng:** Với mỗi objective, theo dõi tiến triển trong cùng điều kiện. Để so outputs, dùng cùng nguồn/reference/ngân sách và chấm ROUGE, đọc nội dung, kiểm tra trùng lặp và tính mạch lạc. Panel Ứng viên k là mean held-out KL, khác objective final fit trên toàn V cộng penalty.

### 48. Hai nhánh cùng alpha và MMR lambda đã là so sánh công bằng hoàn toàn chưa?

**Trả lời:** Đó là so sánh có kiểm soát, chưa phải so hiệu quả tốt nhất của hai phương pháp sau tuning. Cùng alpha không bảo đảm penalty có cùng sức nặng tương đối vì scale KL và Frobenius khác nhau. Một nhánh có thể bị regularization mạnh hơn tương đối so với data loss.

**Thiết kế chặt hơn:** Báo rõ phép so fixed-configuration để cô lập thay đổi loss. Nếu đánh giá method sau tuning, cho mỗi nhánh cùng ngân sách tìm tham số trên validation, rồi giữ cấu hình để chấm test. Trang 03 hiện tại vẫn tắt query ở cả hai nhánh; nếu mở so sánh query-focused sau này, phải dùng cùng truy vấn, cấu hình PRF và trọng số cho cả hai.

### 49. Loss giảm đều và dừng hội tụ có chứng minh tìm được nghiệm tối ưu toàn cục không?

**Trả lời:** Không. W và H được tối ưu cùng nhau tạo bài toán không lồi; điều kiện dừng số học không chứng nhận nghiệm toàn cục. Một lần fit có thể dừng ở nghiệm cục bộ hoặc vùng thay đổi objective rất nhỏ. Tăng max_iter có thể giúp nhưng không bảo đảm tìm được global optimum.

**Phân biệt nghiên cứu với implementation:** [Lee và Seung](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf) phân tích tính không tăng của objective cho các quy tắc MU tương ứng trong điều kiện của bài. Điều đó không tự chứng minh mọi sửa đổi penalty/mask/epsilon và tiêu chí dừng của code đều có cùng bảo đảm.

**Kiểm chứng thực dụng:** Theo dõi loss, thử nhiều khởi tạo, kiểm tra finite/nonnegative và so kết quả. Đây là kiểm chứng, không thay thế chứng minh hội tụ.

### 50. Tại sao W/H không có scale duy nhất? Hệ số lớn có thể chỉ do đổi scale không?

**Trả lời:** Có. Với c>0:

\[
W'=cW,\qquad H'=H/c\quad\Rightarrow\quad W'H'=WH.
\]

Tổng quát hơn, có thể rescale riêng mỗi thành phần bằng ma trận chéo dương. Reconstruction giữ nguyên nhưng các weights thô thay đổi. Đổi thứ tự thành phần đồng thời trong W/H cũng không đổi WH. Vì vậy, không diễn giải trị số W/H như xác suất hay độ quan trọng tuyệt đối mà không xét quy ước scale.

**L1 ảnh hưởng ra sao:** Khi phạt cả hai factors, dọc theo phép rescale toàn bộ, penalty có dạng a×c + b/c với a,b>0 và có mức cân bằng. Điều này không chứng minh factorization duy nhất. Nếu chỉ phạt một factor, có thể giảm penalty bằng rescale nó nhỏ đi và factor còn lại lớn lên; xem câu 56.

### 51. Relevance hiện tại có hạn chế gì khi k=1 hoặc khi factors đổi scale?

**Trả lời:** Khi k=1 và W_i1 dương, chuẩn hóa từng hàng làm P_i xấp xỉ [1], còn salience xấp xỉ [1]. Relevance các câu vì vậy gần bằng nhau, dù weights thô khác nhau. Không query thì MMR dễ chọn câu đầu theo tie-break, sau đó ưu tiên ít trùng với tập đã chọn. Đây là hạn chế riêng của scoring, ngoài hạn chế auto-k.

**Một hạn chế sâu hơn:** Rescale từng thành phần của W và bù trong H giữ WH không đổi, nhưng có thể thay salience và P, nên relevance hiện tại không bất biến theo mọi phép rescale factor. Dominant topic và keywords cũng cần được diễn giải thận trọng.

**Hướng sửa, chưa triển khai:** Quy ước scale thành phần trước scoring bằng cách chuẩn hóa H và bù W; thiết kế relevance vẫn phân biệt câu ở k=1 hoặc bổ sung sentence centrality/coverage. Cần kiểm chứng chất lượng, không chỉ làm điểm nhìn đẹp hơn.

**Khi bị hỏi về độ thưa và relevance:** Relevance cao không chứng minh câu có nhiều thông tin hơn theo nghĩa con người; nó thể hiện độ phù hợp với cách định nghĩa salience trong code.

### 52. MMR có bảo đảm summary tối ưu toàn cục, đủ mọi chủ đề hoặc không trùng ý không?

**Trả lời:** Không. Đây là lựa chọn tham lam: ở mỗi vòng lấy ứng viên có điểm cao nhất theo tập đã chọn hiện tại. Nó không duyệt tất cả tập con câu để chứng minh một summary tối ưu toàn cục. MMR hiện không có quota một câu mỗi topic; cosine TF-IDF cũng không hiểu đầy đủ paraphrase hay phủ định.

**Phản ví dụ về similarity:** Hai câu cùng từ vựng nhưng một câu phủ định ý kia có thể có cosine cao; hai câu dùng từ đồng nghĩa khác nhau có thể cosine thấp nhưng trùng ý.

**Cũng cần phân biệt:** Điểm MMR cao không quyết định KL hay Frobenius là loss tốt hơn. Nó phụ thuộc relevance, lambda, cách chuẩn hóa và vòng chọn.

### 53. PRF có hiểu ý định người dùng hoặc tự tìm kiến thức ngoài văn bản không?

**Trả lời:** Không trong implementation hiện tại. PRF lấy một số câu có similarity dương rồi tạm coi chúng liên quan để mở rộng vector query. Nó không tìm kiếm Internet, không gọi LLM và không có người gán nhãn xác nhận. Chọn nhầm câu có thể làm query drift, tức truy vấn mở rộng lệch khỏi ý định ban đầu.

**Điểm dễ bị hỏi:** Query làm thay đổi relevance/MMR sau fit, không thay V, k hay W/H trong code hiện tại. Vì vậy, query có thể đổi summary mà loss factorization không đổi. Query dùng từ ngoài vocabulary không được tự hiểu bằng embedding; code báo lý do bỏ PRF.

### 54. ROUGE cao có chắc summary đúng, đủ ý và mạch lạc không?

**Trả lời:** Không. Nó chấm overlap theo reference và tokenizer, không trực tiếp xác minh sự thật, phủ định, quan hệ nhân quả hay mạch lạc. Ví dụ "Dự án chưa được phê duyệt" và "Dự án được phê duyệt" có nhiều token chung nhưng ý nghĩa khác nhau. Một paraphrase đúng cũng có thể được chấm thấp nếu dùng từ khác reference.

**Riêng hệ thống trích xuất:** Giữ câu nguồn giảm nguy cơ tự bịa câu mới, nhưng bỏ mất câu giải thích, antecedent của đại từ hoặc ngữ cảnh thời gian vẫn có thể khiến summary gây hiểu nhầm.

**Đánh giá bổ sung:** Đọc coverage, factual consistency với nguồn, redundancy, coherence; dùng nhiều reference hoặc đánh giá người đọc khi có điều kiện. Đó là hướng bổ sung, không phải tính năng đã đo đầy đủ trong web hiện tại.

### 55. Ablation L1 đang kiểm tra gì? Đọc delta và mean ± SD thế nào?

**Trả lời:** Bốn cấu hình là KL không L1, L1 trên W, L1 trên H và L1 trên cả W/H. Trong mỗi bài/seed, chọn k một lần rồi dùng lại, giữ TF-IDF/ngân sách/MMR như nhau và tắt query. Em xem thay đổi ROUGE và thống kê factors, không dùng raw objectives có penalty khác nhau để xếp hạng.

**Delta:** Là chênh lệch ROUGE của cùng ID bài và seed so với nhánh không L1, chỉ khi hai nhánh đều thành công. Delta dương là cải thiện trên những cặp đã đánh giá, không tự chứng minh tổng quát.

**Mean và SD:** Code hiện dùng trung bình và độ lệch chuẩn population của các lượt thành công; SD=0 với một lượt là hệ quả số lượng mẫu, không phải bằng chứng tuyệt đối ổn định. Fallback và lỗi được đếm riêng; mean loại fallback có thể thiên về các bài dễ.

### 56. Chỉ phạt W hoặc H có nhược điểm gì? Tỷ lệ gần zero có thể gây hiểu nhầm không?

**Trả lời:** Nếu chỉ phạt W, rescale W'=cW, H'=H/c với c nhỏ làm W nhỏ và penalty giảm mà reconstruction không đổi. Phạt H riêng có vấn đề đối xứng khi c lớn. Vì vậy, sự thay đổi weights hoặc tỷ lệ gần 0 của nhánh một phía có thể phản ánh scale drift, không chỉ việc loại bỏ thông tin nhiễu.

**Hệ quả:** Ngưỡng tuyệt đối 1e-8 không bất biến theo scale. Không được nhìn W gần zero tăng rồi kết luận "mỗi câu chỉ còn một chủ đề đúng hơn" mà bỏ qua H, reconstruction và summary. Penalty một phía vẫn là một điều kiện ablation đáng khảo sát, nhưng cần báo rõ giới hạn diễn giải.

**Hướng kiểm chứng:** Đọc cả hai factors, theo dõi scale/norm, xem ROUGE và outputs; cân nhắc thêm thống kê sau canonical normalization với định nghĩa công bố rõ.

### 57. L1 không tăng ROUGE, hoặc có một seed tăng mạnh, thì kết luận gì?

**Trả lời:** Nếu delta=0 thì trên các cặp đã thử chưa thấy cải thiện overlap. Factors có thể đổi nhưng tập câu được chọn không đổi; hoặc thay đổi câu không làm n-gram overlap khác. Nếu delta âm thì L1 làm giảm metric ở điều kiện đó, không được che kết quả. Một seed tăng mạnh cũng chưa đủ cho kết luận.

**Vì sao nhiều seeds chưa đủ:** Mười seeds trên một bài vẫn là cùng nội dung và reference, không tương đương mười bài độc lập. Trong ablation auto-k, k còn có thể thay đổi giữa seeds, dù trong mỗi seed được giữ chung cho bốn nhánh; biến động seed không chỉ phản ánh final initialization.

**Muốn báo chắc hơn:** Nhiều bài đa dạng, paired differences, báo số cặp và fallback, xét độ bất định ở cấp bài. Bootstrap/CI hoặc kiểm định phù hợp là hướng bổ sung; code hiện không tự cung cấp chứng nhận ý nghĩa thống kê.

### 58. Dự án đã tái hiện paper nào? Che 20% có phải thuật toán trong paper NMFTS hoặc BCV không?

**Trả lời:** Em không tuyên bố tái hiện nguyên paper. Dự án dùng nền tảng NMF, cơ chế lựa chọn MMR, ROUGE và một phiên bản PRF thực dụng. Hai nhánh đang hoạt động là Local KL và Frobenius với L1/MMR. Các nhánh NMFTS pairwise và SNMF đồ thị đã rút khỏi web.

**Đối chiếu cụ thể:** [Aghdam, NMFTS](../aghdam_automative.pdf), phần 4.3, công thức (31), dùng heuristic dựa trên phân bố term cho K, không phải phép che 20% hiện tại. [Owen và Perry](https://arxiv.org/pdf/0908.2062), phần 5, dùng giữ lại khối hàng-cột; bản NMF được trình bày dùng least squares. Đổi một tỷ lệ che không biến elementwise KL-CV thành BCV.

**Về PRF:** [Park và An](../2108616.2108690.pdf) là nguồn ý tưởng query/feedback; code hiện dùng local TF-IDF centroid, chưa tái lập toàn bộ representation và protocol paper. [Wang và cộng sự](../1390334.1390387.pdf) là symmetric factorization trên sentence similarity, không đồng nhất với V≈WH đang demo.

**Cần nói chính xác:** "Paper-inspired" khác "reproduction". Muốn gọi là tái hiện phải khớp objective, biểu diễn, optimizer, chọn tham số, dữ liệu và đánh giá.

### 59. Muốn chứng minh một phương pháp tốt hơn thì phải làm thí nghiệm thế nào?

**Trả lời:** Em cần dữ liệu có quyền sử dụng và reference độc lập, chia theo bài để tránh nội dung trùng hoặc gần trùng qua các split. Dùng validation chọn hyperparameters và chính sách k; không chọn chúng để tối đa điểm trên bài test đang trình bày. Sau khi chốt, đánh giá test với cùng ngân sách summary và protocol tokenizer.

**Những gì phải báo:** Metric chính được xác định trước, các ROUGE còn lại, runtime, số bài/seed, tỷ lệ fallback/lỗi và đánh giá nội dung. So fixed-configuration và so sau tuning là hai câu hỏi thí nghiệm khác nhau. Với nhiều seeds trên một bài, không giả định mọi lần chạy là những bài độc lập khi tính độ bất định.

**Trạng thái thật:** Các split và metrics batch đang rỗng ở ngày đối chiếu. Em có pipeline và script tuning/experiment, nhưng chưa có benchmark hoàn chỉnh để khẳng định KL, L1 hay auto-k tốt hơn tổng quát. Reference không đi vào fit hiện tại; nếu dùng reference để chọn cấu hình thì chỉ làm trên validation.

### 60. Đóng góp của em là gì nếu NMF/ROUGE dùng thư viện? Dự án hoàn thiện đến đâu và cần làm gì tiếp?

**Trả lời:** Đóng góp kỹ thuật là tổ chức pipeline tiếng Việt cục bộ, custom MU để quan sát từng vòng, chính sách chọn rank, scoring/MMR/PRF, API contracts, giao diện phân tích và ablation có kiểm soát. Sklearn giải bài toán NMF final; Underthesea tách từ; rouge-score tính metrics với tokenizer của dự án. Em không tự nhận đã phát minh KL, L1 hoặc MMR.

**Bằng chứng có thể show:** Bốn trang dùng API thật; V/W/H/WH có kích thước đúng; trace chọn câu và index gốc; bảng ROUGE dùng cùng reference; ablation giữ cấu hình trong từng cặp và xuất artifacts. Có test và ví dụ chạy lại được, nhưng không gán một phần trăm "hoàn thiện" hoặc "độ chính xác" thiếu định nghĩa.

**Giới hạn cần ưu tiên:** Auto-k trên sparse positive-entry validation; relevance ở k=1 và phụ thuộc scale; vocabulary/bigram và sự khớp định dạng stopword với token tiếng Việt; similarity từ vựng chưa hiểu đầy đủ nghĩa; thiếu corpus benchmark và đánh giá độc lập.

**Vận hành:** Frontend gọi backend FastAPI. RUN_BACKEND.txt mặc định 8001, không dừng ứng dụng khác. Nếu dùng 8003 như phiên demo gần nhất để tránh SUN397, truyền `-Port 8003` và đặt `VITE_API_PROXY_TARGET=http://127.0.0.1:8003` trước khi chạy Vite. Giữ terminal mở; restart mất registry job ablation. Đây là vận hành, không đổi thuật toán.

**Câu kết thúc phù hợp:** "Em đã xây dựng một hệ thống phân tích và tóm tắt trích xuất có thể giải thích quyết định. Hiệu quả thuật toán tổng quát và độ tin cậy của bộ chọn k vẫn cần đánh giá trên dữ liệu độc lập."

## Phụ lục A: mẫu minh họa auto-k có thể chọn 3

Đây là **mẫu phụ**, không phải mẫu AI 12 câu điền sẵn trên web; không dùng số của nó để trả lời kết quả mặc định.

Đây là **văn bản tự viết, cố ý lặp từ vựng trong từng nhóm**, không phải benchmark. Dùng nguyên văn, để trống k; seed=42 ở trang 01. Trang 02 dùng seed core mặc định 42 và có thể đặt số câu tóm tắt=3.

Nông dân đã sử dụng hệ thống tưới tự động để tiết kiệm nước cho ruộng lúa. Nông dân đang sử dụng hệ thống tưới tự động để tiết kiệm nước cho ruộng lúa. Nông dân sẽ sử dụng hệ thống tưới tự động để tiết kiệm nước cho ruộng lúa.

Trạm y tế đã tổ chức tiêm vaccine cho trẻ em theo lịch tiêm chủng. Trạm y tế đang tổ chức tiêm vaccine cho trẻ em theo lịch tiêm chủng. Trạm y tế sẽ tổ chức tiêm vaccine cho trẻ em theo lịch tiêm chủng.

Doanh nghiệp đã bảo vệ tài khoản của nhân viên bằng mật khẩu và xác thực hai lớp. Doanh nghiệp đang bảo vệ tài khoản của nhân viên bằng mật khẩu và xác thực hai lớp. Doanh nghiệp sẽ bảo vệ tài khoản của nhân viên bằng mật khẩu và xác thực hai lớp.

Kết quả đã kiểm tra qua API ngày 14-10-2026: 9 câu, 45 đặc trưng, selector chọn k=3. Một số score: k=1: 0.467235; k=2: 0.406447; k=3: 0.263495; k=7: 0.264881. Với nhánh KL, ngân sách 3, không query và defaults, summary giữ index 0/3/6, không fallback. Thay văn bản, phần mềm hoặc seed có thể đổi kết quả. Chênh lệch nhỏ với k=7 không phải bằng chứng chắc chắn có đúng ba chủ đề.

## Phụ lục B: những câu cần tránh và cách nói chính xác hơn

| Câu dễ bị phản biện | Cách nói chính xác |
| --- | --- |
| k=1 nên văn bản chỉ có một chủ đề | k=1 đạt score thấp nhất theo protocol validation hiện tại |
| Che 20% là chuẩn bắt buộc của NMF | 20% là hyperparameter của bộ chọn k trong dự án |
| L1 tự biết từ nào là nhiễu | L1 phạt độ lớn weights; solver cân bằng với data fit |
| Trọng số lớn là sai và phải giảm | Trọng số có tăng/giảm tùy tỷ lệ cập nhật và objective |
| W/H càng thưa thì summary càng tốt | Độ thưa cần đọc cùng reconstruction, outputs và metrics |
| Loss KL nhỏ hơn Frobenius nên KL tốt hơn | Hai raw objectives không cùng thang đo |
| MMR cao chứng minh loss tốt | MMR là score lựa chọn phụ thuộc vòng và relevance |
| ROUGE=0.8 nghĩa là đúng 80% | Đó là metric overlap theo reference/tokenizer |
| Hệ thống đã tìm đúng số ý của mọi bài | Selector là heuristic rank selection, chưa xác minh semantic topic count |
| Một mẫu tăng ROUGE chứng minh L1 hiệu quả | Đó là kết quả pilot trên mẫu/cấu hình đã công bố |
| Mô phỏng là các vòng của sklearn summary | Mô phỏng là custom MU riêng; final summary dùng sklearn |
| Đã giống hoàn toàn các paper | Dùng các nền tảng/ý tưởng nghiên cứu; chưa tái hiện toàn bộ protocol |

## Phụ lục C: mở phần nào để chứng minh câu trả lời?

| Chủ đề cần giải thích | Trên web | Code đối chiếu |
| --- | --- | --- |
| Token, unigram/bigram, TF-IDF | 01: V và danh sách đặc trưng | [preprocess.py](../backend/app/core/preprocess.py), [vectorize.py](../backend/app/core/vectorize.py) |
| NMF, loss, vòng cập nhật | 01: V/W/H/WH và slider | [nmf.py](../backend/app/core/nmf.py): fit_nmf, objective_breakdown |
| Tự chọn k | 01/02: Ứng viên k | [nmf.py](../backend/app/core/nmf.py): select_k_by_imputation |
| Final NMF và sentence relevance | 02: Objective, phân tích từng câu | [summarize.py](../backend/app/core/summarize.py): _fit_local_nmf, _topic_relevance |
| MMR, phá hòa, thứ tự câu | 02: trace theo vòng và summary | [summarize.py](../backend/app/core/summarize.py): _select_sentences |
| Keywords và câu đại diện | 02: từ khóa theo chủ đề | [summarize.py](../backend/app/core/summarize.py): _topic_terms |
| Query PRF | 02: query và câu pseudo-relevant | [summarize.py](../backend/app/core/summarize.py): _query_feedback |
| Hai outputs và ROUGE | 03: nguồn/reference, outputs, bảng điểm | [main.py](../backend/app/main.py): evaluate_api; [evaluate.py](../backend/app/core/evaluate.py) |
| L1 ablation và paired delta | 04: bảng kết quả, tải CSV/JSON | [ablation.py](../backend/app/core/ablation.py), [ablation_jobs.py](../backend/app/core/ablation_jobs.py) |
| Tuning và benchmark | Không có màn hình tuning riêng | [tune_params.py](../scripts/tune_params.py), [run_experiment.py](../scripts/run_experiment.py) |
| Khởi động và kết nối | Trạng thái backend ở sidebar | [RUN_BACKEND.txt](../RUN_BACKEND.txt), [start_backend.ps1](../scripts/start_backend.ps1), [vite.config.js](../frontend/vite.config.js) |

## Tài liệu đọc thêm

- [Cẩm nang hệ thống](NMF_SYSTEM_GUIDE.md): diễn giải dài và ví dụ theo pipeline.
- [Kịch bản demo chi tiết](DEMO_PRESENTATION_SCRIPT.md): thao tác và cấu hình demo.
- [Kịch bản nói 5-7 phút](DEMO_SPOKEN_SCRIPT_5_7_MIN.md): bản lời thoại rút gọn.
- [Thiết kế ablation L1](ABLATION_L1.md): protocol và artifacts.
- [Mapping paper và thí nghiệm](PAPER_METHODS_AND_EXPERIMENTS.md): phạm vi tương ứng với nghiên cứu.

Khi gặp câu hỏi vượt ngoài số liệu đã có, cách trả lời tốt là nêu phần đã biết, chỉ ra giới hạn, và nói rõ phép kiểm chứng cần làm. Không dùng một lời giải thích hợp lý để thay cho bằng chứng thực nghiệm chưa được tạo.
