# NMF: những ngộ nhận, lưu ý và kiến thức phải hiểu đúng

> Đối chiếu code/web ngày **15-10-2026**. Tài liệu này là bản ghi chú để tránh hiểu sai và diễn giải quá mức khi đọc hệ thống hoặc báo cáo. Không phải danh sách tính năng mới; các hạn chế được nêu ở đây chưa được coi là đã sửa.
>
> Đọc cùng [Cẩm nang hệ thống](NMF_SYSTEM_GUIDE.md) để học đầy đủ, và [60 câu hỏi bảo vệ](NMF_DEFENSE_QA.md) để luyện trả lời. File này tập trung vào **ranh giới giữa các khái niệm**, những điều không thể suy ra từ một con số và các điểm phải kiểm tra trong code.

## 1. Bản đồ để không lẫn các bước

```text
Văn bản nguồn
    -> tách câu, giữ index gốc
    -> tách từ, tiền xử lý, tạo đặc trưng
    -> TF-IDF cục bộ V
    -> nhận k nhập tay hoặc chọn k bằng validation
    -> fit NMF để có W, H và WH
    -> tính relevance; bổ sung query relevance nếu có PRF hợp lệ
    -> MMR chọn các câu
    -> sắp theo index gốc, ghép thành summary

Summary + reference
    -> tokenizer đánh giá
    -> ROUGE
```

Không gộp mọi bước thành "NMF tự tạo tóm tắt". Trong pipeline hiện tại, **NMF học factors, scoring định nghĩa mức quan trọng, MMR lựa chọn câu, ROUGE đánh giá overlap**.

### Bốn đại lượng hay bị đánh đồng

| Đại lượng | Trả lời câu hỏi nào? | Không tự trả lời được |
| --- | --- | --- |
| Objective NMF | V được WH tái tạo thế nào, với chi phí regularization bao nhiêu? | Summary có giữ đúng ý chính không? |
| Score chọn k | Rank nào dự đoán các ô giữ lại tốt nhất theo protocol đang dùng? | Văn bản có chính xác bao nhiêu chủ đề ngữ nghĩa? |
| Relevance/MMR | Câu nào được ưu tiên theo quy tắc scoring và tập đã chọn? | Phương pháp loss nào tốt nhất trên mọi bài? |
| ROUGE | Candidate trùng với reference bao nhiêu theo tokenizer và metric? | Candidate chắc chắn đúng sự thật, mạch lạc và đầy đủ không? |

**Câu cần nhớ:** Mỗi con số có đối tượng đo và phạm vi diễn giải riêng.

### Mẫu web mới không phải nhãn huấn luyện NMF

`test_nmf_01` có 12 câu và reference 4 câu S1/S5/S8/S10, tức index `[0,4,7,9]`. Trang 02/03 mặc định 4 câu; auto-k hiện chọn 1 với seed 42. Bốn câu reference không có nghĩa k phải là4; k=1 không buộc summary chỉ có một câu.

`extractive_summary` được điền vào ô tham chiếu trang 03, không phải prediction NMF sẵn. Nhãn 0/1 không dùng để fit hay ép chọn câu; frontend chỉ lấy số index làm ngân sách ban đầu. Reference do người dùng cung cấp, chưa xác minh thông tin/số liệu của bài. Trang 04 chưa nạp sẵn dữ liệu và vẫn mặc định 3 câu, cần tự dán và đổi ngân sách khi dùng cùng mẫu.

## 2. Tám điều phải hiểu trước khi đọc sâu

### A. Câu nguồn và câu đã tiền xử lý là hai bản khác nhau

NMF dùng biểu diễn từ bản đã tiền xử lý. Summary trích xuất lấy lại câu nguồn, không lấy chuỗi token đã bỏ stopword để trả cho người dùng. Vì vậy, tiền xử lý không đồng nghĩa văn bản output bị mất dấu hoặc mất từ nối.

### B. Một document trong TF-IDF của dự án là một câu

IDF được tính trên tập câu của chính bài đang nhập, không trên 3260 bài báo cũ. Thêm hoặc đổi câu có thể đổi IDF và V, kể cả với những câu còn lại.

### C. Một đặc trưng không nhất thiết là một âm tiết hoặc một từ viết cách

Underthesea có thể tạo token nhiều âm tiết. Vectorizer tạo unigram/bigram từ token, nên số cột V là số đặc trưng, không phải tổng số từ nhìn thấy bằng mắt.

### D. W/H là tham số học được; WH là dữ liệu tái tạo

Theo quy ước hiện tại:

\[
V:n\times m,\qquad W:n\times k,\qquad H:k\times m,\qquad WH:n\times m.
\]

\[
(WH)_{ij}=\sum_{t=1}^{k}W_{it}H_{tj}.
\]

V không bị cập nhật trong một lần fit. W/H được điều chỉnh để thay đổi WH.

### E. k là số thành phần được yêu cầu, không phải nhãn đếm ý chính

rank(WH) ≤ k. Có k cột W không bảo đảm có k thành phần độc lập, hoạt động tốt hoặc có tên chủ đề rõ nghĩa.

### F. Loss và solver là hai khái niệm khác nhau

Loss định nghĩa cần tối ưu gì. Solver MU định nghĩa cách cập nhật weights. "Loss cập nhật W/H" chỉ là cách nói tắt; chính thuật toán tối ưu thực hiện cập nhật.

### G. Regularization là lựa chọn cấu trúc, không phải nhãn đúng/sai

L1 đặt chi phí lên độ lớn weights để khuyến khích biểu diễn gọn. Dữ liệu quyết định giảm weights đến đâu vẫn có thể tái tạo tốt. Nó không có danh sách từ nào là nhiễu được gán nhãn sẵn.

### H. Có kết quả chạy được khác với có kết luận thực nghiệm

Một ví dụ là bằng chứng cho hành vi ở cấu hình đó. Muốn kết luận hiệu quả tổng quát cần dữ liệu độc lập, protocol, đối chứng và báo độ bất định phù hợp.

## 3. Ba mươi ngộ nhận thường gặp

### 01. "NMF tự hiểu bài và sinh ra câu tóm tắt mới"

**Hiểu đúng:** NMF của dự án học quan hệ trong biểu diễn số, không phải mô hình ngôn ngữ sinh câu. Output lấy các câu có sẵn qua MMR rồi ghép theo thứ tự nguồn.

**Điều phải nhớ:** Trích xuất giúp tránh bịa câu mới, nhưng có thể làm mất ngữ cảnh, đại từ quy chiếu hoặc câu giải thích. Giữ câu nguồn không bảo đảm summary luôn dễ hiểu.

### 02. "NMF tách ra các token và cụm từ trong bảng từ khóa"

**Hiểu đúng:** Tokenization thuộc Underthesea; unigram/bigram thuộc vectorizer. NMF chỉ học trọng số trên vocabulary đã xây dựng.

**Ví dụ:** Nếu tokenizer tạo token ghép, token đó đã có trước khi fit NMF. Từ khóa hiện trên web không chứng minh NMF tự học cách phân đoạn tiếng Việt.

### 03. "Bigram chắc chắn là một cụm từ có nghĩa và liền nhau trong câu gốc"

**Hiểu đúng:** Bigram là hai token liên tiếp ở chuỗi đưa vào vectorizer. Dự án tạo chuỗi đó sau khi bỏ stopword, nên adjacency có thể khác câu gốc. Một token ghép cũng có thể gồm nhiều âm tiết.

**Điều phải nhớ:** Từ khóa bigram lạ có thể bắt nguồn từ representation, không chỉ từ thuật toán phân rã. Không đặt tên topic chỉ từ một bigram rời rạc.

### 04. "TF-IDF cao nghĩa là chi tiết đó quan trọng nhất"

**Hiểu đúng:** TF-IDF phản ánh tần suất và độ phân biệt trong tập câu. Một từ rất hiếm, lỗi chính tả hoặc tên riêng phụ vẫn có thể có IDF cao. Relevance của câu không phải lấy ô TF-IDF lớn nhất.

**Lưu ý trong code:** Các hàng V được chuẩn hóa L2; sentence length và norm sau chuẩn hóa không đơn giản phản ánh lượng thông tin.

### 05. "W/H không âm nên chúng là xác suất"

**Hiểu đúng:** Không âm khác với xác suất. W/H không bắt buộc có tổng hàng bằng 1. Việc code chuẩn hóa một hàng để tính scoring không chứng minh xác suất đó được hiệu chỉnh hoặc có nhãn xác nhận.

**Ví dụ:** W_it=0.8 không có nghĩa câu i có 80% xác suất thuộc topic t.

### 06. "WH là nhân từng ô, hoặc lấy W so trực tiếp với V"

**Hiểu đúng:** WH là phép nhân ma trận. So sánh tái tạo diễn ra giữa V_ij và (WH)_ij vì chúng có cùng vai trò câu - đặc trưng. W và V thường khác kích thước, không thể so từng ô như hai dự đoán cùng loại.

**Ví dụ tính tay:** W_i=[0.2,0.5], H_j=[0.4,0.8] cho (WH)_ij=0.48.

### 07. "Không có nhãn summary thì không thể có loss"

**Hiểu đúng:** Loss NMF dùng dữ liệu V làm mục tiêu tái tạo, cộng penalty lên factors. Đây là bài toán biểu diễn không giám sát, không cần biết câu nào phải có trong summary.

**Phân biệt:** Reference cần cho ROUGE; không tham gia fit NMF hoặc chọn câu trong luồng hiện tại.

### 08. "Loss nhỏ hơn nghĩa là summary tốt hơn"

**Hiểu đúng:** Loss nhỏ hơn cho biết objective đó nhỏ hơn trong điều kiện đang xét. Chọn câu còn phụ thuộc relevance, MMR, query và ngân sách. Vì vậy, không có quan hệ bắt buộc loss giảm thì ROUGE tăng.

**Điều phải nhớ:** NMF không được huấn luyện để trực tiếp tối đa ROUGE trong pipeline hiện tại.

### 09. "Frobenius là đúng bằng MSE, còn KL chỉ là sai số tương đối"

**Hiểu đúng:** Data loss Frobenius của code là nửa tổng bình phương sai số, không phải mean squared error chia cho số ô:

\[
D_F=\frac12\sum_{i,j}(V_{ij}-(WH)_{ij})^2.
\]

Generalized KL là:

\[
d(x,y)=x\log(x/y)-x+y,\qquad d(0,y)=y.
\]

KL có phần log tỷ lệ, nhưng không chỉ là phần trăm sai số tương đối. V/WH không cần là hai phân phối xác suất tổng bằng 1 để dùng generalized KL.

**Điều phải nhớ:** Chuẩn hóa loss theo số ô không biến KL và Frobenius thành cùng một thước đo.

### 10. "KL luôn tốt hơn Frobenius vì đây là dữ liệu văn bản"

**Hiểu đúng:** Hai losses có cách đánh đổi sai khác nhau. TF-IDF là representation đã biến đổi, không tự có phân phối đếm Poisson chỉ vì dùng KL. Chất lượng tóm tắt của hai nhánh phải được kiểm chứng, không suy ra chỉ từ tên hàm loss.

**Trong dự án:** Hai nhánh final đều dùng L1 và MMR; không so KL+L1+MMR với một Frobenius thuần không penalty.

### 11. "L1 phạt mạnh vì ma trận đã thưa, hoặc đếm số ô khác 0"

**Hiểu đúng:** L1 là tổng trị tuyệt đối, trong NMF không âm là tổng weights. Nó không trực tiếp đếm nonzero như L0.

**Phản ví dụ:** [1,0,0] và [0.5,0.3,0.2] đều có L1=1 nhưng số nonzero khác nhau. Vì vậy, số zero riêng nó không quyết định L1 penalty.

**Điều phải nhớ:** L1 khuyến khích sparsity qua tối ưu, không hứa mỗi câu chỉ còn 1-2 thành phần.

### 12. "L1 thấy trọng số lớn thì tự biết nó nhiễu và xóa nó"

**Hiểu đúng:** Không có ngưỡng trọng số X rồi mới bắt đầu phạt. Tất cả weights trong factor được regularize đều chịu penalty. Solver cân bằng penalty với lợi ích tái tạo; weight lớn vẫn có thể cần giữ hoặc tăng.

**Ví dụ:** Cùng W_it=0.8, tỷ lệ MU=0.6 làm nó thành 0.48; tỷ lệ 1.2 làm nó thành 0.96. Lớn/nhỏ không tự quyết định chiều cập nhật.

**Điều phải nhớ:** "Cần giảm" phải xét objective và dữ liệu, không chỉ nhìn màu đậm của một ô hoặc tổng một cột.

### 13. "l1_ratio=1 nghĩa là phạt 100% trọng số; alpha càng lớn càng tốt"

**Hiểu đúng:** l1_ratio=1 nghĩa là toàn bộ phần regularization theo L1, không pha L2. Độ mạnh còn do alpha và scaling:

\[
R_{L1}=\alpha_W m\sum_{i,t}W_{it}
+\alpha_H n\sum_{t,j}H_{tj}.
\]

Mặc định alpha_W=alpha_H=0.001 là cấu hình hiện tại, chưa phải mức tốt nhất đã được benchmark. Alpha quá lớn có thể làm factors sụp về 0.

**Điều phải nhớ:** Giảm penalty không đồng nghĩa giảm data loss; optimizer cân bằng tổng objective.

### 14. "Solver tái tạo lại W/H từ đầu ở mỗi vòng, hoặc hội tụ là tối ưu toàn cục"

**Hiểu đúng:** Một lần fit khởi tạo W/H rồi cập nhật luân phiên. Custom MU dùng H trước W; sklearn MU đang cài dùng W trước H. Không bốc lại factors ngẫu nhiên ở mỗi vòng, cũng không tối ưu một factor hoàn chỉnh rồi mới bắt đầu factor kia.

**Điều phải nhớ:** Max_iter là giới hạn, không phải số vòng luôn chạy đủ. Dừng theo điều kiện số học không chứng nhận nghiệm tối ưu toàn cục của bài toán không lồi.

### 15. "k là số câu tóm tắt"

**Hiểu đúng:** k quyết định số cột W và số hàng H. Ngân sách summary quyết định giữ bao nhiêu câu; chúng là hai tham số khác nhau.

**Ví dụ:** k=2 và summary_sentences=3 là cấu hình hợp lệ. MMR không bắt buộc lấy một câu cho mỗi thành phần.

### 16. "k là số chủ đề đúng; k lớn chứng tỏ đoạn văn trùng lặp nhiều"

**Hiểu đúng:** k là số thành phần biểu diễn. Số mảng nội dung người đọc nhận ra là một khái niệm khác. Các câu giống hệt nhau có thể tạo cấu trúc rank 1 dù lặp rất nhiều; nhiều nhóm từ vựng khác nhau có thể cần nhiều thành phần hơn.

**Phân biệt quan trọng:** Lặp trong từng nhóm giúp có quan sát chung để học; khác biệt giữa các nhóm tạo cấu trúc cần phân biệt. Hai điều đó không giống nhau.

**Điều phải nhớ:** Không dùng k để đo redundancy. Redundancy trong MMR được tính từ similarity giữa các câu.

### 17. "Auto-k che đúng 20% toàn ma trận và dùng cùng loss với nhánh cuối"

**Hiểu đúng:** Hiện code chọn khoảng 20% vị trí **V>0**, có làm tròn số lượng, qua 10 trial. Bộ chọn luôn fit KL không L1, tối đa 120 vòng mỗi lần, dù final model dùng Frobenius hoặc có L1.

**Điều phải nhớ:** Các ô zero không nằm trong validation hiện tại nhưng những ô zero không bị che vẫn thuộc training objective. Thay alpha của final fit không trực tiếp thay score chọn k nếu V/seed không đổi.

### 18. "Che một ô tương đương coi giá trị thật của nó bằng 0"

**Hiểu đúng:** Mask=0 tại ô bị che loại ô đó khỏi fit; không gán nhãn "từ này không xuất hiện". Sau fit, lấy giá trị gốc của ô đó để chấm prediction.

**Ví dụ:** Một ô thật 0.5 bị che thì WH dự đoán 0.5 là tốt. Nếu thay thành 0 và vẫn huấn luyện trên ô đó, mô hình lại được khuyến khích dự đoán 0; đó là bài toán khác.

### 19. "k=1 có score thấp nhất thì đó là đáp án chuẩn nhất về nội dung"

**Hiểu đúng:** Code có thể tính đúng argmin nhưng tiêu chí có thể chưa phù hợp với mục tiêu phân biệt chủ đề. Với bài ngắn, các ô bị che có thể mất hết quan sát dương của một từ. Validation chỉ trên ô dương cũng không trực tiếp chấm các dự đoán dàn trải tại ô zero.

**Điều phải nhớ:** Nói "k=1 tốt nhất theo positive-entry KL validation hiện tại", không nói "văn bản chắc chắn chỉ có một chủ đề". Nhiều mẫu đều chọn 1 là lý do kiểm tra protocol, không tự là bằng chứng phương pháp ngày càng đúng.

**Mẫu kiểm tra:** [Phần khó và phụ lục của bộ hỏi-đáp](NMF_DEFENSE_QA.md) ghi cả hạn chế và mẫu có auto-k=3. Mẫu được thiết kế để minh họa không chứng minh độ chính xác trên corpus thực tế.

### 20. "Tăng k luôn tốt hơn, hoặc loss trong mỗi lần chạy phải giảm khi k tăng"

**Hiểu đúng:** Trong bài toán lý tưởng xét các nghiệm tốt nhất của cùng objective, mô hình k+1 có thể biểu diễn mô hình k bằng cách thêm thành phần zero. Nhưng nghiệm mà solver thực tế tìm được còn phụ thuộc khởi tạo, giới hạn vòng và điều kiện dừng. Điểm trên validation có thể xấu hơn khi tăng k.

**Điều phải nhớ:** Chỉ dùng khả năng fit dữ liệu training để chọn k dễ ưu tiên mô hình lớn hơn mức cần thiết. Nhưng thay bằng một validation thiếu thông tin cũng không bảo đảm chọn đúng số chủ đề.

### 21. "Kết quả không đổi sau mỗi lần bấm nghĩa là cache hoặc dữ liệu giả"

**Hiểu đúng:** Cùng văn bản, cấu hình, seed và môi trường thường tạo kết quả tái lập. Trang tóm tắt hiện dùng seed core mặc định 42; API request tóm tắt không khai báo trường seed.

**Điều phải nhớ:** Đổi seed ở trang hỗ trợ giúp khảo sát độ nhạy; không cần cố làm kết quả thay đổi mỗi lần bấm để chứng minh thuật toán chạy thật. Nếu thay input vẫn cùng k thì xem toàn bộ candidate scores, không chỉ giá trị k.

### 22. "Relevance là tổng weights của câu, hoặc loss thấp của câu"

**Hiểu đúng:** Công thức hiện tại chuẩn hóa W theo hàng, tính salience từ tổng cột W, rồi kết hợp hai đại lượng:

\[
P_{it}=\frac{W_{it}}{\sum_u W_{iu}},\qquad
s_t=\frac{\sum_i W_{it}}{\sum_{i,u}W_{iu}},\qquad
r_i=\sum_tP_{it}s_t.
\]

Code có epsilon và chuẩn hóa r theo giá trị lớn nhất. Đây không phải tổng hàng W và cũng không phải lấy âm loss từng hàng V.

**Điều phải nhớ:** Khi k=1 và weights dương, P_i và s đều gần [1], nên relevance các câu gần bằng nhau. Điều này có thể làm giảm khả năng phân biệt câu, dù weights thô khác nhau.

### 23. "MMR chỉ phạt với câu vừa chọn, hoặc chỉ phạt khi cùng topic"

**Hiểu đúng:** Redundancy là cosine lớn nhất với **mọi câu đã chọn**, không lọc theo topic:

\[
\mathrm{Redundancy}(i,S)=\max_{j\in S}\cos(V_i,V_j).
\]

**Ví dụ:** Với S gồm câu 1 và 4, câu 6 có similarity 0.2 và 0.8 thì redundancy=0.8. Cùng topic không tự đồng nghĩa trùng ý; khác topic cũng không miễn phạt.

### 24. "Câu relevance cao nhất luôn được chọn; điểm MMR càng cao thì loss càng tốt"

**Hiểu đúng:** MMR cân bằng relevance và redundancy:

\[
\mathrm{MMR}(i,S)=\lambda r_i-(1-\lambda)\mathrm{Redundancy}(i,S).
\]

Giả sử đã có câu trong S, lambda=0.7: A có relevance=0.9, redundancy=0.9 thì MMR=0.36; B có relevance=0.8, redundancy=0.2 thì MMR=0.50. B thắng dù relevance thấp hơn.

**Điều phải nhớ:** MMR là quy tắc lựa chọn, không phải thước đo khách quan để xếp hạng các loss. Điểm còn phụ thuộc lambda, normalization và vòng chọn.

### 25. "Có thể so mọi score ở bảng cuối; score âm thì câu phải bị loại"

**Hiểu đúng:** Khi S thay đổi, score của các ứng viên thay đổi. Câu đã chọn giữ score tại vòng nó được chọn; câu chưa chọn giữ score của vòng cuối nó được xét. Bảng cuối có thể trộn nhiều score_round, nên muốn so ứng viên phải mở cùng vòng trong selection_trace.

**Về số câu:** Code vẫn lấy ứng viên tốt nhất còn lại nếu điểm âm, cho tới min(ngân sách, số câu nguồn). Không có threshold cố định làm rơi bớt câu trong MMR hiện tại.

**Về hòa điểm:** Ưu tiên MMR, rồi relevance, rồi index gốc. Với lambda=0, vòng đầu mọi redundancy bằng 0 nhưng code vẫn phá hòa bằng relevance. Số đã làm tròn trên UI không phải full precision.

### 26. "Query/PRF làm NMF học lại hoặc tìm thêm thông tin ngoài bài"

**Hiểu đúng:** Query được transform bằng vocabulary cục bộ. PRF lấy centroid từ tối đa ba câu pseudo-relevant rồi trộn với query. Nó không gọi Internet/LLM và không thêm từ ngoài vocabulary.

**Điều đặc biệt:** Query được xử lý sau final fit; nó thay relevance/MMR, không thay V, k hoặc W/H trong luồng hiện tại. Vì vậy, query có thể đổi summary nhưng objective NMF giữ nguyên.

**Giới hạn:** Query OOV có lý do bỏ PRF; "pseudo-relevant" là giả định tạm thời, không phải nhãn xác nhận. Từ cập nhật 15-10-2026, PRF hỗ trợ cả KL và Frobenius; không có ràng buộc lý thuyết buộc query chỉ đi với KL.

### 27. "ROUGE là accuracy; điểm 0.8 nghĩa là đúng 80% nội dung"

**Hiểu đúng:** Web báo ROUGE-1/2/L F1, là overlap với reference theo tokenizer. Nó không xác minh factuality, phủ định hoặc coherence. Một paraphrase đúng có thể điểm thấp; một câu sai phủ định có thể trùng nhiều token.

**Lưu ý trong code:** Tokenizer đánh giá giữ dấu và stopword, không English stemming. Nó không lấy trực tiếp các cột TF-IDF để tính ROUGE. Bigram ROUGE và bigram TF-IDF có thể khác vì bước tiền xử lý khác.

### 28. "Reference là văn bản gốc hoặc là câu trả lời đúng duy nhất"

**Hiểu đúng:** Reference là bản tóm tắt mục tiêu cho phép đánh giá, không phải toàn bộ nguồn. Nó có thể được coi là ground truth theo nghĩa nhãn đánh giá, nhưng tóm tắt có nhiều phương án hợp lý.

**Điều phải nhớ:** Giữ cùng reference khi so outputs. Không sửa reference theo output mình muốn thắng. Nếu dùng reference để tune tham số, phải làm trên validation chứ không tune trên test rồi gọi đó là đánh giá độc lập.

**Trên web:** Trang 03 có sẵn nguồn AI và reference riêng, vẫn sửa được; Nạp ví dụ khôi phục cả hai và số câu 4. API tự sinh hai outputs rồi chấm chúng. Nguồn không tự được dùng làm reference; summary điền sẵn trong ô tham chiếu không phải output hệ thống đã suy luận.

### 29. "Delta ROUGE dương hoặc SD bằng 0 chứng minh L1 hiệu quả"

**Hiểu đúng:** Delta dương chỉ phản ánh cải thiện trên những cặp bài/seed được chấm. Một bài với nhiều seeds không biến thành nhiều bài độc lập. SD=0 với một lượt là do số mẫu, không chứng minh ổn định tổng quát.

**Điều phải nhớ:** Đọc số cặp, số lượt thành công, lỗi và fallback. Báo cáo hiện dùng mean/SD của các lượt không fallback; nếu nhiều lượt thất bại, phần còn lại có thể không đại diện toàn bộ dữ liệu.

**Trong ablation:** Không lấy objective có penalty khác nhau để kết luận nhánh tốt hơn. k được dùng chung trong mỗi bài/seed, nhưng có thể khác giữa seeds khi auto-k.

**Mẫu AI:** Với k nhập tay 3, 4 câu, alpha 0.001 và seeds 42/43/44, chỉ phạt H tăng nhẹ R2/L nhưng giảm R1, phạt W hoặc W+H giảm cả ba; tỷ lệ gần zero lại bằng nhau giữa bốn nhánh. Không dùng bảng này để bảo đảm L1 có lợi hoặc so thẳng với trang 03 auto-k=1; xem [protocol và số đo](ABLATION_L1.md).

### 30. "Dùng cấu hình thư viện giống ví dụ hoặc có demo đẹp nghĩa là đã chuẩn paper"

**Hiểu đúng:** Cần phân biệt dùng nền tảng nghiên cứu, dùng thư viện và tái hiện nguyên paper. Reproduction còn liên quan objective, representation, optimizer, chọn tham số, dataset và evaluation protocol.

**Điều phải nhớ:** Che 20% là cấu hình hiện tại, không phải tỷ lệ bắt buộc. Đổi tỷ lệ che không biến masked elementwise CV thành BCV. Một ví dụ tăng ROUGE hoặc có k=3 không đủ chứng minh thuật toán tốt hơn tổng quát.

**Phạm vi thật:** Các nhánh đang có trên web là Local KL/Frobenius cùng L1/MMR. NMFTS pairwise và SNMF đồ thị đã rút khỏi web. Xem [mapping paper](PAPER_METHODS_AND_EXPERIMENTS.md) và câu 58 trong [bộ hỏi-đáp](NMF_DEFENSE_QA.md) trước khi nói "giống paper".

## 4. Sáu điểm cần đặc biệt lưu ý để hiểu bản chất

### 4.1. W/H có thể đổi scale mà WH không đổi

Với c>0:

\[
W'=cW,\qquad H'=H/c,\qquad W'H'=WH.
\]

Có thể rescale riêng từng thành phần, hoặc đổi thứ tự các thành phần đồng thời trong W/H. Vì vậy:

- Không so weights thô giữa hai lần fit như thể có cùng đơn vị tuyệt đối.
- Không coi T1 là cùng một nhãn semantic giữa mọi lần chạy.
- Tỷ lệ gần zero theo ngưỡng tuyệt đối có thể thay đổi vì scale, dù WH không đổi.
- Relevance hiện tại có thể đổi khi rescale từng thành phần, vì salience và chuẩn hóa hàng W đổi.

**L1 một phía:** Nếu chỉ phạt W, có thể làm W nhỏ và H lớn để giảm penalty mà giữ reconstruction; phạt H riêng có vấn đề đối xứng. Khi phạt cả hai, dọc theo rescale toàn bộ, penalty dạng a×c+b/c với a,b>0 tạo một mức cân bằng, nhưng không bảo đảm nghiệm factors duy nhất.

**Hướng cải tiến, chưa triển khai:** Quy ước scale trước scoring và báo thống kê chuẩn hóa bên cạnh raw weights. Không được tự coi các tỷ lệ zero trong ablation là bằng chứng semantic sparsity.

### 4.2. L1 là đánh đổi trên tổng loss, không phải công cụ nhận biết nhiễu

Khi l1_ratio=1:

\[
L=D(V,WH)+R_{L1}(W,H).
\]

Giả sử hệ số penalty hiệu dụng là 0.1. Nếu bỏ một weight 0.02:

| Data loss tăng | Penalty giảm | Tổng loss thay đổi | Diễn giải |
| ---: | ---: | ---: | --- |
| 0.001 | 0.002 | -0.001 | Có lợi cho objective |
| 0.01 | 0.002 | +0.008 | Không có lợi cho objective |

Đây là ví dụ so sánh hai trạng thái giả định, không phải solver thực sự thử xóa từng ô theo bảng. Cần đọc cả data loss và regularization loss. L1 cũng không bảo đảm hệ số bị xóa chính xác bằng 0 trong MU; có thể chỉ rất nhỏ.

### 4.3. Biểu diễn từ vựng đặt giới hạn cho những gì mô hình có thể học

TF-IDF chủ yếu thể hiện sự xuất hiện/đồng xuất hiện của token. Nó không tự có đầy đủ kiến thức ngữ nghĩa, phủ định hay suy luận nhân quả.

Các vấn đề cần kiểm tra:

- Từ hiếm và bigram có nhiều ô chỉ xuất hiện một lần, gây khó cho held-out validation.
- Bigram tạo sau khi bỏ stopword có thể khác cụm từ trong câu gốc.
- Vocabulary bị giới hạn tối đa 800 đặc trưng; không mặc định mọi đặc trưng đều được giữ với bài lớn.
- Định dạng stopword phải khớp token. Code hiện đọc từng dòng và so trực tiếp với token; một cụm viết cách và một token nối gạch dưới có thể không match.
- Đọc từ khóa kèm câu đại diện, không suy từ một danh sách token rời rạc ra tên chủ đề chắc chắn.

**Phân biệt:** Hạn chế tiền xử lý/representation không nên tự quy hết thành lỗi của NMF hoặc của KL.

### 4.4. Từ khóa và câu đại diện không phải nhãn giám sát

Code hiện không chỉ lấy H thô lớn nhất. Điểm từ khóa đặc trưng kết hợp phân bố trong thành phần và mức độc quyền giữa các thành phần, giản lược epsilon:

\[
q_{tj}=
\frac{H_{tj}}{\sum_{\ell}H_{t\ell}}
\frac{H_{tj}}{\sum_uH_{uj}}.
\]

Câu đại diện được xếp theo W đã chuẩn hóa theo hàng. Chúng giúp đọc factors, không xác nhận chủ đề đã được gán nhãn đúng.

**Cảnh giác:** "Top 5" là thứ hạng tương đối, không bảo đảm cả năm từ có trọng số đáng kể. Một thành phần rất yếu hoặc không hoạt động cần được kiểm tra trước khi đặt tên; không mặc định mọi ô topic trên UI đều là một chủ đề chất lượng.

### 4.5. Dự án có ba quá trình fit khác nhau

| Quá trình | Implementation/init | Loss/penalty | Giới hạn và mục đích |
| --- | --- | --- | --- |
| Chọn k | Custom MU, random | KL không L1 | 10 mask trial; tối đa 120 vòng mỗi fit; chấm ô dương giữ lại |
| Mô phỏng | Custom MU, random | KL hoặc Frobenius + penalty mặc định | Số vòng form, API tối đa 20; quan sát snapshots |
| Tóm tắt | Sklearn NMF, nndsvdar | KL hoặc Frobenius; alpha/ratio theo request | Tối đa 500 vòng; lấy factors cho scoring |

Vì vậy, không lấy score CV so trực tiếp với loss slider hoặc final objective. Cùng k/V/seed cũng không buộc hai implementations có cùng weights và loss cuối.

**Thứ tự cập nhật:** Custom H rồi W, sklearn MU đang cài W rồi H. Đây là khác biệt implementation, không phải một bên tự sai chỉ vì đổi thứ tự.

### 4.6. Bằng chứng hiệu quả và cơ chế dự phòng phải được báo riêng

Ngày đối chiếu, các file train/val/test và results/metrics.csv còn rỗng. Các ví dụ trong tài liệu là tính tay hoặc pilot có cấu hình, không phải benchmark corpus đã hoàn thành.

**Ablation:** Bốn cấu hình giữ chung k/MMR/ngân sách trong mỗi cặp bài/seed; không query. Đọc paired delta và số mẫu, không chỉ nhìn màu xanh của một ô điểm.

**Fallback:** Khi không có TF-IDF hữu ích hoặc factors sụp về 0, pipeline có thể lấy các câu đầu và báo lý do. Bỏ Lead-N khỏi phép so trên web không đồng nghĩa xóa cơ chế dự phòng khỏi code. Output fallback không được ghi là output do NMF+MMR thành công.

**Báo cáo đúng:** Tách khả năng chạy được, chất lượng minh họa và hiệu quả tổng quát. Không đưa một phần trăm "hoàn thiện" hoặc "chính xác" nếu chưa định nghĩa cách đo.

## 5. Checklist trước khi giải thích hoặc demo

- [ ] Phân biệt được văn bản nguồn, V và reference.
- [ ] Giải thích được n, m, k và phép nhân WH.
- [ ] Biết tokenizer/vectorizer tạo features; NMF học weights trên features.
- [ ] Biết loss khác solver, và objective khác relevance/MMR/ROUGE.
- [ ] Nói rõ k là thành phần biểu diễn, không bảo đảm số ý semantic đúng.
- [ ] Mô tả đúng auto-k: giữ lại ô dương, cùng mask giữa các k, KL không L1.
- [ ] Không coi 20%, alpha=0.001 hoặc max_iter=500 là cấu hình tối ưu đã được chứng minh.
- [ ] Hiểu L1 phạt độ lớn và cân bằng với data fit, không tự gán nhãn nhiễu.
- [ ] Biết W/H có scale ambiguity và relevance k=1 có thể gần bằng nhau.
- [ ] So MMR trong cùng vòng; không lọc redundancy theo topic.
- [ ] Dùng cùng reference và cấu hình kiểm soát khi so hai outputs.
- [ ] Báo điều kiện chạy, seed, số bài/seed và fallback; không giấu kết quả xấu.
- [ ] Nhận diện phần nào là kiến thức nền, code thực tế và hướng cải tiến chưa làm.
- [ ] Không dùng ví dụ được chọn có chủ đích để suy rộng thành hiệu quả corpus.

## 6. Những câu ngắn nên dùng khi bị hỏi

| Tình huống | Câu trả lời chính xác |
| --- | --- |
| Người xem thấy ba ý nhưng k=1 | "Đây là rank được chọn theo score validation, không phải nhãn đếm chủ đề ngữ nghĩa." |
| Hỏi k=3 nhập tay có sai không | "Em cố định 3 để khảo sát; không gọi đó là kết quả auto-k." |
| Hỏi L1 biết từ nào là nhiễu | "L1 không biết nhãn nhiễu; solver cân bằng chi phí weights với lợi ích tái tạo." |
| Hỏi loss nào tốt hơn từ raw value | "Hai losses khác scale; cần so outputs bằng cùng protocol đánh giá." |
| Hỏi score MMR cao thì câu chắc tốt không | "Nó tốt hơn theo quy tắc chọn trong vòng này, không phải chứng nhận chất lượng tuyệt đối." |
| Hỏi ROUGE cao có đúng hết không | "Nó đo overlap với reference; factuality và coherence cần kiểm tra thêm." |
| Hỏi đã giống paper hoàn toàn chưa | "Em dùng các nền tảng và ý tưởng nghiên cứu, chưa tái lập toàn bộ protocol." |
| Hỏi một mẫu L1 thắng có chứng minh gì | "Nó là kết quả ở mẫu/cấu hình này; kết luận tổng quát cần nhiều bài độc lập." |

## 7. Code và tài liệu để đối chiếu

| Nội dung | Nơi đọc |
| --- | --- |
| Token và stopword | [preprocess.py](../backend/app/core/preprocess.py), [stopwords_vi.txt](../backend/app/core/stopwords_vi.txt) |
| Local TF-IDF và bigram | [vectorize.py](../backend/app/core/vectorize.py) |
| Loss, custom MU và chọn k | [nmf.py](../backend/app/core/nmf.py) |
| Final NMF, scoring, keywords, MMR, PRF, fallback | [summarize.py](../backend/app/core/summarize.py) |
| ROUGE và tokenizer đánh giá | [evaluate.py](../backend/app/core/evaluate.py) |
| Ablation và cách tổng hợp số liệu | [ablation.py](../backend/app/core/ablation.py) |
| Chia dữ liệu, tuning và benchmark | [prepare_data.py](../scripts/prepare_data.py), [tune_params.py](../scripts/tune_params.py), [run_experiment.py](../scripts/run_experiment.py) |
| Kiến thức dài và ví dụ | [NMF_SYSTEM_GUIDE.md](NMF_SYSTEM_GUIDE.md) |
| Câu hỏi bảo vệ | [NMF_DEFENSE_QA.md](NMF_DEFENSE_QA.md) |
| Quan hệ với paper | [PAPER_METHODS_AND_EXPERIMENTS.md](PAPER_METHODS_AND_EXPERIMENTS.md) |

**Kết luận cần giữ:** Có một phép tính đúng không đồng nghĩa mọi cách diễn giải kết quả của phép tính đều đúng. Hiểu bản chất là biết đại lượng đang đo gì, nó phụ thuộc điều kiện nào, và những kết luận nào còn thiếu bằng chứng.
