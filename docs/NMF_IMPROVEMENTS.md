# Tổng hợp cải tiến NMF / TopicSum

Tài liệu này ghi lại luồng cũ, các thay đổi thuật toán và nơi triển khai trong project. Các phương pháp lấy ý tưởng từ paper được đánh dấu là **paper-inspired** khi biểu diễn hoặc thiết lập chưa tái hiện nguyên trạng nghiên cứu.

**Cập nhật hiện tại:** hai nhánh tóm tắt/đánh giá là Local KL-NMF + L1 + MMR và Local Frobenius-NMF + L1 + MMR. Chúng dùng cùng `alpha_W`, `alpha_H`, `l1_ratio`, `k`, seed, TF-IDF, ngân sách câu và MMR; phần được thay đổi để so sánh là hàm mất mát. Trang mô phỏng dùng MU tự cài đặt để minh họa vòng lặp, không phải trace nội bộ của scikit-learn ở luồng sản xuất. NMFTS pairwise và SNMF đồ thị câu đã được bỏ khỏi API/web.

## Tóm tắt trước và sau

| Hạng mục | Trước | Hiện tại | Tác động |
| --- | --- | --- | --- |
| Tóm tắt chính | Dựa vào vocabulary/topic của Global NMF train trên corpus cũ | Fit TF-IDF và NMF cục bộ trên câu trong văn bản đầu vào | Topic, từ khóa và câu trích gắn với văn bản hiện tại |
| Loss mặc định | Frobenius/MSE trong một số luồng cũ | Local KL divergence + L1, solver MU | Hiển thị đúng objective đang dùng; có thể tách loss tái tạo và regularization |
| Chọn k | Heuristic hoặc k cố định | Masked KL imputation khi không truyền k | Chọn k dựa trên khả năng khôi phục các phần tử TF-IDF bị che |
| Chọn câu | Xếp hạng theo trọng số/topic đơn giản | MMR: cân bằng relevance và cosine redundancy | Hạn chế lặp ý, sau cùng khôi phục thứ tự câu gốc |
| Phương pháp so sánh | Chỉ tập trung Local KL-NMF | Thêm Local Frobenius-NMF với cùng L1 | So sánh KL và Frobenius trong khi giữ chung penalty, MMR và chính sách chọn k |
| Đánh giá | Chưa có dữ liệu benchmark dùng được | Có batch runner, validation tuning và API đọc kết quả | Chưa báo điểm cho đến khi có reference summary hợp pháp |

## Luồng Local KL-NMF

Trong `backend/app/core/summarize.py`, luồng tóm tắt mặc định:

1. Tách văn bản thành câu và giữ chỉ số gốc.
2. Tiền xử lý từng câu; fit TF-IDF cục bộ ngay trên các câu đó.
3. Chọn k bằng masked KL imputation nếu người dùng không nhập k.
4. Fit scikit-learn NMF mới với `init="nndsvdar"`, `solver="mu"`; chọn `beta_loss="kullback-leibler"` hoặc `"frobenius"`. Cả hai nhánh dùng chung L1 trên W/H.
5. Tính mức liên quan từ phân phối topic; chọn câu bằng MMR.
6. Ghép các câu theo thứ tự xuất hiện trong văn bản.

Cấu hình mặc định hiện là `alpha_w=0.001`, `alpha_h=0.001`, `l1_ratio=1.0`, `max_iter=500`, `seed=42`. Mức 0.001 được chọn bảo thủ sau khi phát hiện 0.1 làm nhánh Frobenius suy biến thành W=H=0 trên văn bản demo; đây chưa phải giá trị tối ưu thực nghiệm. Alpha/L1 ratio có thể điều chỉnh và cần chọn trên validation. Scikit-learn scale penalty theo kích thước ma trận: `alpha_W * n_terms` cho W và `alpha_H * n_sentences` cho H. `nndsvdar` sử dụng seed làm thay đổi thành phần khởi tạo ngẫu nhiên; cùng seed/input/config cho kết quả có thể tái lập.

### Mô phỏng và luồng sản xuất

`backend/app/core/nmf.py::fit_nmf` được gọi bởi `/api/simulate` để trả snapshot MU tự cài đặt. `backend/app/core/summarize.py::_fit_local_nmf` dùng `sklearn.decomposition.NMF` cho bản tóm tắt thật. Hai implementation có thể khác nhau về khởi tạo, cập nhật số và đường hội tụ; giao diện mô phỏng ghi rõ đây là minh họa, không tuyên bố snapshot là trạng thái bên trong của scikit-learn.

## Ghi chú các nhánh nghiên cứu đã rút khỏi web

Các nhánh dưới đây không còn được API tóm tắt chấp nhận và không xuất hiện trong giao diện. Hai phương pháp đang hoạt động là KL+L1 và Frobenius; cả hai dùng MMR để trích xuất câu.

### NMFTS-inspired pairwise

Nhánh này tối ưu reconstruction Frobenius cùng symmetric-KL penalty giữa các hàng W, có trọng số theo độ tương tự câu. Phần chọn câu tiếp tục dùng MMR sau phân rã. `pairwise_lambda` điều chỉnh độ mạnh của liên kết.

Đây là bản **paper-inspired**, không phải tuyên bố tái hiện đầy đủ paper Aghdam và cộng sự: đồ thị hiện được dựng từ TF-IDF cosine kNN, không phải toàn bộ semantic representation và protocol dữ liệu của paper.

### SNMF đồ thị câu

Nhánh này phân rã đồ thị tương tự câu theo dạng `A ≈ G Gᵀ`, gán câu vào cụm theo hệ số G rồi chọn đại diện từ các cụm. Ranking dùng độ tương tự nội cụm và centrality.

Đây là bản **paper-inspired** theo hướng Wang và cộng sự. TF-IDF cosine kNN là thay thế thực dụng cho semantic-role/lexical semantic similarity của paper, nên chưa tương đương hoàn toàn về biểu diễn.

### Query PRF

Trên nhánh Local KL-NMF, hệ thống biến query thành vector theo vocabulary của chính văn bản, lấy tối đa ba câu pseudo-relevant, trộn centroid các câu đó với query rồi kết hợp query relevance và topic relevance trước MMR. Nếu query không có từ khớp vocabulary, API trả lý do và tiếp tục bằng Local KL-NMF + MMR thông thường.

Query PRF được đưa vào batch chỉ khi **mọi bài** trong split có trường `query`. Nhánh này lấy ý tưởng từ Park và An; dữ liệu/query protocol paper chưa được tái lập trong project.

Chi tiết mapping với nghiên cứu và các giới hạn học thuật nằm tại [PAPER_METHODS_AND_EXPERIMENTS.md](PAPER_METHODS_AND_EXPERIMENTS.md).

## Chọn k và diễn giải loss

- Auto-k thử mọi k khả thi từ 1 đến `min(10, n_sentences, n_terms)`. Mỗi lượt giữ lại ngẫu nhiên 20% vị trí có TF-IDF dương, fit KL không regularization trên các vị trí còn lại, rồi tính mean KL trên các vị trí giữ lại. Mặc định chạy 10 lượt; điểm thấp hơn là reconstruction validation tốt hơn.
- Nếu số quan sát dương quá ít để validation có ý nghĩa, API dùng heuristic `max(1, n_sentences // 2)`, trả `score=null` và gắn nhãn heuristic. Nếu chỉ có một k khả thi, cũng không tạo score giả. Kết quả này không phải ROUGE, không phản ánh trực tiếp chất lượng câu tóm tắt và không tối ưu riêng cho Frobenius.
- Cả hai loss chính đều có objective `data-fit loss + L1(W,H)` với alpha và l1 ratio chung. Các scale/units của KL và Frobenius khác nhau, do đó không so sánh trực tiếp số loss giữa hai method.
- Mô phỏng áp dụng MU tự cài đặt với cùng penalty để minh họa; nó không phải execution trace của sklearn production estimator.
- Chủ đề hiện là latent factors, không được gán nhãn ngữ nghĩa tự động. Từ khóa xếp hạng bằng trọng số H đã chuẩn hóa kết hợp độ độc quyền từ giữa topic; mỗi topic hiển thị thêm câu đại diện từ W.
- Seed trong summarize được truyền tới `nndsvdar`. Trong ablation, cùng seed được ghép cặp giữa bốn nhánh. Web ablation nhận một bài viết thủ công nên là pilot; độ lệch qua seed không phải độ bất định trên tổng thể bài viết.
- MMR trace ghi điểm relevance/redundancy/MMR của từng ứng viên ở từng vòng. Chỉ nên so sánh điểm trong cùng một vòng vì tập đã chọn và redundancy thay đổi theo vòng.
- Không so sánh trị số loss trực tiếp giữa hai nhánh; dùng ROUGE trên cùng reference để đánh giá đầu ra.

## Benchmark và dữ liệu

`scripts/run_experiment.py` tạo so sánh trên cùng bài và cùng ngân sách câu cho Lead-N, Global NMF cũ, Local KL+L1+MMR và Local Frobenius+L1+MMR. Lỗi từng lần chạy vào `report.json`; không biến lỗi thành ROUGE bằng 0.

`scripts/tune_params.py` tìm k, số câu, alpha W/H, L1 ratio và MMR lambda cho cả hai phương pháp trên validation; dùng cùng không gian tham số để giữ so sánh công bằng. Cấu hình chỉ được xếp hạng nếu chạy thành công trên toàn bộ bài validation. Không dùng test để tuning.

Mỗi dòng JSONL cần có `id`, `article`, `reference_summary`; thêm `query` nếu đánh giá Query PRF. Tập dữ liệu cần chia theo **bài**, không chia câu của cùng bài sang nhiều split, và cần manifest ghi nguồn/quyền sử dụng.

Hiện `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl` đang rỗng và `results/metrics.csv` chưa có kết quả. Vì vậy chưa có cơ sở kết luận phương pháp nào tốt hơn; không tạo điểm giả từ demo.

## Tác dụng và vị trí trên web của phần mới

| Phần mới | Tác dụng dự kiến trong thuật toán | Vị trí người dùng nhìn thấy trên web |
| --- | --- | --- |
| So sánh Local KL+L1 và Local Frobenius+L1 | Quan sát hai objective dưới cùng alpha/L1, k, seed và MMR; không mặc định objective nào luôn tốt hơn | Tab **01 / Mô phỏng NMF** → menu phương pháp; Tab **02 / Tóm tắt văn bản** → menu **Phương pháp**; Tab **03 / Đánh giá** → ROUGE hai đầu ra. |
| Query PRF | Dùng query người dùng để ưu tiên câu phù hợp với ý định tìm kiếm; nếu query OOV/không khớp, hệ thống báo lý do và vẫn trả tóm tắt Local KL+MMR | Tab **02 / Tóm tắt văn bản** → khi chọn Local KL, nhập vào **Truy vấn (không bắt buộc)**; trọng số xuất hiện khi đã nhập query. Trong kết quả, thông tin pseudo-relevant được ghi dưới summary, và điểm Query xuất hiện trong từng câu nếu PRF được áp dụng. |
| Điều chỉnh alpha W/H và L1 ratio | Có thể khảo sát độ thưa thay vì mặc định cố định; alpha W tác động độ thưa của phân phối câu-topic, alpha H tác động độ thưa từ-topic | Tab **02 / Tóm tắt văn bản** → mở **Tham số thuật toán**; dùng chung cho hai loss. Giá trị mặc định 0.001 / 0.001 / 1.0 chưa được tune trên corpus. |
| Loss theo objective | Phân biệt KL+L1 và Frobenius; mô phỏng hiện phần loss theo từng vòng, tóm tắt hiện objective cuối | Tab **01 / Mô phỏng NMF** → đồ thị loss và breakdown; Tab **02 / Tóm tắt văn bản** → panel **Objective**. Không so sánh trực tiếp trị số giữa hai objective. |
| Auto-k và candidate scores | Cho xem điểm positive-entry masked KL validation; trường hợp heuristic không có score | Tab **01 / Mô phỏng NMF**, Tab **02 / Tóm tắt văn bản** và Tab **03 / Đánh giá**. Mặc định quét k=1..10 trong giới hạn kích thước; score không phải ROUGE. |
| Diễn giải topic | Hiển thị từ đặc trưng và các câu gần topic, không tự đặt nhãn ngữ nghĩa | Tab **02 / Tóm tắt văn bản** → **Từ khóa đặc trưng theo chủ đề** và **Câu đại diện**. |
| Giải thích MMR | Cho xem candidate scores ở đúng vòng greedy và nhắc không so điểm giữa các vòng | Tab **02 / Tóm tắt văn bản** → **Xem so sánh điểm MMR theo từng vòng** và score_round trong phân tích câu. |
| Benchmark nhiều phương pháp và validation tuning | So sánh cùng tập bài/ngân sách câu, tune trên validation; lỗi không bị ngụy trang thành điểm ROUGE 0 | Benchmark chạy riêng bằng `scripts/run_experiment.py`. Trang **03 / Đánh giá** tạo hai summary dùng KL+L1+MMR và Frobenius+L1+MMR. Tuning ở `scripts/tune_params.py`; không có màn hình tune riêng. |

Tab **04 / Ablation L1** dùng để kiểm tra riêng tác động của penalty L1 lên W, H và W+H. Form web nhập bài viết và reference trực tiếp; CLI vẫn nhận JSONL. Global NMF cũ và Lead-N chỉ còn là baseline trong benchmark batch ngoại tuyến `scripts/run_experiment.py`. Trang **03 / Đánh giá** so ROUGE của KL+L1+MMR và Frobenius+L1+MMR trên cùng văn bản.

**Hiệu quả đã đo và hiệu quả kỳ vọng:** các tác dụng trong bảng là mục tiêu thuật toán, chưa phải kết luận thực nghiệm. `data/splits/*.jsonl` và `results/metrics.csv` hiện rỗng, nên chưa có ROUGE/runtime để khẳng định nhánh nào tốt hơn. Cần bổ sung reference summary hợp pháp, tune trên validation và chỉ dùng test cho đánh giá cuối.

## File chính

- `backend/app/core/nmf.py`: custom MU cho mô phỏng, objective KL/Frobenius + regularization và positive-entry masked-KL k selection.
- `backend/app/core/vectorize.py`: TF-IDF cục bộ theo câu.
- `backend/app/core/summarize.py`: sklearn Local KL/Frobenius-NMF + cùng L1, MMR, topic term distinctiveness, representative sentences và Query PRF tùy chọn trên nhánh KL.
- `backend/app/core/paper_methods.py`: mã nghiên cứu NMFTS/SNMF cũ, không còn được gọi từ luồng web.
- `backend/app/schemas.py`, `backend/app/main.py`: API contracts, evaluate và batch metrics.
- `scripts/run_experiment.py`: benchmark các baseline/phương pháp.
- `scripts/tune_params.py`: tuning trên validation.
- `frontend/src/pages/Summarizer.jsx`: chọn method, query và tham số; loss theo objective.
- `frontend/src/pages/Evaluation.jsx`: ROUGE một mẫu và batch.
- `frontend/src/api.js`: gọi endpoint metrics backend.
- `docs/PAPER_METHODS_AND_EXPERIMENTS.md`: mapping paper, protocol chạy và hướng tiếp theo.

## Việc nên làm tiếp

1. Chuẩn bị dữ liệu tham chiếu tiếng Việt có quyền sử dụng; chạy validation tuning rồi đánh giá test một lần.
2. Báo cáo ROUGE-1/2/L cùng runtime, số câu và tỷ lệ fallback; bổ sung BERTScore hoặc đánh giá người đọc khi có nguồn lực.
3. Chạy ablation tách riêng KL/Frobenius, L1 và MMR; giữ nguyên split, ngân sách câu và seed.
4. Nếu mục tiêu là tái hiện paper, triển khai đúng semantic similarity/role representation, tiền xử lý, dataset và siêu tham số gốc trước khi so sánh.
