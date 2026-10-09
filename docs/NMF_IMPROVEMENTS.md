# Tổng hợp cải tiến NMF / TopicSum

Tài liệu này ghi lại luồng cũ, các thay đổi thuật toán và nơi triển khai trong project. Các phương pháp lấy ý tưởng từ paper được đánh dấu là **paper-inspired** khi biểu diễn hoặc thiết lập chưa tái hiện nguyên trạng nghiên cứu.

## Tóm tắt trước và sau

| Hạng mục | Trước | Hiện tại | Tác động |
| --- | --- | --- | --- |
| Tóm tắt chính | Dựa vào vocabulary/topic của Global NMF train trên corpus cũ | Fit TF-IDF và NMF cục bộ trên câu trong văn bản đầu vào | Topic, từ khóa và câu trích gắn với văn bản hiện tại |
| Loss mặc định | Frobenius/MSE trong một số luồng cũ | Local KL divergence + L1, solver MU | Hiển thị đúng objective đang dùng; có thể tách loss tái tạo và regularization |
| Chọn k | Heuristic hoặc k cố định | Masked KL imputation khi không truyền k | Chọn k dựa trên khả năng khôi phục các phần tử TF-IDF bị che |
| Chọn câu | Xếp hạng theo trọng số/topic đơn giản | MMR: cân bằng relevance và cosine redundancy | Hạn chế lặp ý, sau cùng khôi phục thứ tự câu gốc |
| Phương pháp paper | Chưa có các nhánh quan hệ câu | Thêm NMFTS-inspired, SNMF graph và Query PRF | Có thể so sánh nhiều giả thuyết trên cùng pipeline |
| Đánh giá | Chưa có dữ liệu benchmark dùng được | Có batch runner, validation tuning và API đọc kết quả | Chưa báo điểm cho đến khi có reference summary hợp pháp |

## Luồng Local KL-NMF

Trong `backend/app/core/summarize.py`, luồng tóm tắt mặc định:

1. Tách văn bản thành câu và giữ chỉ số gốc.
2. Tiền xử lý từng câu; fit TF-IDF cục bộ ngay trên các câu đó.
3. Chọn k bằng masked KL imputation nếu người dùng không nhập k.
4. Fit NMF mới với `init="nndsvda"`, `solver="mu"`, `beta_loss="kullback-leibler"`, L1 trên W/H.
5. Tính mức liên quan từ phân phối topic; chọn câu bằng MMR.
6. Ghép các câu theo thứ tự xuất hiện trong văn bản.

Cấu hình mặc định vẫn là `alpha_w=0.1`, `alpha_h=0.1`, `l1_ratio=1.0`, `max_iter=500`, `seed=42`. Alpha và L1 ratio hiện có thể thay đổi qua API/UI và được tune trên validation. Loss Local được tính thành KL data loss cộng regularization; hệ số regularization phản ánh cách scikit-learn scale alpha theo kích thước ma trận.

## Các nhánh bổ sung theo paper

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

- K tự động hiện dùng masked KL imputation; đây là tiêu chí khôi phục TF-IDF bị che, không phải thước đo trực tiếp chất lượng tóm tắt.
- Cùng chính sách chọn k được dùng cho Local KL, NMFTS và SNMF để so sánh có kiểm soát. Nó **không** tối ưu riêng objective của NMFTS/SNMF. Có thể truyền cùng k cố định để đối chiếu.
- Local KL loss = KL reconstruction + L1 penalty.
- NMFTS loss = Frobenius reconstruction + pairwise symmetric-KL penalty.
- SNMF loss = symmetric similarity reconstruction.
- Không so sánh trị số loss trực tiếp giữa các nhánh vì objective, thang đo và ma trận được phân rã khác nhau.

## Benchmark và dữ liệu

`scripts/run_experiment.py` tạo so sánh trên cùng bài và cùng ngân sách câu cho Lead-N, Global NMF cũ, Local KL+L1+MMR, NMFTS pairwise, SNMF graph và Query PRF khi split có query đầy đủ. Lỗi từng lần chạy vào `report.json`; không biến lỗi thành ROUGE bằng 0.

`scripts/tune_params.py` tìm k, số câu, alpha W/H, L1 ratio, MMR lambda, pairwise lambda và query weight trên validation. Cấu hình chỉ được xếp hạng nếu chạy thành công trên toàn bộ bài validation. Không dùng test để tuning.

Mỗi dòng JSONL cần có `id`, `article`, `reference_summary`; thêm `query` nếu đánh giá Query PRF. Tập dữ liệu cần chia theo **bài**, không chia câu của cùng bài sang nhiều split, và cần manifest ghi nguồn/quyền sử dụng.

Hiện `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl` đang rỗng và `results/metrics.csv` chưa có kết quả. Vì vậy chưa có cơ sở kết luận phương pháp nào tốt hơn; không tạo điểm giả từ demo.

## Tác dụng và vị trí trên web của phần mới

| Phần mới | Tác dụng dự kiến trong thuật toán | Vị trí người dùng nhìn thấy trên web |
| --- | --- | --- |
| Chọn Local KL-NMF, NMFTS hoặc SNMF | Cho phép đổi giả thuyết phân rã/chọn câu trên cùng văn bản; không ép các objective khác nhau thành một cấu hình | Tab **02 / Tóm tắt văn bản** → form **Văn bản nguồn** → menu **Phương pháp**. Bên phải form, vùng **Phương pháp đang chọn** mô tả objective của nhánh hiện tại. |
| NMFTS-inspired pairwise và `pairwise_lambda` | Phạt độ lệch phân phối topic giữa các câu có liên hệ trong đồ thị; MMR sau đó vẫn giảm câu trùng lặp. Kỳ vọng giữ các câu liên quan trong cùng mạch chủ đề hơn | Tab **02 / Tóm tắt văn bản** → chọn **NMFTS-inspired pairwise KL** → mở **Tham số thuật toán** để chỉnh Pairwise λ. Sau khi chạy, kết quả hiển thị loss tổng, reconstruction và pairwise penalty. |
| SNMF đồ thị câu | Gom câu theo cấu trúc tương tự; lần lượt lấy câu đại diện từ các cụm để tăng độ phủ chủ đề | Tab **02 / Tóm tắt văn bản** → menu **Phương pháp** → **SNMF đồ thị câu**. Kết quả có từ khóa theo cụm và phần **Phân tích từng câu**. SNMF không dùng MMR; vùng phân tích hiển thị score/cụm của phương pháp này. |
| Query PRF | Dùng query người dùng để ưu tiên câu phù hợp với ý định tìm kiếm; nếu query OOV/không khớp, hệ thống báo lý do và vẫn trả tóm tắt Local KL+MMR | Tab **02 / Tóm tắt văn bản** → khi chọn Local KL, nhập vào **Truy vấn (không bắt buộc)**; trọng số xuất hiện khi đã nhập query. Trong kết quả, thông tin pseudo-relevant được ghi dưới summary, và điểm Query xuất hiện trong từng câu nếu PRF được áp dụng. |
| Điều chỉnh alpha W/H và L1 ratio | Có thể khảo sát độ thưa thay vì mặc định cố định; alpha W tác động độ thưa của phân phối câu-topic, alpha H tác động độ thưa từ-topic | Tab **02 / Tóm tắt văn bản** → mở **Tham số thuật toán** khi dùng Local KL-NMF. Giá trị mặc định vẫn là 0.1 / 0.1 / 1.0. |
| Loss theo objective | Tránh nhầm loss KL+L1 với loss Frobenius hoặc loss graph; breakdown cho biết phần tái tạo và phần phạt | Tab **02 / Tóm tắt văn bản** → sau khi chạy, ngay dưới hàng kết quả/k/solver là panel **Objective / ...**. Local hiện KL + L1; NMFTS hiện Frobenius + pairwise symmetric-KL; SNMF hiện symmetric Frobenius. Các con số giữa objective khác nhau không so sánh trực tiếp được. |
| Auto-k và candidate scores | Cho xem k nào được chọn và điểm masked imputation của các ứng viên; lưu ý đây là độ khôi phục TF-IDF, không phải điểm chất lượng summary | Tab **02 / Tóm tắt văn bản** → bỏ trống k, sau khi chạy xem panel **Ứng viên k**. Panel có chú thích chính sách k dùng chung, không tối ưu riêng objective NMFTS/SNMF. |
| Benchmark nhiều phương pháp và validation tuning | So sánh cùng tập bài/ngân sách câu, tune trên validation; lỗi không bị ngụy trang thành điểm ROUGE 0 | Sau khi chạy `scripts/run_experiment.py`, mở tab **03 / Đánh giá** → vùng **Thí nghiệm batch**, bảng trung bình và phần xem kết quả từng bài → nhấn **Làm mới dữ liệu**. Tuning chạy bằng `scripts/tune_params.py`; best config và toàn bộ grid nằm trong `results/tuning/tuning.json`/`tuning.csv`, không có màn hình tune riêng. |

Tab **04 / So sánh NMF** hiện vẫn dành cho đối chiếu Global NMF cũ với Local KL-NMF; các nhánh NMFTS/SNMF/PRF được chọn ở tab 02 và benchmark tại tab 03.

**Hiệu quả đã đo và hiệu quả kỳ vọng:** các tác dụng trong bảng là mục tiêu thuật toán, chưa phải kết luận thực nghiệm. `data/splits/*.jsonl` và `results/metrics.csv` hiện rỗng, nên chưa có ROUGE/runtime để khẳng định nhánh nào tốt hơn. Cần bổ sung reference summary hợp pháp, tune trên validation và chỉ dùng test cho đánh giá cuối.

## File chính

- `backend/app/core/nmf.py`: objective KL/Frobenius, regularization và masked-imputation k selection.
- `backend/app/core/vectorize.py`: TF-IDF cục bộ theo câu.
- `backend/app/core/summarize.py`: Local KL-NMF, MMR, NMFTS-inspired, SNMF và Query PRF.
- `backend/app/core/paper_methods.py`: đồ thị câu và updates cho NMFTS/SNMF.
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
3. Chạy ablation tách riêng KL/Frobenius, L1, pairwise penalty, MMR và Query PRF; giữ nguyên split, ngân sách câu và seed.
4. Nếu mục tiêu là tái hiện paper, triển khai đúng semantic similarity/role representation, tiền xử lý, dataset và siêu tham số gốc trước khi so sánh.
