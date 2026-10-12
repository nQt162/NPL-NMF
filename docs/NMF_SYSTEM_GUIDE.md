# Cẩm nang NMF và hệ thống TopicSum

> Tài liệu nền để đọc hiểu thuật toán, luồng xử lý, các trang web, cách đánh giá và giới hạn hiện tại của dự án. Nội dung mô tả code trong repository tại ngày 12-10-2026; khi code đổi, cần đối chiếu lại các file được dẫn.

## 1. Tóm tắt nhanh

TopicSum là ứng dụng **tóm tắt trích xuất tiếng Việt**. Hệ thống giữ nguyên câu trong văn bản gốc, dùng TF-IDF và NMF để biểu diễn câu theo các chủ đề tiềm ẩn, rồi dùng MMR để chọn một số câu ít trùng lặp hơn. Đây không phải hệ thống sinh văn bản mới bằng LLM.

Luồng tóm tắt chính là **Local NMF**: mỗi bài người dùng gửi được xem như một tập tài liệu nhỏ, trong đó mỗi câu là một document. TF-IDF và NMF được fit lại trên chính bài đó. File `nmf_corpus.joblib` không được dùng trong API tóm tắt hiện tại; nó còn được gọi bởi benchmark ngoại tuyến để làm baseline Global NMF.

Ứng dụng có bốn trang:

1. **Mô phỏng NMF**: xem `V`, `W`, `H`, `WH` và loss qua các vòng MU tự cài đặt.
2. **Tóm tắt văn bản**: chạy hai biến thể Local KL-NMF + L1 + MMR hoặc Local Frobenius-NMF + L1 + MMR.
3. **Đánh giá**: tạo cả hai bản tóm tắt trên cùng cấu hình, rồi so từng bản với cùng reference bằng ROUGE.
4. **Ablation L1**: so sánh KL không L1, L1 trên W, L1 trên H, và L1 trên cả hai.

**Trạng thái dữ liệu cần nói rõ trong báo cáo:** `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl` và `data/raw/articles.jsonl` hiện rỗng; `results/metrics.csv` cũng chưa có benchmark hợp lệ. Vì vậy chưa thể kết luận định lượng rằng KL tốt hơn Frobenius, L1 nâng ROUGE, hay một cấu hình k/alpha là tối ưu trên tập dữ liệu tổng quát.

## 2. Bài toán và phạm vi

### Tóm tắt trích xuất và tóm tắt trừu tượng

- **Trích xuất (extractive)**: chọn câu có sẵn trong nguồn, không viết lại. Ưu điểm là bảo toàn cách diễn đạt và số liệu gốc; nhược điểm là câu được chọn có thể thiếu mạch nối, đại từ tham chiếu hoặc ngữ cảnh.
- **Trừu tượng (abstractive)**: diễn đạt lại nội dung bằng câu mới. Có thể gọn và tự nhiên hơn, nhưng có rủi ro thay đổi hoặc bịa thông tin.

TopicSum làm dạng trích xuất. Nếu kết quả chưa mạch lạc, nguyên nhân có thể là câu nguồn phụ thuộc ngữ cảnh, không nhất thiết là NMF phân rã sai.

### Bài toán mỗi lần chạy

Cho văn bản có `n` câu và một ngân sách tóm tắt `B`, mục tiêu là chọn tối đa `min(B,n)` câu, sắp theo thứ tự gốc. `B` là số câu đầu ra; `k` là số thành phần/chủ đề của NMF. Hai tham số này khác nhau và không có công thức bắt buộc `k = B`.

## 3. NMF: ý tưởng và ký hiệu

NMF tìm hai ma trận không âm để xấp xỉ một ma trận không âm:

```text
V ≈ W H,       V >= 0, W >= 0, H >= 0
```

Trong TopicSum, `V` có chiều **câu × từ/ngữ**:

| Ma trận | Kích thước | Cách đọc trong dự án |
| --- | --- | --- |
| `V` | `n_sentences × n_terms` | Mỗi hàng là một câu; mỗi cột là một unigram/bigram TF-IDF. |
| `W` | `n_sentences × k` | Hàng `i` biểu diễn mức tham gia của câu `i` vào `k` chủ đề. |
| `H` | `k × n_terms` | Hàng `t` biểu diễn trọng số từ/ngữ trong chủ đề `t`. |
| `WH` | `n_sentences × n_terms` | Ma trận TF-IDF được mô hình tái tạo. |

Một cách hình dung: mỗi hàng của `V` là hồ sơ từ vựng của một câu; `W` mô tả câu pha các chủ đề ra sao; `H` mô tả chủ đề gắn với từ nào. Đây là diễn giải **latent factor**, không phải NMF đã tìm được nhãn chủ đề có sẵn.

`k` là rank/số thành phần ẩn của xấp xỉ. Khi `k` nhỏ hơn nhiều so với số câu và số từ, `WH` biểu diễn dữ liệu qua không gian thấp chiều hơn. “Thấp chiều” là nén biểu diễn, không tự động có nghĩa là chất lượng summary cao hơn. `k` quá nhỏ có thể gộp ý khác nhau; quá lớn có thể chia nhỏ/nhiễu và tăng độ phức tạp.

### Không âm có ý nghĩa gì?

Các hệ số không âm khiến biểu diễn là tổ hợp cộng: một từ/câu được giải thích bằng cách cộng đóng góp của các thành phần, không dùng một hệ số âm để triệt tiêu thành phần khác. Điều này thường giúp diễn giải kiểu “parts-based”, nhưng **không đảm bảo** mỗi topic luôn rõ ràng hoặc độc quyền.

SVD/LSA cho phép hệ số âm; dấu âm không phải lỗi toán học. Có thể nói NMF có ràng buộc phù hợp với cách diễn giải cộng tính, không nên nói SVD “sai” hoặc “không thể dùng cho văn bản”.

### `W` và `H` không tự động là xác suất

Các hàng của `W` và `H` không được ép tổng bằng 1 trong mô hình. Giá trị là trọng số nhân tử, không nên đọc trực tiếp như xác suất, tỷ lệ phần trăm hay độ tin cậy. Màn hình có thể chuẩn hóa một số điểm cho mục đích xếp hạng, nhưng đó là bước hậu xử lý.

NMF còn có tính **không xác định**:

- Có thể đổi thứ tự các topic mà `WH` không đổi.
- Có thể nhân `W` với `c` và chia `H` cho `c` mà tích vẫn như cũ.
- Nghiệm phụ thuộc khởi tạo và có thể dừng ở nghiệm cục bộ.

Vì vậy, topic số 1 giữa hai lần fit không nhất thiết là cùng một chủ đề; cũng không nên so trực tiếp trị số riêng lẻ của `W/H` giữa hai lần chạy.

## 4. Dữ liệu đầu vào và vector hóa

Trong [sentences.py](../backend/app/core/sentences.py), văn bản được tách câu bằng regex, giữ nguyên nội dung câu và thứ tự. Đây là bộ tách gọn, không phải bộ phân tích ngữ pháp đầy đủ; viết tắt, số thập phân, dấu câu bất thường hoặc văn bản không chuẩn có thể làm tách câu chưa đúng.

Trong [preprocess.py](../backend/app/core/preprocess.py):

1. Tạo bản sao để xử lý, chuyển chữ thường.
2. Dùng `underthesea.word_tokenize(..., format="text")` để tách từ tiếng Việt.
3. Loại stopword theo `stopwords_vi.txt` khi tạo đặc trưng NMF.
4. Không sửa câu nguyên bản được dùng để xuất summary.

Trong [vectorize.py](../backend/app/core/vectorize.py), TF-IDF fit cục bộ trên các câu của một bài: unigram + bigram, `min_df=1`, `sublinear_tf=True`, chuẩn hóa L2 từng hàng, tối đa 800 đặc trưng. Ma trận cuối được chuyển sang dense `numpy` để phân rã.

**Hệ quả của Local TF-IDF:** vocabulary và topic gắn với bài hiện tại, tránh topic/vocabulary từ corpus toàn cục kéo vào mọi bài. Đổi lại, bài ngắn có ít quan sát, ma trận rất thưa và topic không ổn định để so sánh xuyên nhiều bài. Mỗi bài có không gian từ vựng riêng.

| | Local NMF hiện tại | Global NMF cũ |
| --- | --- | --- |
| Vocabulary | Fit TF-IDF từ câu của bài đang gửi. | Dùng vocabulary/vectorizer đã fit trên corpus. |
| Factors/topics | Fit mới cho từng bài, số chủ đề do k nhập hoặc bộ chọn. | Tái dùng các factor đã học từ model lưu; code benchmark ghi nhận model corpus 3260 bài. |
| Điểm mạnh | Bám từ ngữ cục bộ; bài không bị áp vocabulary toàn cục. | Topic có thể ổn định hơn giữa tài liệu cùng miền và không cần fit factor từ đầu mỗi lần. |
| Đổi lại | Bài ngắn/thưa làm k và topic không ổn định; không so topic trực tiếp giữa bài. | Từ ngoài vocabulary là OOV; topic toàn cục có thể không hợp nội dung bài mới. |
| Trong dự án | API `/api/summarize` và hai method trên web. | Baseline trong `scripts/run_experiment.py` qua `compare.py`; không phải luồng summarize API. |

## 5. Objective: loss tính từ đâu?

Loss được tính từ dữ liệu đầu vào `V` và ma trận tái tạo `WH`, không cần nhãn summary. Nó đo mức mô hình khớp lại ma trận TF-IDF. Trong code, tổng objective có dạng:

```text
J(W,H) = data_fit(V, WH) + regularization(W,H)
```

### Frobenius

```text
D_F(V || WH) = 0.5 * sum_ij (V_ij - (WH)_ij)^2
```

Frobenius phạt bình phương sai lệch tuyệt đối; một sai lệch lớn đóng góp nhiều hơn sai lệch nhỏ. Đây là một **tổng bình phương có hệ số 1/2**, không phải ROUGE và không nhất thiết là MSE đã lấy trung bình theo số ô.

### Generalized KL divergence

```text
D_KL(V || WH) = sum_ij [ V_ij * log(V_ij / (WH)_ij) - V_ij + (WH)_ij ]
```

Với ô `V_ij = 0`, giới hạn của số hạng là `(WH)_ij`. Trong implementation có epsilon để tránh chia cho 0/log(0). KL bất đối xứng: đổi vai trò dữ liệu và reconstruction sẽ đổi kết quả.

Ví dụ một ô `V=1`, `WH=2`:

- Frobenius: `0.5 * (1 - 2)^2 = 0.5`.
- KL: `1 * ln(1/2) - 1 + 2 ≈ 0.307`.

Hai con số khác công thức, không thể từ ví dụ này nói loss nào tốt hơn. Trong dự án, KL là beta-divergence cấu hình cho NMF; **TF-IDF không phải phân phối xác suất đếm từ được chuẩn hóa theo topic**. Vì vậy không nên giải thích KL như “đúng hơn chắc chắn cho mọi văn bản” hay coi đây là bằng chứng KL tốt hơn Frobenius.

### L1/L2 và objective mà scikit-learn dùng

Scikit-learn cộng penalty vào data-fit. Trong cách bố trí `X = V` của project:

```text
L1(W,H) = alpha_W * n_terms * ||W||_1
        + alpha_H * n_sentences * ||H||_1
```

Với `l1_ratio=1`, objective hiện tại là `data_fit + L1(W,H)`. Tổng objective hiển thị ở kết quả tóm tắt gồm data loss và regularization; API còn trả riêng `data_loss` và `regularization_loss` để phân biệt hai phần.

Trong `scikit-learn`, penalty của W được scale theo số feature (`n_terms`) và penalty của H theo số sample (`n_sentences`). Vì thế alpha không thể diễn giải độc lập với kích thước ma trận. Tham số `l1_ratio=0` chuyển sang L2; giá trị giữa 0 và 1 trộn L1/L2; nó không phải “phần trăm phần tử bằng 0”.

### Loss không phải điểm chọn câu hoặc điểm chất lượng summary

- Loss/objective: mức khớp `V ≈ WH` cộng penalty, đo trong quá trình fit NMF.
- MMR score: điểm tham lam để chọn câu, tính sau khi có `W` và `V`.
- ROUGE: so bản tóm tắt máy với reference do người tạo.

Không có reference vẫn tính được loss và vẫn tạo được summary. Muốn đánh giá similarity với bản tóm tắt tham chiếu thì cần reference để tính ROUGE. Không so trực tiếp số objective KL với Frobenius vì thang đo và hình dạng hàm khác nhau.

## 6. Điều chuẩn thưa: L1 đang làm gì?

L1 cộng một khoản tỷ lệ với tổng độ lớn tuyệt đối của các hệ số. Vì `W,H >= 0`, chuẩn L1 tương đương tổng các phần tử. Penalty làm mô hình có xu hướng giảm nhiều hệ số nhỏ và có thể tạo biểu diễn thưa hơn:

- L1 trên `W`: giảm độ phân tán câu-trên-chủ-đề; kỳ vọng câu tập trung hơn.
- L1 trên `H`: giảm số từ/ngữ có trọng số đáng kể trong topic; kỳ vọng topic gọn hơn.
- L1 trên cả hai: tác động lên hai biểu diễn cùng lúc.

Đây là **xu hướng/regularization**, không phải cam kết “mỗi câu chỉ có 1-2 topic” hay mỗi topic có chính xác N từ. L1 không đảm bảo các topic độc quyền hoặc semantic-coherent. Alpha quá lớn có thể làm factor gần 0, làm giảm thông tin và khiến hệ thống fallback Lead-N. Trong demo, `alpha=0.1` đã làm nhánh Frobenius suy biến; mặc định hiện là `0.001` như mức bảo thủ, **chưa được tune/khẳng định tối ưu trên corpus**.

Độ thưa của ablation được đo theo ngưỡng tuyệt đối `1e-8`: tỷ lệ phần tử W/H không vượt ngưỡng, số topic hoạt động trung bình trên mỗi câu, và số từ hoạt động trung bình trong mỗi topic. Ngưỡng này là quy ước đo; đổi thang/chuẩn hóa có thể đổi số đếm. Đọc kèm ROUGE/fallback, không kết luận chỉ từ tỷ lệ zero.

## 7. Solver, initialization và cấu hình thực tế

### Luồng tóm tắt/đánh giá

[summarize.py](../backend/app/core/summarize.py) tạo estimator `sklearn.decomposition.NMF` mới cho mỗi lần fit:

```python
NMF(
    n_components=k,
    init="nndsvdar",
    solver="mu",
    beta_loss="kullback-leibler"  # hoặc "frobenius"
    alpha_W=alpha_w,
    alpha_H=alpha_h,
    l1_ratio=l1_ratio,
    max_iter=500,
    random_state=seed,
)
```

`solver="mu"` là multiplicative update; đây là solver được scikit-learn yêu cầu khi dùng KL/Itakura-Saito beta loss. Frobenius cũng có thể chạy bằng MU. `nndsvdar` là NNDSVD với các giá trị nhỏ ngẫu nhiên để lấp chỗ zero; seed kiểm soát phần ngẫu nhiên. Số vòng thực tế còn phụ thuộc điều kiện hội tụ/tolerance, nên `max_iter=500` là giới hạn trên, không phải lời khẳng định luôn chạy đủ 500 vòng.

Hai method tóm tắt hiện dùng chung `alpha_W`, `alpha_H`, `l1_ratio`, `k`, seed, TF-IDF và MMR; phần chủ đích thay đổi là `beta_loss`: KL so với Frobenius. Dù tham số được ghép cặp, nghiệm có thể khác vì objective khác.

### Luồng trang mô phỏng

`/api/simulate` gọi `fit_nmf` trong [nmf.py](../backend/app/core/nmf.py), một MU tự cài đặt với khởi tạo ngẫu nhiên đều và snapshots qua số vòng do người dùng yêu cầu. Response ghi `implementation="educational_custom_mu"`.

Đây là mô phỏng độc lập để quan sát phép cập nhật và loss; không phải trace từng vòng từ estimator scikit-learn của luồng tóm tắt. Vì khác initialization, công thức triển khai và điều kiện dừng, không dùng đồ thị trang này để khẳng định W/H của production giống hệt từng vòng.

## 8. Cách chọn `k`

Nếu người dùng nhập `k`, code giới hạn nó không vượt quá `min(n_sentences, n_terms)`. Nếu không nhập, [select_k_by_imputation](../backend/app/core/nmf.py) dùng quy trình:

1. Xét các `k` từ 1 đến `min(10, n_sentences, n_terms)`.
2. Lấy vị trí có TF-IDF dương; ở mỗi trial giữ lại ngẫu nhiên khoảng 20% vị trí để validation, 10 trial.
3. Fit custom KL-NMF **không regularization** trên các ô còn lại, với mask để các ô bị giữ lại không tham gia fit.
4. Tính mean generalized-KL trên các phần tử dương bị giữ lại.
5. Chọn `k` có điểm validation nhỏ nhất.

Trường hợp số quan sát dương quá ít, code dùng heuristic `max(1, n_sentences // 2)` (giới hạn theo k khả thi) và trả `score=null`; nếu chỉ có một k khả thi, trả một k khả thi, cũng không tạo score giả. Seed điều khiển các lượt mask/khởi tạo, nên cùng input + seed cho kết quả lặp lại.

**Cách diễn giải đúng:** đây là masked positive-entry KL reconstruction CV, một heuristic chọn độ hạng theo khả năng khôi phục các ô TF-IDF dương bị che. Nó không dùng reference summary, không tối ưu ROUGE, không đo topic interpretability, và không tối ưu riêng cho Frobenius. Do bỏ qua các ô zero trong tập validation, đây cũng không phải đánh giá đầy đủ mọi ô ma trận. Hãy báo rõ tên phương pháp, không gọi nó là k tối ưu tuyệt đối.

`k` là số thành phần tiềm ẩn chứ không phải số câu giữ lại. Với cùng một văn bản/seed, mô phỏng, tóm tắt và đánh giá dùng chung chính sách chọn k; trang Đánh giá truyền k đã chọn cho nhánh thứ hai để ghép cặp.

## 9. Cách xếp hạng topic và giải thích từ khóa

Trong `_topic_terms`:

1. Chuẩn hóa mỗi hàng H thành phân phối tương đối trong topic.
2. Tính độ độc quyền tương đối của từ bằng tỷ lệ trọng số từ đó ở topic này so với tổng qua các topic.
3. Xếp từ theo tích hai đại lượng, hiển thị tối đa 5 cụm từ mỗi topic.
4. Chọn tối đa 2 câu đại diện theo tỷ lệ W chuẩn hóa theo hàng.

Các từ có thể là unigram/bigram sau Underthesea và loại stopword; chúng không phải nhãn tự nhiên. Ví dụ một nhóm `thành phố`, `tuyến xe`, `hành khách` có thể cần người xem đặt tên chủ đề dựa trên câu đại diện. Nếu topic chưa rõ, kiểm tra chất lượng phân câu, stopword, số câu, n-gram, alpha và k; không “sửa nhãn” như thể NMF đã sinh nhãn đúng.

## 10. Chọn câu bằng MMR

MMR trong project dùng relevance theo topic và redundancy theo cosine TF-IDF:

```text
MMR(i) = lambda * relevance(i)
       - (1 - lambda) * max_cosine(V_i, V_j), j thuộc các câu đã chọn
```

Ở lượt đầu, chưa có câu đã chọn nên redundancy bằng 0. Relevance được tính từ W: salience topic tỷ lệ với tổng W theo topic, W mỗi câu được chuẩn hóa theo hàng, sau đó relevance được scale theo giá trị lớn nhất. Đây là điểm xếp hạng nội bộ, không phải xác suất hay điểm ROUGE.

Nếu có query (chỉ nhánh KL hiện hỗ trợ): query được biến thành vector trong vocabulary cục bộ; chọn tối đa ba câu pseudo-relevant, trộn query với centroid câu đó, rồi trộn điểm query relevance với topic relevance theo `query_weight`. Nếu query không khớp vocabulary, backend trả lý do và vẫn chạy MMR thường.

Ví dụ minh họa với `lambda=0.7`:

| Câu ứng viên | Relevance | Redundancy lớn nhất | MMR |
| --- | ---: | ---: | ---: |
| A | 0.82 | 0.00 | `0.7×0.82 - 0.3×0 = 0.574` |
| B | 0.68 | 0.00 | `0.476` |
| C ở lượt 2 | 0.55 | 0.10 | `0.355` |
| B ở lượt 2, sau khi A đã chọn | 0.68 | 0.70 | `0.266` |

Lượt 1 chọn A. Sang lượt 2, C có điểm cao hơn B vì B giống A quá nhiều, dù B có relevance cao hơn. Redundancy không yêu cầu cùng dominant topic: câu bị so độ tương tự với **mọi** câu đã chọn, bất kể nhãn topic của chúng. Hệ thống chọn greedy đúng ngân sách rồi sắp các câu đã chọn theo index gốc để summary đọc theo thứ tự nguồn.

`selection_trace` lưu ứng viên và điểm theo từng vòng. Chỉ so điểm của các câu trong **cùng vòng**; sang vòng sau, tập câu đã chọn thay đổi nên redundancy/điểm thay đổi. Lambda cao ưu tiên relevance; lambda thấp phạt trùng lặp mạnh hơn, nhưng không có một lambda tốt cho mọi văn bản.

## 11. Đánh giá: ROUGE khác loss thế nào?

[evaluate.py](../backend/app/core/evaluate.py) dùng chung tokenizer Underthesea cho mọi phương pháp; với ROUGE giữ dấu tiếng Việt và không bỏ stopword, không dùng English stemmer.

- **ROUGE-1 F1**: overlap unigram, tính precision/recall rồi F1.
- **ROUGE-2 F1**: overlap bigram liên tiếp, nhạy hơn với cụm từ/trật tự cục bộ.
- **ROUGE-L F1**: overlap theo longest common subsequence, có tính đến trật tự nhưng không yêu cầu các token đứng cạnh nhau.

Với mỗi metric, gọi `overlap` là số đơn vị unigram/bigram/LCS khớp theo cách tính của metric:

```text
precision = overlap / số đơn vị trong candidate
recall    = overlap / số đơn vị trong reference
F1        = 2 * precision * recall / (precision + recall)
```

Nếu precision + recall bằng 0 thì F1 bằng 0. Trong project các giá trị được trả ở thang `[0,1]`, giao diện thường định dạng bốn chữ số thập phân.

Các điểm nằm trong `[0,1]`; cao hơn nghĩa là bề mặt từ/ngữ giống reference hơn theo metric đó. ROUGE không trực tiếp đo độ thật, mạch lạc, mức bao phủ mọi ý hay chất lượng diễn đạt; một reference duy nhất cũng có thể không bao quát hết các summary tốt.

Trang **Đánh giá** nhận văn bản nguồn và reference, sinh **hai** summary (KL+L1+MMR và Frobenius+L1+MMR) trên cùng cấu hình, sau đó chấm mỗi summary với cùng reference. Nó không chỉ chấm một summary có sẵn. Mục tiêu là quan sát tác động của objective lên đầu ra dưới cấu hình được ghép cặp, không phải suy luận loss nào tốt hơn chỉ từ trị số objective.

## 12. Ablation L1 và benchmark ngoại tuyến

### Ablation trên trang 04

Bốn điều kiện của `run_l1_ablation`:

| Tên | Alpha W | Alpha H | Câu hỏi kiểm tra |
| --- | ---: | ---: | --- |
| KL không L1 | 0 | 0 | Mốc không regularization. |
| KL + L1 trên W | α | 0 | Tác động phạt biểu diễn câu-topic. |
| KL + L1 trên H | 0 | α | Tác động phạt biểu diễn topic-từ. |
| KL + L1 trên W và H | α | α | Tác động của phạt cả hai factor. |

Các điều kiện được ghép cặp theo bài/seed, dùng cùng k (chọn một lần), ngân sách câu, MMR lambda và query PRF tắt. So ROUGE ghép cặp với baseline, tỷ lệ zero/active của W,H, runtime, fallback. Không xếp hạng các objective khác penalty bằng tổng loss vì penalty khác nhau.

Form web nhận trực tiếp **một bài + reference**, chạy các seed đã nhập; đây là pilot. Nhiều seed trên một bài chỉ cho biết độ nhạy khởi tạo của bài đó, không thay thế nhiều bài độc lập.

### Benchmark CLI

`scripts/run_experiment.py` chạy ngoại tuyến trên JSONL và so sánh Lead-N, Global NMF cũ, Local KL+L1+MMR, Local Frobenius+L1+MMR. Kết quả gồm `metrics.csv`, `predictions.jsonl`, `report.json`. Nó yêu cầu `id`, `article`, `reference_summary`.

Global NMF với 3260 bài được nạp qua `backend/app/core/compare.py` cho **baseline batch**; route `/api/summarize` không gọi module đó. Trên web hiện tại không còn trang so sánh Global NMF và Lead-N. `nmf_corpus.joblib` vẫn được đóng gói/đọc ở nhánh benchmark, nên không mô tả là file đã bị xóa khỏi toàn bộ repo.

## 13. Các trang web và API

| Trang | Backend/API | Dùng để trình bày |
| --- | --- | --- |
| 01 / Mô phỏng NMF | `POST /api/simulate` | Nhập bài, chọn objective/k/seed/số vòng; xem V/W/H/WH, breakdown loss và snapshot custom MU. |
| 02 / Tóm tắt văn bản | `POST /api/summarize` | Chọn method, k hoặc auto, số câu, MMR lambda, seed, alpha W/H, L1 ratio; query tùy chọn chỉ cho KL; xem summary/topic/trace/fallback. |
| 03 / Đánh giá | `POST /api/evaluate` | Nhập nguồn + reference; sinh hai summary cùng cấu hình và trả ROUGE-1/2/L. |
| 04 / Ablation L1 | `POST /api/ablation`, polling job và tải artifact | Chạy các nhánh penalty; form web hiện là một bài pilot. |

FastAPI schemas nằm trong [schemas.py](../backend/app/schemas.py), route trong [main.py](../backend/app/main.py). React điều hướng trong [App.jsx](../frontend/src/App.jsx); các trang giao diện nằm ở `frontend/src/pages/`.

## 14. So sánh dễ nhớ

| Cặp khái niệm | Câu hỏi mà nó trả lời | Có reference cần không? |
| --- | --- | --- |
| KL/Frobenius loss | `WH` tái tạo `V` tốt đến đâu theo objective đã chọn? | Không. |
| L1 penalty | Factor bị phạt tổng độ lớn bao nhiêu, và có xu hướng thưa ra sao? | Không. |
| k masked-KL | Mức độ hạng nào khôi phục tốt hơn các ô TF-IDF dương bị che? | Không. |
| MMR | Trong các câu còn lại, câu nào cân bằng relevance và novelty tốt nhất vòng này? | Không. |
| ROUGE | Summary có overlap từ/ngữ bao nhiêu với reference? | Có. |

Một summary có thể có loss thấp nhưng ROUGE thấp; một summary có ROUGE cao không chứng minh factorization tối ưu. Đây là các bài toán/tiêu chí khác nhau.

## 15. Điều nên và không nên khẳng định

### Có thể nói

- Hệ thống fit TF-IDF/NMF cục bộ cho từng văn bản; summary trích nguyên câu.
- Hai biến thể chính khác nhau ở data-fit KL so với Frobenius, còn L1/MMR và cấu hình được ghép cặp trong trang đánh giá.
- L1 được đưa vào để regularize/khuyến khích factor gọn; tác động phải kiểm tra bằng sparsity và ROUGE.
- Auto-k là masked positive-entry KL heuristic; score không phải chất lượng summary.
- MMR greedy cân bằng topic relevance với cosine redundancy; câu cuối được đưa về thứ tự gốc.
- Trang mô phỏng custom MU, còn fit summary dùng scikit-learn.

### Không nên nói

- “Loss là điểm chất lượng summary”, “loss thấp thì ROUGE cao”, hoặc so trị số KL trực tiếp với Frobenius.
- “L1 làm topic chắc chắn rõ/độc quyền”, “W/H là xác suất”, hoặc “weight lớn luôn xấu”. Alpha quá lớn cũng có thể xóa tín hiệu.
- “k được chứng minh tối ưu”, “k bằng số câu summary”, hoặc “k search tối ưu cho Frobenius/ROUGE”.
- “KL chắc chắn tốt hơn cho TF-IDF”, “đã tái hiện nguyên paper”, hoặc “hệ thống đã thắng baseline” khi chưa có benchmark hợp lệ.
- “Trang mô phỏng là trace sklearn production” hoặc “Global model đã bị xóa hoàn toàn”.
- “Ablation web với một bài và nhiều seed đại diện cho cả corpus”.

## 16. Câu hỏi thường gặp khi báo cáo

**Không có nhãn thì loss lấy ở đâu?** Từ `V` và `WH` của NMF. Đây là objective nội bộ không giám sát; reference chỉ cần khi tính ROUGE.

**Tại sao dùng hai loss?** Để kiểm tra cách đo sai số phân rã có làm thay đổi factor và câu được chọn thế nào; không có loss tốt nhất cho mọi dữ liệu. So đầu ra bằng cùng reference/cấu hình.

**Tại sao L1?** Để phạt tổng độ lớn factor và khuyến khích ít trọng số hoạt động hơn; không phải vì mọi trọng số lớn đều sai.

**Tại sao k ở các trang từng khác nhau?** Trước đây có thể chạy chính sách khác nhau hoặc nhập tay. Code hiện dùng chung selector masked-KL khi `k` bỏ trống. Nếu người dùng nhập k, đó là cấu hình yêu cầu, không phải auto optimum.

**MMR redundancy có bắt buộc trùng topic không?** Không. Hiện code lấy cosine TF-IDF cao nhất với bất kỳ câu đã chọn nào, không lọc theo dominant topic.

**Kết quả dự án đã chứng minh phương pháp nào thắng chưa?** Chưa. Split benchmark đang rỗng; cần dữ liệu có reference, validation để chọn cấu hình, test độc lập để báo cáo.

## 17. Tài liệu và mã nguồn tham khảo

- [Scikit-learn: NMF API và objective](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html): beta loss, MU, alpha scaling, L1/L2, initialization.
- [Lee & Seung, Algorithms for Non-negative Matrix Factorization (NeurIPS 2000)](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html): nền tảng thuật toán cập nhật NMF.
- [Carbonell & Goldstein, The Use of MMR (SIGIR 1998)](https://doi.org/10.1145/290941.291025): relevance-diversity reranking.
- [Lin, ROUGE (ACL 2004)](https://aclanthology.org/W04-1013/): metric overlap cho đánh giá tóm tắt.
- [Aghdam, Automatic Extractive and Generic Document Summarization Based on NMF (2023)](../aghdam_automative.pdf): NMFTS/pairwise sentence regularization; tài liệu paper trong repo.
- [Wang et al., Multi-Document Summarization via Sentence-Level Semantic Analysis and Symmetric Matrix Factorization (SIGIR 2008)](../1390334.1390387.pdf): sentence similarity, symmetric NMF và chọn đại diện.
- [Park & An, Automatic Query-based Personalized Summarization using PRF with NMF](../2108616.2108690.pdf): query-focused summarization và pseudo-relevance feedback.
- [Ghi chú phương pháp/paper của dự án](PAPER_METHODS_AND_EXPERIMENTS.md), [lịch sử cải tiến](NMF_IMPROVEMENTS.md), [thiết kế ablation L1](ABLATION_L1.md).

Các paper trên giúp đặt bối cảnh; code hiện tại chỉ có một số ý tưởng paper-inspired. Cần đối chiếu dữ liệu, objective, preprocessing và protocol gốc trước khi gọi một nhánh là tái hiện paper.
