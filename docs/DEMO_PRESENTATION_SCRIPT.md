# Kịch bản demo TopicSum theo web hiện tại

> Cập nhật và chạy kiểm tra ngày **15-10-2026**. Mẫu mặc định là `test_nmf_01`: **12 câu nguồn, reference trích xuất 4 câu**, không còn bài xe buýt. Kết quả dưới đây được chạy lại từ core và API TestClient trên working tree hiện tại, không phải benchmark corpus.
>
> Môi trường: scikit-learn 1.7.2, Underthesea 9.5.0, NumPy 2.2.6, rouge-score 0.1.2. Thay text, stopword, seed, tham số hoặc phiên bản có thể làm kết quả đổi. Các thông tin và số liệu thực tế trong bài AI là nội dung mẫu do người dùng cung cấp, chưa được xác minh độc lập.

## 1. Mục tiêu và cách dùng

Demo cần làm rõ: NMF học biểu diễn, MU tối ưu W/H, MMR chọn câu, PRF điều chỉnh ưu tiên và ROUGE so output với reference. Không dùng loss thấp để khẳng định summary tốt, không biến một bài thử thành bằng chứng ưu thế tổng quát.

| Cần mở nhanh | Đến phần |
| --- | --- |
| Chạy server | [Chuẩn bị](#chuan-bi) |
| Văn bản và reference | [Dữ liệu](#du-lieu) |
| Ma trận, loss và k | [Trang 01](#mo-phong) |
| Chọn câu và PRF | [Trang 02](#tom-tat) |
| Hai output và ROUGE | [Trang 03](#rouge) |
| Kiểm tra riêng L1 | [Trang 04](#ablation) |
| Lời nói ngắn | [Kịch bản 5-7 phút](DEMO_SPOKEN_SCRIPT_5_7_MIN.md) |

<a id="chuan-bi"></a>

## 2. Chuẩn bị trước giờ báo cáo

Repo mặc định backend 8001, nhưng lần kiểm tra gần nhất backend NMF dùng **8003** vì 8001 thuộc ứng dụng SUN397 khác. Không dừng ứng dụng khác để lấy cổng.

Terminal backend, chạy từ bất kỳ thư mục nào:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\Documents\CODE\NMF\NPL-NMF\scripts\start_backend.ps1" -Port 8003
```

Terminal frontend:

```powershell
Set-Location 'F:\Documents\CODE\NMF\NPL-NMF\frontend'
$env:VITE_API_PROXY_TARGET = 'http://127.0.0.1:8003'
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Nếu đã có đúng server đang chạy thì dùng lại, không mở bản trùng cổng. Nếu dùng 8001, đổi cả `-Port` lẫn proxy về 8001. Biến proxy chỉ áp dụng cho terminal và tiến trình frontend đó, không sửa mặc định của repo.

```powershell
Invoke-RestMethod 'http://127.0.0.1:8003/health'
Invoke-RestMethod 'http://127.0.0.1:5173/health'
```

Cả hai phải trả `status=ok`. `/openapi.json` của backend đúng có tên **TopicSum API** và các route simulate/summarize/evaluate/ablation. `/health` trả 404 có thể do nối nhầm ứng dụng.

Mở `http://127.0.0.1:5173/`. Giữ các terminal server mở. Script chỉ thay backend được nhận diện thuộc đúng dự án/cổng; restart làm mất registry job ablation của tiến trình cũ.

Chạy thử bốn trang, tải report ablation làm bản dự phòng. Điều hướng hiện tạo lại component, nên k/query/kết quả không tự chuyển giữa các trang. Sau cập nhật mẫu, tải lại web; nút **Nạp ví dụ** khôi phục dữ liệu mẫu trong trang.

<a id="du-lieu"></a>

## 3. Dữ liệu và cấu hình thống nhất

Dữ liệu đầy đủ ở [toy.json](../data/demo/toy.json). `text` là 12 câu nối theo thứ tự; `sentences` giữ từng câu; `extractive_summary` là reference được cung cấp, không phải output NMF.

| Câu | Nội dung chính do người đọc diễn giải | Trong reference |
| --- | --- | --- |
| S1 | AI trở thành động lực kinh tế số tại Việt Nam | Có |
| S2 | Đầu tư siêu máy tính và trung tâm dữ liệu | Không |
| S3 | Mô hình ngôn ngữ tiếng Việt trong doanh nghiệp | Không |
| S4 | Ứng dụng học máy vào nông nghiệp | Không |
| S5 | Cảm biến, thị giác máy tính, sâu bệnh và độ mặn | Có |
| S6 | Dự báo thời tiết, năng suất và phân bón | Không |
| S7 | Triển khai học sâu phân tích ảnh y khoa | Không |
| S8 | Sàng lọc bệnh và hỗ trợ chẩn đoán | Có |
| S9 | Rào cản thiếu nhân lực | Không |
| S10 | Nguồn cung kỹ sư chưa đáp ứng nhu cầu | Có |
| S11 | Bảo mật dữ liệu và đạo đức | Không |
| S12 | Phối hợp đại học, viện và doanh nghiệp | Không |

Reference dùng đúng **S1, S5, S8, S10**; index JSON là `[0,4,7,9]`. Các nhóm ý trong bảng không phải nhãn topic được cung cấp cho NMF.

Trang 01/02/03 có sẵn bài nguồn mới. Trang 03 có sẵn reference; **Nạp ví dụ** khôi phục nguồn, reference và ngân sách 4 câu. Trang 04 vẫn để trống nguồn/reference và mặc định 3 câu: cần dán dữ liệu và đổi thành **4**, không tải JSON lên web.

| Cấu hình | Thiết lập cho demo chính |
| --- | --- |
| k trên trang 01/02/03 | Để trống để chạy auto-k |
| Trang 01 | KL, 10 vòng, seed 42 |
| Trang 02/03 | 4 câu, MMR lambda 0.7, alpha W/H 0.001, L1 ratio 1 |
| Query ban đầu | Để trống |
| Thử PRF | Frobenius, query `ung thư phổi`, query weight 0.5 |
| Trang 03 | Seed 42, không query cho cả hai nhánh |
| Trang 04 | ID `test_nmf_01`, **k nhập tay 3**, 4 câu, alpha 0.001, lambda 0.7, seeds `42,43,44` |

Phần 4-6 dùng **auto-k=1**. Phần 7 dùng **k=3 cố định** để khảo sát penalty trong cùng rank; không gọi 3 là k được auto chọn, không so thẳng bảng ablation với output k=1 để quy khác biệt cho L1.

<a id="mo-phong"></a>

## 4. Trang 01: mô phỏng, loss và chọn k

### 4.1. Mở đầu và ma trận

**Thao tác:** mở Mô phỏng NMF, bấm Nạp ví dụ nếu cần; để k trống, giữ KL, 10 vòng, seed 42 rồi bấm **Chạy mô phỏng**.

> Đây là bài mẫu 12 câu về AI trong kinh tế, nông nghiệp, y tế và các thách thức triển khai. Em không gán các nhóm ý đó làm nhãn cho NMF; mỗi câu được coi là một document để fit TF-IDF cục bộ.

**Đã kiểm tra:** V có **12 hàng, 321 đặc trưng** unigram/bigram. Auto-k chọn 1, nên W là **12×1**, H là **1×321**, WH là **12×321**. Các cột không phải 321 từ viết cách trong bản gốc.

Nếu muốn xem nhiều thành phần, nhập k=3 và chạy lại: W thành 12×3, H thành 3×321. Gọi đây là **cấu hình chọn tay để quan sát**, không phải kết quả auto hay cam kết tách đúng ba nhóm ngữ nghĩa.

### 4.2. Thanh vòng lặp và objective

Kéo từ vòng 0 đến vòng 10. Với KL, auto-k=1:

| Vòng | Data loss | L1 penalty | Tổng objective |
| ---: | ---: | ---: | ---: |
| 0 | 1066.703661 | 4.198276 | 1070.901937 |
| 1 | 153.374098 | 2.359197 | 155.733296 |
| 5 | 153.365100 | 2.109975 | 155.475075 |
| 10 | 153.357631 | 1.877725 | 155.235355 |

Làm tròn riêng từng thành phần có thể gây lệch một đơn vị ở chữ số cuối khi cộng trên màn hình.

> Loss đo khả năng tái tạo V bằng WH cộng chi phí regularization, không cần nhãn summary. Mỗi vòng custom MU giữ W để cập nhật H rồi dùng H mới cập nhật W; web lưu sau cả hai bước, chưa hiển thị riêng từng nửa vòng.

Trang 02 dùng sklearn với `nndsvdar` và điều kiện dừng riêng; trang 01 dùng random và custom MU. Đây **không phải nhật ký của chính lần fit tạo summary**. Sklearn MU đang cài cập nhật W trước H; thứ tự khác không tự làm một bản sai hay tốt hơn.

Với Frobenius, cùng k=1/seed/10 vòng, objective từ **264.117992** xuống **5.804056**. Không so trực tiếp với 155.235355 để kết luận Frobenius thắng: hai objective khác thang đo.

### 4.3. Ứng viên k

| k | Mean held-out KL |
| ---: | ---: |
| 1 | 3.073278 |
| 2 | 3.166094 |
| 3 | 3.253927 |
| 4 | 3.262430 |
| 5 | 3.266727 |
| 6 | 3.273322 |
| 7 | 3.283471 |
| 8 | 3.946506 |
| 9 | 3.712653 |
| 10 | 4.304974 |

Giao diện sắp theo score, nên k=9 có thể đứng trước k=8.

> Bộ chọn thử k bằng 10 lượt giữ lại khoảng 20% ô TF-IDF dương; các ô đó bị loại khỏi objective fit, không coi chúng là số 0 thật, rồi kiểm tra bằng KL trung bình. Kết quả 1 là rank tốt nhất theo tiêu chí này, không khẳng định bài chỉ có một chủ đề ngữ nghĩa; nó cũng không phải số câu summary.

Không gọi đây là tái hiện nguyên cách chọn K của NMFTS hay block bi-cross-validation. Đây là chính sách positive-entry masked KL CV của dự án, có hạn chế với dữ liệu thưa và được dùng chung cho cả hai loss.

<a id="tom-tat"></a>

## 5. Trang 02: câu trích, MMR và query

### 5.1. Tạo summary không query

**Thao tác:** Nạp ví dụ, k trống, 4 câu, tham số mặc định, query trống; chạy KL rồi Frobenius.

| Method, k=1 | Data loss | Penalty | Objective | Thứ tự greedy | Thứ tự trong summary |
| --- | ---: | ---: | ---: | --- | --- |
| KL + L1 + MMR | 153.334189 | 1.222446 | 154.556635 | S8 → S6 → S11 → S4 | S4, S6, S8, S11 |
| Frobenius + L1 + MMR | 5.492791 | 0.316336 | 5.809127 | S10 → S7 → S3 → S2 | S2, S3, S7, S10 |

Cả hai đủ 4 câu, nguyên văn và không fallback trong lần kiểm tra. Không có threshold điểm làm rơi câu trước ngân sách.

> W cung cấp phân bố chủ đề để tính relevance. MMR chọn câu có điểm tốt nhất trong vòng hiện tại, ưu tiên liên quan và trừ phần trùng với tập đã giữ, rồi sắp summary lại theo nguồn.

Từ khóa là đặc trưng thống kê, không phải nhãn ngữ nghĩa. Ở k=1, KL có các từ `doanh_nghiệp`, `dữ_liệu`, `học`, `máy_tính`, `công_nghệ`; không nói đã tách đủ các lĩnh vực. Relevance ở k=1 gần bằng nhau; chênh lệch cực nhỏ do chuẩn hóa/epsilon vẫn có thể ảnh hưởng thứ tự.

### 5.2. Một vòng MMR thật

Chọn **Frobenius**, không query, mở trace MMR, xem vòng 2 sau khi đã chọn S10:

| Ứng viên | Relevance | Redundancy với S10 | MMR |
| --- | ---: | ---: | ---: |
| S7 | 0.999117 | 0.020520 | 0.693226 |
| S3 | 0.999869 | 0.031214 | 0.690544 |

> S3 có relevance nhỉnh hơn, nhưng S7 ít giống câu S10 hơn, nên sau `0.7×relevance - 0.3×redundancy`, S7 được chọn. Redundancy xét tất cả câu đã chọn, không chỉ câu vừa chọn và không yêu cầu cùng topic.

Chỉ so ứng viên **trong cùng vòng**; bảng cuối có thể chứa score ở các `score_round` khác nhau.

### 5.3. PRF dùng được cho cả hai loss

**Thao tác:** giữ Frobenius, k trống; query **ung thư phổi**, trọng số 0.5; chạy lại.

- Pseudo-relevant là **S8** (index 7).
- Summary đổi từ **S2, S3, S7, S10** sang **S2, S3, S4, S8**.
- Objective vẫn **5.809127**.

> Query dùng vocabulary của bài; PRF mở rộng bằng centroid của các câu gần query rồi trộn query relevance với topic relevance trước MMR. S8 được ưu tiên nhưng NMF không học lại theo nhãn, không tìm dữ liệu ngoài bài.

Chuyển sang KL vẫn giữ query/trọng số. Với k=1, KL đã có S8 nên tập câu không đổi khi thêm query này; PRF vẫn áp dụng, objective vẫn 154.556635. **Có query không bắt buộc đổi output**.

Query OOV như `qzxv987654321` được báo lý do bỏ PRF và dùng relevance/MMR thường. Trang 03 không kế thừa query từ trang 02.

<a id="rouge"></a>

## 6. Trang 03: reference và ROUGE

**Thao tác:** mở Đánh giá, Nạp ví dụ; k trống, 4 câu, seed 42, tham số mặc định; bấm **Tạo và chấm hai bản tóm tắt**.

| Method | Câu giữ lại | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
| --- | --- | ---: | ---: | ---: |
| KL + L1 + MMR | S4, S6, S8, S11 | 0.3897 | 0.2694 | 0.3282 |
| Frobenius + L1 + MMR | S2, S3, S7, S10 | 0.3979 | 0.2646 | 0.3246 |

> Reference là S1, S5, S8, S10 được cung cấp trước, không phải bản gốc và không phải output NMF đã suy luận sẵn. Hai nhánh tự tạo summary rồi được so với cùng reference.

> ROUGE-1 xét token đơn, ROUGE-2 xét bigram, ROUGE-L xét chuỗi con chung dài nhất; số hiển thị là F1 từ 0 đến 1. Bài này Frobenius cao hơn R1, KL cao hơn R2/L; không có một nhánh thắng cả ba.

Không gọi 0.3979 là đúng 39.79% thông tin. ROUGE không trực tiếp kiểm chứng sự thật/mạch lạc; reference trích xuất ưu tiên những câu người viết đã chọn. Không sửa reference để làm một method thắng.

API dùng chung k, không query. Trang này sinh hai outputs chứ không nhận hai candidates có sẵn. Response evaluate chưa có fallback reason riêng từng nhánh; khi nghi ngờ, kiểm tra trang 02/report thay vì mặc định mọi output đều là NMF thành công.

<a id="ablation"></a>

## 7. Trang 04: ablation L1

**Thao tác:** dán nguồn/reference từ mẫu mới, ID `test_nmf_01`; **k=3 nhập tay**, 4 câu, alpha 0.001, lambda 0.7, seeds `42,43,44`; chạy bốn cấu hình và chờ Hoàn tất.

> Rank ba ở phần này là cấu hình cố định để khảo sát penalty, không phải auto-k đã chọn ba. Bốn nhánh cùng k, TF-IDF, ngân sách, MMR và seed trong mỗi cặp; chỉ đổi vị trí phạt L1, query luôn tắt.

Số đo ngày 15-10-2026, một bài qua ba seed, mean ± population SD:

| Cấu hình | ROUGE-1 | ROUGE-2 | ROUGE-L | Delta R1 | Delta R2 | Delta RL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KL không L1 | 0.612277 ± 0.004395 | 0.518893 ± 0.001072 | 0.567988 ± 0.005494 | 0 | 0 | 0 |
| L1 trên W | 0.209424 ± 0 | 0.010582 ± 0 | 0.115183 ± 0 | -0.402853 | -0.508311 | -0.452804 |
| L1 trên H | 0.606061 ± 0 | 0.520408 ± 0 | 0.575758 ± 0 | -0.006216 | +0.001516 | +0.007770 |
| L1 trên W và H | 0.209424 ± 0 | 0.010582 ± 0 | 0.115183 ± 0 | -0.402853 | -0.508311 | -0.452804 |

Các nhánh thành công 3/3, không fallback/lỗi. W gần 0 **0.666667**, H gần 0 **0.658359**, topic/câu **1**, feature/topic **109.666667** ở cả bốn nhánh, theo ngưỡng tuyệt đối `1e-8`.

> L1 trên H tăng nhẹ R2/L nhưng giảm R1; phạt W hoặc cả hai giảm mạnh cả ba metric trong cấu hình này. Cùng tỷ lệ gần zero không có nghĩa weights hoặc summary giống nhau. L1 không cứ thêm vào là tốt hơn; cần đọc scoring, scale, outputs và metric.

Không đổi alpha chỉ để có bảng xanh, không xếp hạng bằng objective khác penalty, không so trực tiếp bảng k=3 với trang 03 k=1 để quy kết cho L1. Ba seed của một bài không phải ba bài độc lập. One-sided regularization còn có vấn đề scale W/H; đây là **pilot**, không có chứng nhận ý nghĩa thống kê.

Tải CSV metrics, JSONL predictions và JSON report. Web lưu riêng `results/ablation/<run_id>/`; lượt core để viết tài liệu dùng thư mục tạm, không điền kết quả vào metrics benchmark. Trang 04 không tự nạp mẫu hay đổi ngân sách sang 4.

## 8. Thí nghiệm phụ: k=3 trên trang 02/03

Cùng bài/reference/seed; đây không phải cấu hình được chọn sau tuning:

| Method, k nhập tay 3 | Objective | Câu giữ lại | R1 | R2 | RL |
| --- | ---: | --- | ---: | ---: | ---: |
| KL + L1 + MMR | 88.362459 | S2, S3, S4, S12 | 0.2094 | 0.0106 | 0.1152 |
| Frobenius + L1 + MMR | 5.362207 | S1, S4, S8, S9 | 0.5521 | 0.4947 | 0.5104 |

KL giảm objective từ 154.556635 ở k=1 xuống 88.362459 ở k=3 nhưng ROUGE giảm: tái tạo tốt hơn không bảo đảm khớp reference hơn. Không suy ra một loss luôn tốt hơn từ ví dụ này.

## 9. Chốt demo và xử lý câu hỏi

> Hệ thống đã có bốn luồng: quan sát phân rã, tóm tắt có giải thích chọn câu, chấm hai output và ablation có kiểm soát. Nó cho phép kiểm tra quyết định, nhưng độ tin cậy ngữ nghĩa của topic, chính sách k và hiệu quả tổng quát vẫn cần corpus và đánh giá độc lập.

| Câu hỏi/tình huống | Cách xử lý |
| --- | --- |
| Nhiều lĩnh vực sao k=1? | Rank tốt theo masked-KL CV, không phải đếm chủ đề; k khác là cấu hình chọn tay nếu tự nhập. |
| Loss trang 01 và 02 khác? | Khác initialization, implementation, thứ tự cập nhật và số vòng/dừng; không cùng một lần fit. |
| W/H có phải phần trăm? | Không, chỉ là hệ số không âm; chuẩn hóa scoring không biến cả mô hình thành xác suất. |
| PRF của Frobenius đổi loss? | Không trong flow hiện tại: query bổ sung relevance sau fit. |
| Query không đổi câu? | Không bắt buộc đổi; ở mẫu KL k=1, S8 đã được giữ. Xem cờ PRF và điểm Query. |
| L1 làm điểm giảm là lỗi? | Không tự suy ra lỗi; xem outputs, scale/scoring, fallback và báo đúng kết quả. |
| 404 hoặc health sai | Kiểm tra backend TopicSum và proxy; không dừng nhầm ứng dụng khác. |
| ECONNREFUSED | Backend chưa chạy hoặc proxy sai cổng; kiểm tra health trực tiếp và qua frontend. |
| Điểm lệch bảng | Kiểm tra text/reference, k, seed, penalty, lambda, query, tokenizer và phiên bản; đọc kết quả đang chạy. |
| Job mất sau restart | Registry nằm trong tiến trình; giữ artifact đã tải, chạy job mới khi cần. |

## 10. Checklist trước khi trình bày

- [ ] Đúng backend TopicSum và proxy, không nhầm SUN397.
- [ ] Mẫu 12 câu, V 12×321 trong môi trường ghi đầu file.
- [ ] Reference S1/S5/S8/S10, ngân sách 4 trên trang 02/03.
- [ ] Demo chính để trống k; không đọc bảng k=3 cho kết quả k=1.
- [ ] Trang 03 cùng cấu hình, không query cho hai nhánh.
- [ ] Trang 04 dán dữ liệu, nhập k=3/4 câu/ba seed và công bố pilot.
- [ ] Phân biệt loss, score k, MMR và ROUGE; không tuyên bố thắng tổng quát.
- [ ] Có report dự phòng; đã đọc [cẩm nang](NMF_SYSTEM_GUIDE.md), [hỏi-đáp](NMF_DEFENSE_QA.md), [ghi chú ngộ nhận](NMF_MISCONCEPTIONS_AND_KEY_NOTES.md).
