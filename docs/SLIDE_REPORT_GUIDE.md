# Hướng dẫn làm slide báo cáo TopicSum/NMF

## 1. Mục tiêu của bộ slide

Kể một câu chuyện liền mạch: **bài toán tóm tắt là gì → NMF tạo biểu diễn câu/chủ đề ra sao → loss/L1/k ảnh hưởng gì → MMR chọn câu thế nào → web đánh giá ra sao → hiện có thể kết luận đến đâu**.

Nên dùng slide để giải thích một luận điểm mỗi trang, không chép toàn bộ tài liệu kỹ thuật lên màn hình. File nền để chuẩn bị nội dung chi tiết là [NMF_SYSTEM_GUIDE.md](NMF_SYSTEM_GUIDE.md).

## 2. Tham khảo slide Nhóm 04

File [Slide_Nhom_04.pdf](../Slide_Nhom_04.pdf) có 8 trang và một mạch kể dễ theo dõi:

1. Tiêu đề/chủ đề.
2. Bài toán tóm tắt bằng chọn câu, kèm ví dụ bài báo và phân biệt trích xuất/trừu tượng.
3. Trực giác NMF so với biểu diễn có số âm.
4. Định nghĩa `V ≈ W × H` và giải thích kích thước ma trận.
5. Objective và ý nghĩa sai số.
6. Cập nhật MU, vòng lặp và diễn biến loss.
7. Pipeline từ bài báo đến bản tóm tắt.
8. So sánh, hạn chế, rồi mời xem demo.

Điểm nên học: bắt đầu bằng ví dụ cụ thể, giải thích ký hiệu trước công thức dài, chuyển từ thuật toán sang ứng dụng, và kết thúc bằng hạn chế/demo. **Không bê số liệu minh họa, kết luận thực nghiệm hoặc kích thước ma trận của nhóm đó sang dự án này.** Đặc biệt, dự án này dùng ma trận `câu × từ`, khác quy ước trục `từ × câu` trên một số slide.

## 3. Quy tắc kể chuyện và thiết kế

- Với báo cáo 10-15 phút, dùng khoảng 11-13 slide gồm mở đầu và kết luận; dành khoảng một phút cho slide kỹ thuật, lâu hơn cho demo.
- Một slide có một thông điệp chính. Dùng sơ đồ/công thức để hỗ trợ, phần giải thích đầy đủ để ở speaker notes hoặc tài liệu nền.
- Khi hiện công thức, ngay cạnh đó ghi ký hiệu và chiều ma trận. Đừng để người nghe phải đoán `W`/`H` là hàng hay cột.
- Dùng cùng một ví dụ văn bản 5-7 câu xuyên suốt pipeline, MMR và demo để người nghe thấy cùng dữ liệu được biến đổi ra sao.
- Nếu hiển thị biểu đồ loss từ trang mô phỏng, đặt nhãn **“custom MU minh họa, seed/input cụ thể”**. Không vẽ số liệu giả hoặc gọi đó là log của scikit-learn.
- Dùng màu khác nhau nhất quán cho `V`, `W`, `H`, câu được chọn và câu bị MMR phạt. Không dùng màu sắc thay cho chú giải.
- Bảng so sánh KL/Frobenius nên để cạnh nhau với cùng cấu hình; không đặt hai con số objective trên một trục như thể chúng cùng thang đo.
- Mỗi biểu đồ có tiêu đề, đơn vị/metric, nguồn và số lượng mẫu. Nếu là demo một bài thì ghi “minh họa”, không ghi “kết quả benchmark”.

## 4. Dàn ý chi tiết đề xuất

### Slide 1 — Tiêu đề

**Tiêu đề gợi ý:** “TopicSum: Tóm tắt trích xuất tiếng Việt bằng Local NMF và MMR”.

Đưa tên môn/nhóm/thành viên ở phần phụ; có thể dùng sơ đồ lớn `V ≈ W × H` như dấu hiệu trực quan. Không mở bằng danh sách tham số.

**Câu mở đầu:** “Hệ thống thử biến các câu trong một bài thành biểu diễn câu-chủ-đề, rồi chọn các câu đại diện mà không viết lại nội dung.”

### Slide 2 — Bài toán: giữ câu nào?

Đưa một đoạn nguồn ngắn 4-6 câu, đánh số câu. Đặt cạnh nhau:

- **Trích xuất:** chọn một vài câu nguyên bản và giữ thứ tự.
- **Trừu tượng:** viết lại nội dung thành câu mới.

Nêu rõ đề tài chọn trích xuất để giảm nguy cơ sinh sai dữ kiện; hệ quả là câu summary có thể còn phụ thuộc ngữ cảnh.

**Hình nên dùng:** đoạn văn có 2-3 câu được tô màu, nối sang summary. Không cần công thức ở trang này.

### Slide 3 — NMF và biểu diễn cộng tính

Giới thiệu NMF là phân rã ma trận không âm thành hai ma trận hệ số không âm. Giải thích rằng hệ số không âm giúp diễn giải các thành phần như đóng góp cộng lại; trái lại SVD/LSA vẫn hợp lệ nhưng có thể dùng dấu âm/cộng trừ nên kém trực quan theo cách “parts-based”.

**Tránh nói:** “SVD sai với văn bản”, “âm là không hợp lệ”, hoặc “NMF luôn tạo topic dễ hiểu”. Tính không âm chỉ là ràng buộc của factorization, không bảo đảm coherence.

### Slide 4 — Ma trận của chính hệ thống

Hiển thị đúng quy ước TopicSum:

```text
V: n_câu × n_từ
W: n_câu × k
H: k × n_từ
V ≈ W H
```

Giải thích: hàng `i` của W nói câu `i` liên hệ với các topic ra sao; hàng `t` của H nói topic `t` gắn với từ/ngữ nào. Ghi rõ một ô của `V` là TF-IDF, không phải nhãn.

**Hình nên dùng:** ma trận ô màu nhỏ, có nhãn hàng “câu” và cột “từ”; thêm một hàng W và một hàng H phóng to. Dùng dữ liệu toy nhưng ghi rõ minh họa.

### Slide 5 — Loss và L1: đo gì, không đo gì?

Đặt hai data-fit loss cạnh nhau:

```text
Frobenius: 0.5 * ||V - WH||_F^2
KL:        sum [V log(V / WH) - V + WH]
Objective: data-fit + L1(W,H)
```

Giải thích loss đo độ tái tạo `V`, không chấm bản tóm tắt và không cần reference. L1 phạt tổng độ lớn hệ số, khuyến khích W/H thưa; alpha quá mạnh có thể làm factor co về gần 0.

**Một ô minh họa:** nếu `V=1`, `WH=2`, Frobenius hiện tại đóng góp `0.5`; KL xấp xỉ `0.307`. Đây chỉ là minh họa công thức, không phải kết quả chạy. Không so trị số KL/Frobenius như cùng đơn vị.

### Slide 6 — Cấu hình và cách tối ưu

Nêu cấu hình backend tóm tắt: scikit-learn NMF, `solver="mu"`, `init="nndsvdar"`, `max_iter=500`, seed; KL hay Frobenius chọn qua `beta_loss`; hai nhánh dùng chung alpha W/H và L1 ratio.

Đặt bên cạnh khối nhỏ “Trang mô phỏng”: custom MU để hiện snapshot; “Trang tóm tắt”: estimator scikit-learn fit cục bộ. Hai luồng có mục đích khác nhau, không phải cùng một trace nội bộ.

**Hình nên dùng:** pipeline cập nhật `W,H → WH → objective → cập nhật tiếp`. Nếu dùng chart, lấy trực tiếp từ endpoint/UI với một seed/văn bản cố định, ghi rõ đầu vào và implementation.

### Slide 7 — Từ văn bản đến Local NMF và chọn k

Vẽ pipeline thực tế:

```text
Văn bản → tách câu giữ index → Underthesea/stopwords
→ TF-IDF unigram + bigram theo bài → chọn k → Local NMF
```

Giải thích auto-k ngắn gọn: thử `k=1..min(10,n_câu,n_từ)`, che ngẫu nhiên khoảng 20% ô TF-IDF dương trong 10 trial, fit KL không L1 trên phần còn lại, chọn mean held-out KL nhỏ nhất.

**Cảnh báo ngay trên slide:** đây là tiêu chí tái tạo ô dương, không phải ROUGE hay “k tối ưu ngữ nghĩa”; dữ liệu ít thì hệ thống dùng heuristic và không có score. `k` khác số câu cần giữ.

### Slide 8 — MMR chọn câu ra sao?

Hiển thị công thức:

```text
MMR(i) = λ × relevance(i) - (1-λ) × redundancy(i)
```

Trong project, relevance lấy từ W/topic salience; redundancy là cosine TF-IDF lớn nhất với **bất kỳ câu đã chọn nào**, không yêu cầu hai câu thuộc cùng topic. Chọn tham lam từng câu, cập nhật redundancy, rồi đưa các câu được chọn về thứ tự gốc.

**Ví dụ trình bày:** với `λ=0.7`, A có relevance `0.82`, redundancy `0`: MMR=`0.574`; B có relevance `0.68`, sau khi A được chọn redundancy `0.70`: MMR=`0.266`; C có relevance `0.55`, redundancy `0.10`: MMR=`0.355`. Vòng sau chọn C thay vì câu B gần trùng A. Ghi đây là số minh họa, không phải output benchmark.

Nhắc người nghe điểm MMR chỉ so sánh giữa ứng viên trong cùng vòng.

### Slide 9 — Topic và câu đầu ra

Chụp trang **02 / Tóm tắt văn bản** sau khi chạy cùng đoạn ví dụ. Chỉ vào summary nguyên văn, k, topic terms/câu đại diện, câu được giữ và MMR trace.

Nói rõ danh sách terms là đặc trưng thống kê từ H đã chuẩn hóa/điều chỉnh độ độc quyền; topic chưa có nhãn tự nhiên. Người xem phải diễn giải qua các từ và câu đại diện. Không gọi top terms là “chủ đề đã được phân loại chính xác”.

### Slide 10 — Đánh giá bằng ROUGE

Phân biệt luồng trang 03:

1. Người dùng nhập bài nguồn và một reference summary.
2. Hệ thống sinh hai summary: KL+L1+MMR, Frobenius+L1+MMR; dùng cùng k, số câu, seed, alpha/L1 và MMR.
3. Tính ROUGE-1/2/L F1 cho từng summary so với cùng reference.

ROUGE-1 so unigram; ROUGE-2 so bigram; ROUGE-L so chuỗi con chung dài nhất. Điểm cao hơn thể hiện overlap tốt hơn theo metric, không chứng minh factuality/mạch lạc tuyệt đối. Loss không phải đầu vào tính ROUGE.

**Bảng nên dùng:** hai hàng method, ba cột ROUGE; chỉ chèn số thật từ API hoặc benchmark, ghi sample size. Không tự điền ví dụ thành kết quả thực nghiệm.

### Slide 11 — Ablation L1 và trạng thái thực nghiệm

Trình bày 4 điều kiện trang 04:

| Điều kiện | W bị phạt | H bị phạt |
| --- | --- | --- |
| KL không L1 | Không | Không |
| L1 trên W | Có | Không |
| L1 trên H | Không | Có |
| L1 trên W và H | Có | Có |

Các điều kiện dùng chung bài/seed/k/ngân sách câu/MMR; đọc ROUGE ghép cặp cùng sparsity W/H, fallback và thời gian. Form web hiện nhập một bài, nên là pilot.

Nói minh bạch: split JSONL benchmark hiện rỗng, nên chưa có cơ sở tuyên bố L1 cải thiện ROUGE hay một loss thắng. `data/ablation_demo.jsonl` là ví dụ demo, không phải tập kiểm thử đại diện.

### Slide 12 — Giới hạn và hướng tiếp theo

Chọn 4-5 giới hạn quan trọng:

- Tập validation/test có reference hợp pháp chưa được điền.
- Auto-k dựa trên masked positive-entry KL, không tối ưu trực tiếp ROUGE.
- TF-IDF local của bài ngắn có vocabulary nhỏ/thưa; topic giữa các bài không trực tiếp so sánh.
- Regex tách câu và stopword ảnh hưởng kết quả.
- L1, MMR, k cần tune trên validation; một alpha chưa phải giá trị tối ưu.
- ROUGE nên đi kèm kiểm tra định tính hoặc đánh giá người đọc.

Hướng tiếp: thu thập corpus có quyền, tách split theo bài, tune validation, khóa cấu hình, báo test độc lập; bổ sung ablation/tác động MMR và đánh giá chất lượng bằng người đọc.

### Slide 13 — Kết luận / demo / hỏi đáp

Chốt ba đóng góp thực tế: local vocabulary/factorization, so sánh hai objective trong điều kiện ghép cặp, và giao diện quan sát NMF/MMR/ROUGE. Kết lại bằng một câu trung thực: đây là một hệ thống nghiên cứu/demo có pipeline chạy thật, nhưng hiệu quả tổng quát còn chờ benchmark đủ dữ liệu.

## 5. Cách trình bày phần paper

Có thể dành một slide phụ/backup cho các nghiên cứu đã đọc:

| Nghiên cứu | Ý tưởng chính để kể | Quan hệ với TopicSum |
| --- | --- | --- |
| Lee & Seung, MU/NMF | NMF và các quy tắc cập nhật nhân tử; nền tảng thuật toán. | Trang mô phỏng có MU tự cài; luồng tóm tắt dùng sklearn. Không gọi đây là tái hiện toàn bộ paper. |
| Aghdam (2023), NMFTS | Thêm regularization quan hệ cặp câu vào NMF cho tóm tắt. | Hiện không phải một trong hai method web; paper-inspired branch cũ đã rút khỏi giao diện. |
| Wang et al. (SIGIR 2008), symmetric NMF | Dùng semantic sentence similarity và symmetric factorization để gom câu/chọn đại diện. | Đồ thị SNMF đã rút khỏi web; không đồng nhất với NMF `V≈WH` hiện tại. |
| Park & An, query-based PRF with NMF | Dùng truy vấn và pseudo-relevance feedback để ưu tiên nội dung cá nhân hóa. | Query PRF hiện là tùy chọn đơn giản trên cả KL và Frobenius; cần gọi là paper-inspired, không tuyên bố tái hiện đúng protocol. |

Chỉ nêu những chi tiết từ paper đã đọc/kiểm tra. Không nói rằng paper “chứng minh KL+L1+MMR của project tốt nhất” nếu paper không nghiên cứu đúng cấu hình đó.

## 6. Kịch bản demo web 3-4 phút

1. **Trang 01:** nạp `data/demo/toy.json`, giữ seed cố định, chạy KL rồi Frobenius. Chỉ cách đọc V/W/H/WH và loss theo vòng. Nhắc custom MU.
2. **Trang 02:** dùng cùng văn bản, auto-k, lấy 2-3 câu; chỉ topic terms/câu đại diện, trace MMR và thứ tự gốc. Giải thích `k` khác summary sentence count.
3. **Trang 03:** nhập reference phù hợp với cùng bài rồi tạo hai summary; đọc ROUGE như overlap, không so hai loss.
4. **Trang 04:** nếu đủ thời gian, nhập bài + reference từ `data/ablation_demo.jsonl`, chạy cùng seed hoặc vài seed; gọi đây là demo/pilot một bài.

Trước khi thuyết trình, chạy thử toàn bộ thao tác trên đúng máy/mạng và xác nhận Vite đang proxy tới backend đang chạy. `frontend/vite.config.js` mặc định proxy sang cổng `8001`; nếu backend chạy cổng khác, đặt `VITE_API_PROXY_TARGET` **trước khi khởi động Vite**. Dùng URL mà Vite in ra, đừng mặc định `5173` nếu cổng đó đã bị chiếm.

Ví dụ PowerShell, từ hai terminal riêng:

```powershell
# Backend, từ thư mục backend; chọn một cổng còn trống.
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Frontend, từ thư mục frontend; proxy phải khớp cổng backend bên trên.
$env:VITE_API_PROXY_TARGET = "http://127.0.0.1:8001"
npm.cmd run dev
```

Nếu cổng đã có tiến trình, không khởi chạy chồng lên đó; xác định backend đang nghe cổng nào hoặc chọn cổng mới và cập nhật proxy cho đúng.

## 7. Những hình/đồ thị nên chuẩn bị

- **Sơ đồ pipeline:** câu → token/TF-IDF → `V` → NMF → W/H → relevance/redundancy → câu summary.
- **Ma trận minh họa:** màu sắc phân biệt trục câu/từ/topic; ghi số chiều thật của một lần chạy demo.
- **Hai loss:** công thức và một ô tính mẫu; không dùng một trục chung để so số KL/Frobenius.
- **Loss theo iteration:** ảnh chụp/response thật của custom simulator, seed/input ghi kèm.
- **MMR:** bảng score theo vòng hoặc ảnh chụp trace; đánh dấu câu bị phạt vì cosine similarity.
- **ROUGE:** bảng 2 method × 3 metric; thêm số bài thành công/fallback và cấu hình.
- **Ablation:** ROUGE ghép cặp và sparsity W/H; phải phân biệt pilot với benchmark corpus.

Không cần dùng biểu đồ “sàn SVD”, phần trăm variance hay số liệu của slide mẫu trừ khi dự án tự tính cùng protocol và có thể giải thích nguồn.

## 8. Câu hỏi thường gặp từ hội đồng

**Loss tính mà không có nhãn bằng cách nào?** Loss tính từ chênh lệch giữa TF-IDF `V` và reconstruction `WH`; nó là tiêu chí fit không giám sát. Reference chỉ cần cho ROUGE.

**Loss nhỏ có nghĩa summary tốt không?** Không. Nó chỉ nói reconstruction tốt hơn theo objective; summary score phụ thuộc chọn câu và reference.

**Vì sao vẫn hiển thị loss ở hai method?** Để xem optimizer đang tối ưu gì trong từng method. KL và Frobenius khác công thức/thang đo; hãy so ROUGE/đánh giá người đọc để bàn về đầu ra.

**L1 có làm W/H tốt hơn không?** Nó regularize và khuyến khích sparse factors. Có thể tăng khả năng diễn giải nhưng cũng có thể làm mất tín hiệu; cần ablation trên validation/test.

**k có phải số topic thật của bài không?** Không nhất thiết. Đây là rank/siêu tham số của factorization; auto-k hiện chọn theo masked reconstruction heuristic.

**MMR có lọc trùng topic không?** Không. Hiện redundancy là cosine TF-IDF cao nhất với mọi câu đã chọn.

**Dự án đã thắng phương pháp nào chưa?** Chưa thể kết luận tổng quát khi benchmark splits còn rỗng. Có thể demo chức năng, nhưng không trình bày demo một văn bản như kết quả khoa học.

## 9. Checklist trước khi nộp slide

- [ ] Mọi slide dùng cùng orientation `V: câu × từ`, `W: câu × k`, `H: k × từ`.
- [ ] Tách rõ objective/loss, MMR score và ROUGE.
- [ ] Ghi rõ trang mô phỏng dùng custom MU, production summary dùng scikit-learn NMF.
- [ ] Mô tả auto-k là masked positive-entry KL heuristic, không gọi là ground-truth k.
- [ ] Khi so hai loss, cùng văn bản, k, seed, alpha/L1, MMR và số câu; không so objective value trực tiếp.
- [ ] Mọi số ROUGE có reference, số bài và split; không có số liệu thì ghi “chưa có benchmark”.
- [ ] Mọi hình minh họa ghi nguồn; mọi số mô phỏng ghi seed/config; không chép số của slide mẫu.
- [ ] Chạy thử bốn trang, backend/frontend, đúng proxy/cổng; có phương án nếu mạng hoặc server lỗi.
- [ ] Chuẩn bị câu trả lời về local vs global, L1, MMR, chọn k, ROUGE và giới hạn dataset.

## 10. Tài liệu dùng để chuẩn bị

- Slide mẫu: [Slide_Nhom_04.pdf](../Slide_Nhom_04.pdf).
- Tài liệu nền hệ thống: [NMF_SYSTEM_GUIDE.md](NMF_SYSTEM_GUIDE.md).
- Các ghi chú dự án: [NMF_IMPROVEMENTS.md](NMF_IMPROVEMENTS.md), [PAPER_METHODS_AND_EXPERIMENTS.md](PAPER_METHODS_AND_EXPERIMENTS.md), [ABLATION_L1.md](ABLATION_L1.md).
- [Scikit-learn NMF API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).
- [Lee & Seung, Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html).
- [Carbonell & Goldstein, MMR](https://doi.org/10.1145/290941.291025).
- [Lin, ROUGE](https://aclanthology.org/W04-1013/).
