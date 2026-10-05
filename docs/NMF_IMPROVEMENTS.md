# Tong hop cac cai tien NMF

Tai lieu nay tom tat cac thay doi da thuc hien cho project NMF/TopicSum, muc tieu cua tung thay doi, hieu qua so voi phien ban cu va vi tri code lien quan.

## 1. Tong quan truoc va sau

| Hang muc | Truoc khi sua | Sau khi sua | Hieu qua |
| --- | --- | --- | --- |
| Luong tom tat | Dung global NMF model tu `nmf_corpus.joblib` da train san tren corpus lon | Fit NMF cuc bo tren chinh van ban nguoi dung nhap | Tom tat bam noi dung cuc bo, khong bi lan tu vung cua corpus cu |
| Don vi tai lieu | Van ban nguoi dung bi dua vao model toan cuc | Moi cau trong van ban la mot document rieng | Phu hop hon voi extractive summarization cho van ban ngan |
| TF-IDF | Phu thuoc vocabulary/model da train tu truoc | Tao `TfidfVectorizer` moi va `fit_transform` tren cac cau dau vao | Loai bo loi out-of-vocabulary do vocabulary cu |
| Loss NMF | Frobenius/MSE mac dinh hoac objective cu | KL divergence + L1 regularization | Phu hop hon voi du lieu van ban thua, topic sac net hon |
| Solver | Mac dinh cua sklearn | `solver="mu"` | Dung yeu cau cua sklearn khi dung `beta_loss="kullback-leibler"` |
| Sparsity | Khong co L1 penalty ro rang | `alpha_W=0.1`, `alpha_H=0.1`, `l1_ratio=1.0` | Ep W va H thua hon, giam tu khoa nhieu va topic chong cheo |
| Chon k | Heuristic/fixed k | Masked KL imputation neu nguoi dung de trong k | Chon k dua tren kha nang tai tao du lieu bi che, thuc te hon heuristic cung |
| Chon cau | De bi phu thuoc diem don gian tu W | Greedy scoring gom salience, coverage, redundancy, vi tri va do dai | Giam trung lap, tang bao phu y chinh, giu thu tu goc khi xuat summary |
| Hien thi frontend | Loss/k de gay hieu nham voi cau hinh cu | Hien thi KL + L1, KL data, L1 penalty, candidate k, topic/cau | Nguoi dung thay duoc backend dang dung objective moi |

## 2. Bo global model `nmf_corpus.joblib`

### Van de cu

Phien ban cu load mot NMF model toan cuc tu file `nmf_corpus.joblib`. Model nay da hoc vocabulary va ma tran chu de tu corpus lon, nen khi nguoi dung nhap van ban ngan, cac topic va tu khoa co the den tu corpus cu thay vi tu doan van hien tai.

### Cai tien da lam

- Xoa luong phu thuoc vao `joblib.load()` va global corpus model trong flow tom tat.
- Xoa file code wrapper global model `backend/app/core/corpus_model.py`.
- Go bo package-data lien quan den `nmf_corpus.joblib` trong `backend/pyproject.toml`.
- API tom tat goi truc tiep local summarizer.

### File lien quan

- `backend/app/core/corpus_model.py`: da xoa.
- `backend/pyproject.toml`: bo dong dong goi `nmf_corpus.joblib`.
- `backend/app/main.py:114`: endpoint `/api/summarize` goi ham `summarize(...)`.
- `backend/app/core/summarize.py:217`: ham `summarize(...)` la luong tom tat moi.

## 3. Local NMF on-the-fly cho tom tat

### Van de cu

Global model khong phan anh cac y chinh cuc bo cua doan van ngan. Khi vocabulary cua nguoi dung khac vocabulary corpus train, ket qua transform de lech.

### Cai tien da lam

Luong moi trong `backend/app/core/summarize.py`:

1. Tach cau tu text dau vao.
2. Tien xu ly tung cau.
3. Tao ma tran TF-IDF cuc bo tren chinh danh sach cau.
4. Chon hoac suy ra `k`.
5. Fit NMF moi tren ma tran cuc bo.
6. Lay topic terms tu ma tran H moi.
7. Chon cau va ghep lai theo thu tu goc.

### Hieu qua

- Topic va tu khoa den tu van ban hien tai.
- Tom tat hop ly hon voi van ban ngan.
- Khong con hien tuong lay tu khoa ngoai ngu canh tu corpus cu.

### File lien quan

- `backend/app/core/summarize.py:217`: ham tong `summarize(...)`.
- `backend/app/core/summarize.py:254`: chon hoac resolve `k`.
- `backend/app/core/summarize.py:258`: fit local KL-NMF.
- `backend/app/core/summarize.py:259`: chon cau theo greedy scoring.
- `backend/app/core/summarize.py:269`: ghep summary theo thu tu cau goc.

## 4. Doi objective sang KL divergence + L1 sparsity

### Van de cu

Frobenius/MSE xem sai so tai tao nhu binh phuong khoang cach Euclidean. Voi du lieu text sparse nhu TF-IDF, cach nay thuong tao topic kem sac net hon va de co tu khoa nhieu.

### Cai tien da lam

Cau hinh NMF moi:

```python
NMF(
    n_components=k,
    init="nndsvda",
    solver="mu",
    beta_loss="kullback-leibler",
    alpha_W=0.1,
    alpha_H=0.1,
    l1_ratio=1.0,
    max_iter=500,
    random_state=seed,
)
```

### Hieu qua

- KL divergence phu hop hon voi du lieu khong am, thua va co tinh phan phoi nhu tan suat/TF-IDF.
- `solver="mu"` la solver phu hop khi dung KL trong sklearn.
- L1 tren W giup moi cau tap trung vao it chu de hon.
- L1 tren H giup moi topic co bo tu khoa dac trung hon.
- `max_iter=500` giup MU solver co nhieu vong lap hon de hoi tu.

### File lien quan

- `backend/app/core/nmf.py:14`: khai bao `SPARSE_KL_LOSS_NAME = "kullback-leibler"`.
- `backend/app/core/nmf.py:15`: khai bao `SPARSE_KL_SOLVER = "mu"`.
- `backend/app/core/nmf.py:16`: khai bao `SPARSE_KL_ALPHA_W = 0.1`.
- `backend/app/core/nmf.py:17`: khai bao `SPARSE_KL_ALPHA_H = 0.1`.
- `backend/app/core/nmf.py:18`: khai bao `SPARSE_KL_L1_RATIO = 1.0`.
- `backend/app/core/nmf.py:19`: khai bao `SPARSE_KL_MAX_ITER = 500`.
- `backend/app/core/summarize.py:109`: khoi tao sklearn `NMF`.
- `backend/app/core/summarize.py:112`: gan `solver=SPARSE_KL_SOLVER`.
- `backend/app/core/summarize.py:113`: gan `beta_loss=SPARSE_KL_LOSS_NAME`.
- `backend/app/core/summarize.py:114`: gan `alpha_W=SPARSE_KL_ALPHA_W`.
- `backend/app/core/summarize.py:115`: gan `alpha_H=SPARSE_KL_ALPHA_H`.
- `backend/app/core/summarize.py:116`: gan `l1_ratio=SPARSE_KL_L1_RATIO`.
- `backend/app/core/summarize.py:117`: gan `max_iter=SPARSE_KL_MAX_ITER`.

## 5. Loss moi trong mo phong NMF

### Van de cu

Frontend van hien thi loss cu, lam nguoi dung tuong rang backend chua doi objective.

### Cai tien da lam

- Backend tra ve `loss_name`, `solver`, `alpha_W`, `alpha_H`, `l1_ratio`.
- Snapshot tra ve them:
  - `loss`: total loss.
  - `data_loss`: phan KL data loss.
  - `regularization_loss`: phan L1 penalty.
- Frontend hien thi ro `KL + L1 loss`, `KL data`, `L1 penalty`.

### Hieu qua

- Nguoi dung doc dung ban chat loss hien tai.
- Co the tach duoc phan loi tai tao du lieu va phan penalty dieu chuan.
- Gia tri loss la so thap phan binh thuong, nhung khong nen so truc tiep voi Frobenius loss cu vi objective da khac.

### File lien quan

- `backend/app/core/nmf.py:44`: `Snapshot` co `data_loss` va `regularization_loss`.
- `backend/app/core/nmf.py:76`: `objective_breakdown(...)` tinh data loss va regularization.
- `backend/app/main.py:89`: API tra `loss_name`.
- `backend/app/main.py:90`: API tra `solver`.
- `backend/app/main.py:91`: API tra `alpha_W`.
- `backend/app/main.py:92`: API tra `alpha_H`.
- `backend/app/main.py:93`: API tra `l1_ratio`.
- `backend/app/main.py:104`: snapshot tra `data_loss`.
- `backend/app/main.py:105`: snapshot tra `regularization_loss`.
- `backend/app/schemas.py:25`: schema `SnapshotResponse`.
- `frontend/src/pages/Visualizer.jsx:141`: hien thi ten loss.
- `frontend/src/pages/Visualizer.jsx:143`: block breakdown loss.
- `frontend/src/pages/Visualizer.jsx:144`: hien thi `KL data`.
- `frontend/src/pages/Visualizer.jsx:145`: hien thi `L1 penalty`.

## 6. Cai tien cach tu dong tim k

### Van de cu

Cong thuc `k = max(1, len(sentences) // 2)` don gian, de sai voi van ban co it/nhieu chu de that. Fixed k cung khong linh hoat.

### Cai tien da lam

Neu nguoi dung khong nhap k:

1. Lay cac o TF-IDF duong.
2. Che mot phan cac o nay lam validation.
3. Fit NMF voi nhieu ung vien k.
4. Tai tao cac o bi che.
5. Tinh KL imputation error.
6. Chon k co diem thap nhat.

### Hieu qua

- K duoc chon dua tren du lieu dau vao, khong phai chi dua vao so cau.
- Giam nguy co chon k qua lon cho van ban ngan.
- Giam nguy co ep tat ca van ban vao 1 topic khi co nhieu cum y ro.

### File lien quan

- `backend/app/core/nmf.py:20`: `K_SELECTION_MAX_K = 6`.
- `backend/app/core/nmf.py:21`: `K_SELECTION_TRIALS = 3`.
- `backend/app/core/nmf.py:22`: `K_SELECTION_VALIDATION_FRACTION = 0.2`.
- `backend/app/core/nmf.py:305`: ham `select_k_by_imputation(...)`.
- `backend/app/core/nmf.py:326`: gioi han ung vien k.
- `backend/app/core/nmf.py:327`: dat lower bound k cho van ban co du du lieu.
- `backend/app/core/nmf.py:341`: tao danh sach ung vien k.
- `backend/app/core/nmf.py:359`: tinh KL error tren phan bi che.
- `backend/app/core/nmf.py:368`: sap xep va chon k tot nhat.
- `backend/app/core/summarize.py:97`: tom tat goi `select_k_by_imputation(...)`.
- `backend/app/main.py:61`: simulate goi `select_k_by_imputation(...)`.
- `frontend/src/pages/Visualizer.jsx:156`: hien thi cac candidate k.
- `frontend/src/pages/Summarizer.jsx:132`: hien thi cac candidate k trong trang tom tat.

## 7. Cai tien cach chon cau tom tat

### Van de cu

Neu chi lay cau co tong trong so W cao nhat, summary co the bi trung lap y, bo sot topic phu hoac chon cau qua ngan/qua dai.

### Cai tien da lam

Ham chon cau moi dung greedy scoring:

- `relevance`: cau co lien quan manh den topic salience.
- `coverage_gain`: cau bo sung topic chua duoc phu.
- `redundancy`: phat cau qua giong cau da chon.
- `position_prior`: uu tien nhe cau o dau van ban.
- `length_quality`: tranh cau qua ngan hoac qua dai.

Sau khi chon xong, cac cau duoc sap xep lai theo index goc truoc khi ghep summary.

### Hieu qua

- Summary bot lap y.
- Bao phu nhieu chu de hon.
- Van ban tom tat doc troi chay hon vi giu thu tu xuat hien ban dau.

### File lien quan

- `backend/app/core/summarize.py:124`: ham `_select_sentences(...)`.
- `backend/app/core/summarize.py:137`: tinh topic salience.
- `backend/app/core/summarize.py:151`: tinh relevance.
- `backend/app/core/summarize.py:152`: tinh coverage gain.
- `backend/app/core/summarize.py:154`: tinh redundancy.
- `backend/app/core/summarize.py:166`: cong thuc score tong.
- `backend/app/core/summarize.py:188`: cap nhat coverage sau khi chon cau.
- `backend/app/core/summarize.py:190`: sap xep cau da chon theo thu tu goc.
- `backend/app/core/summarize.py:269`: ghep summary tu cac cau theo thu tu goc.

## 8. Mo rong API contract

### Cai tien da lam

Schema API duoc mo rong de frontend/doc/debug biet ro backend dang dung cau hinh nao:

- `k`.
- `loss_name`.
- `solver`.
- `alpha_W`.
- `alpha_H`.
- `l1_ratio`.
- `k_selection_method`.
- `k_candidates`.
- `data_loss`.
- `regularization_loss`.

### Hieu qua

- Frontend khong can doan cau hinh backend.
- De debug khi thay loss/k khac ky vong.
- Phu hop hon voi yeu cau giai thich qua trinh NMF.

### File lien quan

- `backend/app/schemas.py:25`: `SnapshotResponse`.
- `backend/app/schemas.py:36`: `KCandidateResponse`.
- `backend/app/schemas.py:40`: `SimulateResponse`.
- `backend/app/schemas.py:82`: `SummarizeResponse`.
- `backend/app/main.py:88`: response `/api/simulate`.
- `backend/app/core/summarize.py:270`: response `/api/summarize`.

## 9. Cai tien frontend

### Cai tien da lam

- Trang mo phong ghi ro NMF dang dung KL + L1.
- Trang tom tat de trong k mac dinh de backend tu chon.
- Hien thi candidate k va diem tung candidate.
- Hien thi topic terms tu H cuc bo.
- Hien thi phan tich tung cau: topic chinh, salience, coverage, redundancy va score.
- Hien thi loss breakdown thay vi chi mot so loss chung.

### Hieu qua

- Giao dien trung thuc hon voi backend moi.
- Nguoi dung nhin duoc vi sao he thong chon k va chon cau.
- Tot hon cho demo, bao cao va giai thich thuat toan.

### File lien quan

- `frontend/src/App.jsx:58`: mo ta luong TF-IDF cuc bo, KL divergence va L1 sparsity.
- `frontend/src/pages/Visualizer.jsx:91`: mo ta masked imputation va KL/L1.
- `frontend/src/pages/Visualizer.jsx:141`: hien thi loss.
- `frontend/src/pages/Visualizer.jsx:143`: hien thi breakdown.
- `frontend/src/pages/Visualizer.jsx:156`: hien thi candidate k.
- `frontend/src/pages/Summarizer.jsx:77`: mo ta luong tom tat moi.
- `frontend/src/pages/Summarizer.jsx:89`: input k co placeholder tu chon.
- `frontend/src/pages/Summarizer.jsx:132`: hien thi candidate k.
- `frontend/src/styles.css:139`: style loss breakdown.
- `frontend/src/styles.css:150`: style candidate k.

## 10. Kiem thu da chay

Da chay cac kiem thu chinh:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests
npm run build
npm test
```

Ket qua:

- Backend tests: 14 passed.
- Frontend tests: 2 passed.
- Frontend build: thanh cong.

## 11. Danh gia muc do dung voi paper va thuc te

### Diem dung huong

- KL divergence la mot mo rong NMF phu hop cho du lieu khong am va co tinh phan phoi.
- Multiplicative Update la solver dung khi dung KL divergence trong sklearn.
- L1 regularization tren W/H dung muc tieu tao sparsity.
- Local NMF dung ban chat extractive summarization cho van ban ngan.
- Chon k bang masked imputation thuc te hon heuristic cung.

### Gioi han con lai

- TF-IDF + KL la cach lam thuc dung, nhung neu muon theo xac suat chat hon co the thu count matrix hoac normalized term-frequency.
- `alpha_W=0.1` va `alpha_H=0.1` la tham so khoi dau hop ly, chua phai ket qua tuning tren dataset lon.
- Diem summary can benchmark bang ROUGE/BERTScore tren tap test de ket luan chat luong khach quan.
- Mo phong W/H trong frontend phu hop giai thich thuat toan, con sklearn NMF moi la solver chinh cho tom tat.

## 12. Tom lai

Project da chuyen tu mo hinh global NMF sang local KL-NMF co L1 sparsity. Day la thay doi quan trong ve ban chat: he thong khong con dung topic/vocabulary cua corpus cu de tom tat van ban moi, ma hoc topic truc tiep tu chinh doan van nguoi dung nhap. Ket qua ky vong la topic sat noi dung hon, tu khoa gon hon, cach chon k minh bach hon va summary it trung lap hon.
