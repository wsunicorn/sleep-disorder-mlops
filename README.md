# Sleep Disorder Detection - Phan tich du lieu cam bien giac ngu

Tai lieu nay mo ta trang thai hien tai cua project sau khi duoc chinh lai theo
file `notebooks/kaggle_cap_training.ipynb`. Notebook nay duoc xem la "source of
truth" cho du lieu, schema feature, cach train model va cac artifact can phuc vu
API.

## Ket luan nhanh

Project phu hop de lam prototype MLOps cho bai toan "Phan tich du lieu cam bien
giac ngu de phat hien roi loan". Repo da co du cac phan quan trong: xu ly tin
hieu EEG, trich xuat feature, train model 7 lop, API Django, dashboard, mo phong
IoT, Docker, CI/CD, AWS ECS/Fargate va Terraform.

Tuy nhien, project hien van nen duoc xem la demo ky thuat, chua du tot cho ung
dung nghien cuu nghiem tuc hoac y te. Ly do chinh la metric model con trung binh,
danh gia chua theo benh nhan, du lieu tap trung vao EEG/CAP hon la nhieu cam
bien giac ngu, va phan ha tang AWS can duoc quan ly bang Terraform state ro rang
neu da xoa ALB/RDS/ECS resource.

## Chuan du lieu va model hien tai

Notebook `notebooks/kaggle_cap_training.ipynb` su dung bo Balanced CAP CSV va
train model phan loai 7 tinh trang:

| Label | Y nghia |
|---|---|
| `healthy` | Binh thuong |
| `insomnia` | Mat ngu |
| `narcolepsy` | Ngu ru |
| `nfle` | Nocturnal frontal lobe epilepsy |
| `plm` | Periodic leg movement |
| `rbd` | REM behavior disorder |
| `sdb` | Sleep disordered breathing |

Chuan feature duoc thong nhat trong project:

| Thuoc tinh | Gia tri |
|---|---|
| Tin hieu vao | EEG 1 chieu |
| Sampling rate goc | 512 Hz |
| Cua so | 1024 mau, tuong duong 2 giay |
| So feature | 24 |
| Label column | `disease` |
| Artifact chinh | `models/model.pkl` |
| Feature names | `models/feature_names.json` |
| Label encoder | `models/label_encoder.pkl` |
| Metadata | `models/metadata.json` |

24 feature dang duoc dung theo dung thu tu:

```text
delta_power, delta_rel, theta_power, theta_rel, alpha_power, alpha_rel,
beta_power, beta_rel, gamma_power, gamma_rel, spectral_entropy,
peak_frequency, mean_frequency, amplitude_mean, amplitude_std, rms,
delta_beta_ratio, theta_alpha_ratio, skewness, kurtosis, zero_crossing_rate,
hjorth_activity, hjorth_mobility, hjorth_complexity
```

## Project da lam gi

### 1. Xu ly du lieu

`feature_engineering/preprocess.py` doc file EDF bang MNE, chon kenh EEG, loc
notch, loc bandpass 0.5-40 Hz, cat epoch va loai artifact. Dau ra la file `.npz`
trong `data/processed/`.

`feature_engineering/cap_features.py` la module feature dung chung cho toan bo
project. Module nay bam theo notebook Kaggle:

- chia tin hieu thanh cua so 2 giay;
- tinh bandpower delta/theta/alpha/beta/gamma;
- tinh feature ty le, entropy pho, tan so dinh, tan so trung binh;
- tinh thong ke bien do, RMS, skewness, kurtosis;
- tinh zero-crossing rate va Hjorth activity/mobility/complexity;
- tra ve dung 24 feature theo `FEATURE_NAMES`.

`feature_engineering/extract_features.py` chuyen cac file `.npz` da preprocess
thanh `data/features/features.parquet`, su dung cung schema 24 feature va label
`disease`.

### 2. Train model

`training/train.py` da duoc chinh de di theo notebook:

- doc truc tiep `data/raw/balanced_CAP/*.csv` neu co;
- ho tro doc san `features.parquet` neu pipeline da tao truoc;
- validate bat buoc du 24 cot feature;
- encode 7 label bang `LabelEncoder`;
- chia train/validation stratified 80/20;
- tinh sample weight cho class imbalance;
- train va so sanh `XGBoost`, `LightGBM`, `RandomForest`;
- chon model co weighted F1 cao nhat;
- log MLflow va export artifact vao `models/`.

Lenh train chinh:

```bash
python training/train.py --data-dir data/raw/balanced_CAP --model-dir models --model-type all
```

`params.yaml` va `dvc.yaml` cung da duoc dong bo lai voi chuan nay.

### 3. API va dashboard

Ung dung Django nam trong `sleep_portal/`.

API chinh:

- `POST /api/v1/predict/`: nhan batch feature 24 cot va tra ve du doan.
- `POST /api/v1/predict-edf/`: nhan file EDF, loc tin hieu, cat cua so 2 giay,
  trich xuat 24 feature bang module chung, roi predict.
- `POST /api/v1/ingest/`: nhan ket qua tu IoT demo va luu vao database.
- `GET /api/v1/health/`: health check cho ALB/ECS.
- `GET /api/v1/model-info/`: thong tin model dang phuc vu.

`sleep_portal/inference/predictor.py` load model tu MLflow Registry neu co,
fallback ve `models/model.pkl`, load feature names, label encoder va metadata.
Code cung co fallback cho artifact pickle tao bang NumPy 2 nhung phuc vu bang
NumPy 1.x.

### 4. Mo phong IoT

`iot_simulation/demo_local.py` mo phong mot benh nhan bang tin hieu EEG synthetic
va dung dung feature extractor 24 cot.

`iot_simulation/multi_patient_demo.py` mo phong nhieu benh nhan bang
`data/raw/balanced_CAP/feature_stats.json`, goi API `/predict/`, sau do ingest
ket qua vao dashboard.

### 5. Docker va production dependencies

`docker/Dockerfile` build app Django va copy them cac module can thiet:

- `feature_engineering/`
- `training/`
- `monitoring/`
- `models/`
- `params.yaml`

`requirements-prod.txt` da them `mne` va `pyedflib` de endpoint upload EDF co
the hoat dong trong container production.

### 6. CI/CD va AWS

Workflow chinh: `.github/workflows/ci.yml`.

Luang CI/CD:

1. Chay test voi Postgres va Redis service.
2. Tai model artifact tu S3 neu co.
3. Build Docker image.
4. Push image len ECR.
5. Update ECS service.
6. Chay `python manage.py migrate --noinput` bang ECS one-off task.

Workflow da duoc chinh de:

- tai them `label_encoder.pkl`, `feature_names.json`, `metadata.json` tu S3;
- khi deploy thi scale ECS service ve `desired-count 1`, phu hop truong hop da
  tam tat task de tiet kiem chi phi;
- lay subnet/security group cua migration task tu ECS service thay vi hard-code
  ID cu.
- tu query ALB `sleep-portal-alb`, in ra URL that va smoke test `/api/v1/health/`
  cung `/api/v1/model-info/` sau deploy.

Luu y quan trong: neu chi scale ECS service ve 0 thi CI/CD co the bat lai bang
`desired-count 1`. Neu da xoa ALB/target group/service hoac resource ha tang
khac, workflow deploy khong tu tao lai duoc. Luc do can chay Terraform voi state
dung, hoac import lai resource vao state truoc khi apply.

## Danh gia do phu hop voi de tai

Project phu hop voi de tai o muc prototype vi:

- co du lieu giac ngu that tu CAP/Balanced CAP;
- co xu ly tin hieu EEG thay vi chi dung du lieu bang gia lap;
- co bai toan phan loai roi loan cu the, khong chi phan loai sleep stage;
- co pipeline tu train den phuc vu model qua API;
- co mo phong luong IoT gui feature/prediction vao dashboard;
- co CI/CD va deployment len AWS.

Nhung project chua that su "tot" neu danh gia nghiem tuc:

- metric model hien trong `models/metadata.json` chi khoang weighted F1 0.59,
  accuracy 0.59, con thap cho bai toan y te;
- validation dang o muc cua so/row, chua chac da tach theo patient, nen co nguy
  co data leakage neu cac cua so cua cung benh nhan xuat hien o ca train va val;
- feature handcrafted co the chua du manh cho 7 lop roi loan phuc tap;
- endpoint `/predict/` tra nhan du doan nhung chua tra confidence/probability;
- chua co calibration, threshold, uncertainty va giai thich model;
- du lieu "cam bien giac ngu" moi chu yeu la EEG, chua ket hop SpO2, nhip tim,
  ho hap, actigraphy;
- Terraform backend S3 dang comment, nen neu lam nhom/deploy that se de lech
  state;
- neu da xoa ALB that, can recreate ha tang bang Terraform, khong chi push code.

## Can cai thien them

### Data va Machine Learning

1. Tach train/validation/test theo patient, khong tach ngau nhien theo cua so.
2. Tao test set doc lap va bao cao macro F1, per-class recall, confusion matrix.
3. Luu ro mapping patient -> disease -> split de tai lap ket qua.
4. Thu them feature theo chuoi thoi gian: rolling statistics, transition pattern,
   hoac model sequence nhu CNN/Transformer/TCN tren signal/window.
5. Them xac suat du doan va calibration cho API.
6. Them explainability: feature importance, SHAP summary, per-prediction reason.
7. Bo sung du lieu/cam bien khac neu de tai yeu cau "cam bien giac ngu" rong hon
   EEG.

### Engineering va MLOps

1. Dua Terraform backend S3/DynamoDB vao dung that de tranh mat state.
2. Tao workflow rieng cho infrastructure, chay manual `terraform plan/apply`.
3. Quan ly secret qua GitHub Actions secrets hoac AWS Secrets Manager, khong
   hard-code trong task definition.
4. Upload model artifact moi len S3 sau retrain, gom ca `model.pkl`,
   `label_encoder.pkl`, `feature_names.json`, `metadata.json`.
5. Them health check hau deploy: goi `/api/v1/health/` va `/api/v1/model-info/`.
6. Them monitoring metric theo class distribution, drift cua feature va ti le loi
   API.
7. Them rollback image ECS neu deploy fail.

### San pham va demo

1. Dashboard nen hien confusion/metric model, model version va ngay train.
2. Demo IoT nen co che do replay du lieu that tu Balanced CAP thay vi chi sample
   tu mean/std.
3. Upload EDF nen hien so epoch, kenh dung, ti le tung class va timeline du doan.
4. Can canh bao ro "khong dung cho chan doan y khoa" trong UI/demo.

## Cach kiem tra local

Chay test:

```bash
cd sleep_portal
..\venv\Scripts\python.exe -m pytest ..\tests -q
```

Kiem tra compile:

```bash
.\venv\Scripts\python.exe -m compileall feature_engineering training iot_simulation sleep_portal
```

Train lai theo notebook:

```bash
.\venv\Scripts\python.exe training\train.py --data-dir data\raw\balanced_CAP --model-dir models --model-type all
```

Bat lai ECS neu may da cau hinh AWS credentials:

```bash
aws ecs update-service --cluster sleep-portal-cluster --service sleep-portal-service --desired-count 1 --region ap-southeast-1
```

Neu ALB da bi xoa, can khoi phuc bang Terraform/state truoc khi service co URL
public hoat dong lai.
