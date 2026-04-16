# Sleep Disorder Detection — MLOps Pipeline

> **CAP Sleep Database · Django · AWS ECS · MLflow · Evidently AI · MQTT IoT**

Hệ thống MLOps end-to-end phát hiện **7 loại rối loạn giấc ngủ** từ tín hiệu EEG, mô phỏng luồng dữ liệu IoT thời gian thực qua MQTT, huấn luyện model trên Kaggle/SageMaker, và triển khai qua Django REST API trên AWS ECS Fargate với tự động retraining khi phát hiện data drift.

---

## Live Deployment

| Endpoint | URL |
|---|---|
| **Web Dashboard** | `http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/` |
| **Health Check** | `GET /api/v1/health/` |
| **Predict (features)** | `POST /api/v1/predict/` |
| **Predict (EDF file)** | `POST /api/v1/predict-edf/` |
| **Model Info** | `GET /api/v1/model-info/` |

---

## Kiến trúc tổng thể

```
┌─────────────────── Layer 1: IoT Simulation ────────────────────────┐
│  Python Simulator → Mosquitto MQTT → AWS S3 (.npy) + PostgreSQL    │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────── Layer 2: Training ──────────────────────────────┐
│  MNE-Python · Feature Builder (24 features) · DVC + S3            │
│  Kaggle GPU · MLflow Registry (XGBoost / LightGBM / RandomForest) │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────── Layer 3: Serving ───────────────────────────────┐
│  GitHub Actions CI/CD → Docker + ECR → ECS Fargate → AWS ALB      │
│  Django Web App · Inference Service · ElastiCache Redis            │
│  Dashboard · Patients · Inference Studio · Pipeline Status         │
└────────────────────────────┬───────────────────────────────────────┘
                             ▼
┌─────────────────── Layer 4: Monitoring ────────────────────────────┐
│  CloudWatch · Evidently AI · Prefect · EventBridge · Auto-retrain  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Model hiện tại

| Thuộc tính | Giá trị |
|---|---|
| **Model** | LightGBM (best of 3: XGBoost / LightGBM / RandomForest) |
| **Tên trong MLflow** | `sleep-disorder-classifier` |
| **Validation F1** | 0.5929 |
| **Validation Accuracy** | 59.1% |
| **Số features** | 24 (spectral + time-domain, single EEG channel) |
| **Cửa sổ phân tích** | 2 giây / 1024 mẫu (512 Hz) |
| **Dữ liệu training** | CAP Sleep Database — Kaggle CSV (~140k epochs) |
| **Model file** | `models/model.pkl` (LightGBM) |

### 24 Features

| # | Tên feature | Mô tả |
|---|---|---|
| 0 | `delta_power` | Công suất băng Delta (0.5–4 Hz) |
| 1 | `delta_rel` | Công suất tương đối Delta / tổng |
| 2 | `theta_power` | Công suất băng Theta (4–8 Hz) |
| 3 | `theta_rel` | Công suất tương đối Theta |
| 4 | `alpha_power` | Công suất băng Alpha (8–13 Hz) |
| 5 | `alpha_rel` | Công suất tương đối Alpha |
| 6 | `beta_power` | Công suất băng Beta (13–30 Hz) |
| 7 | `beta_rel` | Công suất tương đối Beta |
| 8 | `gamma_power` | Công suất băng Gamma (30–40 Hz) |
| 9 | `gamma_rel` | Công suất tương đối Gamma |
| 10 | `spectral_entropy` | Entropy phổ — đo độ phức tạp tín hiệu |
| 11 | `peak_frequency` | Tần số chiếm ưu thế (Hz) |
| 12 | `mean_frequency` | Tần số trung bình theo trọng số công suất |
| 13 | `amplitude_mean` | Biên độ tuyệt đối trung bình |
| 14 | `amplitude_std` | Độ lệch chuẩn biên độ |
| 15 | `rms` | Root Mean Square |
| 16 | `delta_beta_ratio` | Tỉ số Delta/Beta — chỉ số buồn ngủ |
| 17 | `theta_alpha_ratio` | Tỉ số Theta/Alpha — chỉ số ngủ gật |
| 18 | `skewness` | Độ xiên phân phối tín hiệu |
| 19 | `kurtosis` | Độ nhọn phân phối tín hiệu |
| 20 | `zero_crossing_rate` | Tỉ lệ đổi dấu tín hiệu |
| 21 | `hjorth_activity` | Hjorth Activity — phương sai tín hiệu |
| 22 | `hjorth_mobility` | Hjorth Mobility — tần số trung bình |
| 23 | `hjorth_complexity` | Hjorth Complexity — độ phức tạp sóng |

### 7 Classes (Sleep Disorders)

| Label | Bệnh lý | Số bản ghi gốc |
|---|---|---|
| `healthy` | Không rối loạn | 16 subjects |
| `nfle` | Nocturnal Frontal Lobe Epilepsy | 40 subjects |
| `rbd` | REM Behavior Disorder | 22 subjects |
| `plm` | Periodic Leg Movements | 10 subjects |
| `insomnia` | Insomnia | 9 subjects |
| `narcolepsy` | Narcolepsy | 5 subjects |
| `sdb` | Sleep-Disordered Breathing | 4 subjects |

---

## Dữ liệu: CAP Sleep Database

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn chính** | [PhysioNet CAP Sleep DB](https://physionet.org/content/capslpdb/1.0.0/) |
| **Nguồn training** | Kaggle CAP Sleep Database (pre-processed CSV) |
| **Số bản ghi** | 108 polysomnographic recordings |
| **Tín hiệu** | EEG (F3/F4, C3/C4, O1/O2), EOG, EMG, airflow, SpO2, ECG |
| **Sampling rate** | 512 Hz |
| **Epoch size** | 2 giây = 1024 mẫu |

### Cân bằng dữ liệu (training)

Do dữ liệu mất cân bằng (nfle ~50k, sdb ~1.5k), notebook áp dụng per-class cap:

- `nfle` → 20,000 epochs
- `rbd` → 20,000 epochs
- Còn lại → giữ nguyên toàn bộ

---

## Cấu trúc thư mục

```
project/
├── README.md
├── dvc.yaml                        # DVC pipeline stages
├── params.yaml                     # Hyperparameters
├── requirements.txt
├── requirements-prod.txt
├── requirements-dev.txt
│
├── models/                         # Model artifacts (được COPY vào Docker)
│   ├── model.pkl                   # LightGBM (best model — dùng trong production)
│   ├── model.ubj                   # XGBoost native format (backup)
│   ├── label_encoder.pkl           # Sklearn LabelEncoder cho 7 classes
│   ├── feature_names.json          # 24 tên features theo đúng thứ tự
│   └── metadata.json               # Thông tin model: F1, accuracy, classes
│
├── mlruns/                         # MLflow local tracking (được COPY vào Docker)
│   ├── 660434770358903185/         # CAP Sleep experiment
│   │   └── <run_id>/               # Mỗi run: XGBoost / LightGBM / RandomForest
│   └── models/
│       └── sleep-disorder-classifier/  # Model Registry (6 versions)
│
├── notebooks/
│   └── kaggle_cap_training.ipynb   # Training notebook (chạy trên Kaggle GPU)
│
├── iot_simulation/                 # Layer 1: IoT
│   ├── simulator.py                # Đọc .edf → publish MQTT
│   └── subscriber.py               # Nhận MQTT → lưu S3 + PostgreSQL
│
├── feature_engineering/            # Layer 2: Feature extraction (PhysioNet path)
│   ├── preprocess.py               # Bandpass filter, epoch
│   ├── extract_features.py         # Spectral/statistical features
│   ├── annotation_parser.py        # Parse .txt sleep stage annotations
│   └── build_dataset.py            # Ghép features + labels → parquet
│
├── training/                       # Layer 2: Training
│   ├── train.py                    # XGBoost training (SageMaker compatible)
│   ├── sagemaker_train.py          # Launch SageMaker training job
│   └── register_model.py           # Đăng ký model lên MLflow Registry
│
├── sleep_portal/                   # Layer 3: Django Web App
│   ├── manage.py
│   ├── sleep_portal/
│   │   ├── settings/
│   │   │   ├── base.py             # Shared settings
│   │   │   ├── development.py      # SQLite, local mlruns, DEBUG=True
│   │   │   └── production.py       # ECS settings, env-vars driven
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── dashboard/                  # Web UI
│   │   ├── views.py
│   │   ├── models.py               # Patient, EpochPrediction
│   │   └── templates/dashboard/
│   │       ├── base.html           # Navbar + layout
│   │       ├── home.html           # KPIs + activity feed
│   │       ├── predict.html        # Inference Studio (4 tabs)
│   │       ├── pipeline.html       # Model registry + CI/CD status
│   │       ├── patient_list.html
│   │       └── patient_detail.html
│   ├── api/
│   │   ├── views.py                # PredictView, PredictEDFView, Health, ModelInfo
│   │   ├── serializers.py          # Validate input (feature count từ model)
│   │   └── urls.py
│   └── inference/
│       └── predictor.py            # Singleton model loader + predict() + Redis cache
│
├── docker/
│   ├── Dockerfile                  # Multi-stage: builder → production image
│   ├── docker-compose.local.yml    # Full local stack: Django + Redis + DB + MQTT
│   ├── docker-compose.mqtt.yml     # MQTT-only stack
│   └── mosquitto.conf              # Mosquitto config (allow anonymous, port 1883)
│
├── monitoring/
│   ├── drift_detection.py          # Evidently AI drift report (S3 → JSON)
│   ├── retrain_flow.py             # Prefect retraining orchestration
│   └── promote_rules.py            # Auto-promote nếu F1 ≥ threshold
│
├── infrastructure/                 # Terraform IaC
│   ├── main.tf                     # Provider + VPC + subnets + NAT gateway
│   ├── ecr.tf                      # ECR repository
│   ├── ecs.tf                      # ECS cluster + task + service (Fargate)
│   ├── alb.tf                      # Application Load Balancer
│   ├── rds.tf                      # PostgreSQL (db.t3.micro)
│   ├── iam.tf                      # Task execution roles
│   ├── cloudwatch.tf               # CloudWatch alarms
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars.example
│
├── tests/
│   ├── conftest.py                 # Fixtures: django_client, sample_epoch, sample_features
│   ├── test_api.py                 # API endpoint tests (20 tests)
│   ├── test_features.py            # Feature extraction tests (9 tests)
│   └── test_inference.py           # Predictor unit tests (12 tests)
│
└── .github/workflows/
    ├── ci.yml                      # Push → Test → Build → ECR → ECS deploy
    ├── monitoring.yml              # Weekly drift detection
    └── retrain.yml                 # Manual retrain trigger
```

---

## Cách hệ thống vận hành (end-to-end)

### Luồng IoT đến Prediction

```
[Thiết bị đeo EEG / file .edf]
       │
       ▼
[iot_simulation/simulator.py]
  - Đọc file .edf bằng MNE-Python
  - Bandpass filter 0.5–40 Hz
  - Cắt thành epochs 30 giây
  - Publish từng epoch qua MQTT (topic: sleep/sensor/<patient_id>)
  - Payload JSON: {epoch_index, data, sfreq, channels, timestamp, patient_id}
       │
       │ MQTT broker (Mosquitto port 1883)
       ▼
[iot_simulation/subscriber.py]
  - Subscribe topic sleep/sensor/#
  - Lưu epoch data (.npy) lên AWS S3
    → s3://bucket/raw-epochs/<patient_id>/epoch_XXXXX.npy
  - Ghi metadata vào PostgreSQL
    → epoch_metadata(patient_id, epoch_index, timestamp, s3_key)
       │
       ▼
[Feature Engineering Pipeline]
  - Tải epoch từ S3
  - Tính 24 features/epoch (Welch PSD + Hjorth + thống kê)
  - Lưu feature parquet vào S3
       │
       ▼
[POST /api/v1/predict/]
  - Nhận feature vector (24 floats/epoch, tối đa 256 epochs)
  - Kiểm tra Redis cache (SHA-256 hash của features)
  - Cache miss: MLflow pyfunc → LightGBM.predict()
  - Cache hit: trả ngay, không gọi model
  - Lưu kết quả vào Redis (TTL 1 giờ)
       │
       ▼
[Response]
  {
    "predicted_class": "healthy",   ← class của epoch đầu tiên
    "predictions": [...],           ← class của từng epoch
    "prediction_count": N,
    "class_counts": {...},
    "cached": false
  }
```

### Luồng EDF Upload trực tiếp

```
POST /api/v1/predict-edf/ (multipart file)
       │
       ▼
[PredictEDFView]
  1. Ghi file vào tempdir
  2. MNE read_raw_edf()
  3. Bandpass filter 0.5–40 Hz
  4. Chọn kênh EEG đầu tiên
  5. Cắt windows 2 giây (1024 mẫu @ 512 Hz)
  6. Tính 24 features/window
  7. Gọi predict() → batch prediction
  8. Trả về predictions + metadata
```

### Luồng CI/CD

```
git push → main
  └─► GitHub Actions ci.yml
        ├── Job 1: test (pytest 41 tests)
        └── Job 2: build-and-deploy (chỉ sau test pass)
              ├── docker build -f docker/Dockerfile
              ├── docker push ECR
              └── aws ecs update-service --force-new-deployment
```

### Luồng Monitoring & Retraining

```
Mỗi thứ Hai 03:00 UTC
  └─► monitoring.yml
        └── drift_detection.py
              ├── Load reference features (S3)
              ├── Load current week features (S3)
              ├── Evidently AI: DataDriftPreset
              └── Nếu drift > 30% features
                    └─► trigger retrain.yml
                          ├── SageMaker training job
                          ├── MLflow register new version
                          └── promote_rules.py
                                └── Nếu F1 ≥ 0.75 → promote Production
```

---

## Hướng dẫn chạy demo local

### Yêu cầu

- Python 3.11 + venv (đã setup)
- Docker Desktop

### Cách 1 — Django thuần (nhanh nhất)

```powershell
cd d:\StudyDocument\CongNgheMoi\project
.\venv\Scripts\Activate.ps1

cd sleep_portal
python manage.py migrate
python manage.py runserver
# Mở http://localhost:8000
```

### Cách 2 — Full stack với Docker Compose

```powershell
cd d:\StudyDocument\CongNgheMoi\project

# Build image (bao gồm mlruns/ và models/ bên trong)
docker build -f docker/Dockerfile -t sleep-portal:latest .

# Khởi động: Django + Redis + PostgreSQL + MQTT broker
docker-compose -f docker/docker-compose.local.yml up -d

# Theo dõi logs Django
docker-compose -f docker/docker-compose.local.yml logs -f web

# Mở http://localhost:8000
```

### Cách 3 — Chạy IoT Simulation

```powershell
# Bước 1: Khởi động MQTT broker
docker-compose -f docker/docker-compose.mqtt.yml up -d

# Bước 2: Cài dependencies nếu chưa có
pip install paho-mqtt mne

# Bước 3: Chạy subscriber (terminal riêng)
# Cần DATABASE_URL và S3_BUCKET trong .env (hoặc để demo offline)
python iot_simulation/subscriber.py

# Bước 4: Chạy simulator
# Tải file EDF từ: https://physionet.org/content/capslpdb/1.0.0/n1.edf
python iot_simulation/simulator.py --edf data/raw/n1.edf --patient-id n1 --delay 0.1
```

**Giải thích:**

1. `simulator.py` đọc `n1.edf` (bệnh nhân Healthy #1)
2. Lọc EEG 0.5–40 Hz, cắt từng epoch 30 giây
3. Publish lên MQTT topic `sleep/sensor/n1` (100ms giữa các epoch)
4. `subscriber.py` nhận, lưu S3 + ghi PostgreSQL
5. Data sẵn sàng cho feature extraction và API prediction

---

## Demo API thực tế (live trên AWS)

### Health check

```bash
curl http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/health/
# {"status": "ok"}
```

### Prediction với 24 features

```bash
curl -X POST \
  http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[
      0.0012, 0.35, 0.0008, 0.22, 0.0003, 0.09,
      0.0002, 0.05, 0.00005, 0.01, 1.85, 3.2, 4.1,
      0.000015, 0.000022, 0.000018, 0.8, 1.4,
      0.12, 2.8, 0.045, 0.0000003, 0.31, 2.1
    ]]
  }'
```

Response mẫu:

```json
{
  "predicted_class": "healthy",
  "predictions": ["healthy"],
  "prediction_count": 1,
  "class_counts": {"healthy": 1},
  "cached": false
}
```

### Model info

```bash
curl http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/model-info/
```

Response mẫu:

```json
{
  "model_name": "sleep-disorder-classifier",
  "model_stage": "None",
  "feature_count": 24,
  "feature_names": ["delta_power", "delta_rel", "..."],
  "supports_batch": true,
  "ready": true
}
```

### Batch prediction (nhiều epochs)

```bash
curl -X POST \
  http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": [
    [0.0012, 0.35, 0.0008, 0.22, 0.0003, 0.09, 0.0002, 0.05, 0.00005, 0.01,
     1.85, 3.2, 4.1, 0.000015, 0.000022, 0.000018, 0.8, 1.4, 0.12, 2.8,
     0.045, 0.0000003, 0.31, 2.1],
    [0.0025, 0.40, 0.0015, 0.30, 0.0001, 0.05, 0.00015, 0.03, 0.00003, 0.008,
     2.10, 2.8, 3.9, 0.000020, 0.000030, 0.000025, 1.2, 1.8, 0.08, 3.1,
     0.038, 0.0000004, 0.28, 2.4]
  ]}'
```

---

## Chạy tests

```powershell
cd d:\StudyDocument\CongNgheMoi\project\sleep_portal
python -m pytest ../tests/ -v
# Expected: 41 passed (test_api: 20, test_features: 9, test_inference: 12)
```

---

## Deploy lên AWS

### Bước 1: Build và push Docker image

```powershell
cd d:\StudyDocument\CongNgheMoi\project

# Build (bao gồm mlruns/ + models/ → embedded trong image)
docker build -f docker/Dockerfile -t sleep-portal:latest .

# Login ECR
aws ecr get-login-password --region ap-southeast-1 | `
  docker login --username AWS --password-stdin `
  651709558967.dkr.ecr.ap-southeast-1.amazonaws.com

# Push
docker tag sleep-portal:latest `
  651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
docker push `
  651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
```

### Bước 2: Redeploy ECS

```powershell
aws ecs update-service `
  --cluster sleep-portal-cluster `
  --service sleep-portal-service `
  --force-new-deployment `
  --region ap-southeast-1
```

### Bước 3: Theo dõi

```powershell
# Kiểm tra trạng thái (~2-3 phút sau khi push)
aws ecs describe-services `
  --cluster sleep-portal-cluster `
  --services sleep-portal-service `
  --region ap-southeast-1 `
  --query "services[0].{status:status,running:runningCount,desired:desiredCount}"

# Smoke test
curl http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/health/
```

---

## AWS Infrastructure

| Resource | Thông số |
|---|---|
| **ECS Cluster** | `sleep-portal-cluster` (ap-southeast-1) |
| **ECS Service** | `sleep-portal-service` (Fargate, 0.5 vCPU, 1 GB RAM) |
| **ECR** | `651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal` |
| **ALB** | `sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com` |
| **RDS** | PostgreSQL db.t3.micro (ap-southeast-1) |
| **CloudWatch** | Alarms: 5xx rate, latency p99, CPU > 80% |

### GitHub Secrets cần thiết

| Secret | Giá trị |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM deploy key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret |
| `ECR_REGISTRY` | `651709558967.dkr.ecr.ap-southeast-1.amazonaws.com` |
| `AWS_REGION` | `ap-southeast-1` |
| `DATABASE_URL` | `postgresql://user:pass@rds-endpoint:5432/sleep_portal` |
| `DJANGO_SECRET_KEY` | Secret key production |

---

## Training lại từ đầu (Kaggle)

1. Mở [Kaggle](https://www.kaggle.com/) → New Notebook → GPU P100
2. Upload `notebooks/kaggle_cap_training.ipynb`
3. Attach dataset: `shrutimurarka/cap-sleep-unbalanced-dataset`
4. Run All → download `sleep_model_export.zip` từ Output tab
5. Giải nén và copy vào project:

```powershell
$extract = "C:\temp\sleep_model_export"  # đường dẫn giải nén

# Thay mlruns mới
Remove-Item -Recurse -Force mlruns
Copy-Item -Recurse "$extract\mlruns" mlruns

# Copy model artifacts
Copy-Item "$extract\model.pkl"           models\
Copy-Item "$extract\model.ubj"           models\
Copy-Item "$extract\label_encoder.pkl"   models\
Copy-Item "$extract\feature_names.json"  models\
Copy-Item "$extract\metadata.json"       models\
```

6. Rebuild Docker và redeploy (xem phần Deploy)

---

## Terraform Infrastructure

```powershell
cd infrastructure

cp terraform.tfvars.example terraform.tfvars
# Chỉnh sửa: db_password, django_secret_key

terraform init
terraform plan
terraform apply
# Output: alb_dns_name, ecr_repository_url, rds_endpoint
```

---

## Environment Variables

Copy `.env.example` thành `.env`:

```env
# Django
DJANGO_SECRET_KEY=<strong-random-key>
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=<your-alb-dns>,localhost

# Database
DATABASE_URL=postgresql://<user>:<pass>@<rds-host>:5432/sleep_portal

# Redis
REDIS_URL=redis://<elasticache-host>:6379/0

# AWS
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_DEFAULT_REGION=ap-southeast-1
S3_BUCKET=sleep-mlops-651709

# MLflow (model embedded trong Docker — không cần remote server)
MLFLOW_TRACKING_URI=mlruns
MLFLOW_MODEL_NAME=sleep-disorder-classifier
MLFLOW_MODEL_STAGE=None

# MQTT (cho IoT simulation)
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC_PREFIX=sleep/sensor
```

---

## Tài liệu tham khảo

- [PhysioNet CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/)
- [MNE-Python Documentation](https://mne.tools/stable/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [AWS ECS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/userguide/what-is-fargate.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
# Sleep Disorder Detection  MLOps Pipeline

> **CAP Sleep Database  Django  AWS  MLflow  Evidently AI  Prefect**

Hệ thống MLOps end-to-end phát hiện rối loạn giấc ngủ từ tín hiệu EEG/PSG, mô phỏng luồng dữ liệu IoT thời gian thực, huấn luyện model trên AWS SageMaker, và triển khai qua Django REST API trên AWS ECS Fargate với tự động retraining khi phát hiện data drift.

---

##  Live Deployment

| Endpoint | URL | Status |
|---|---|---|
| **Web Dashboard** | `http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/` |  Running |
| **Health Check** | `GET /api/v1/health/` |  200 OK |
| **Predict (features)** | `POST /api/v1/predict/` |  Working |
| **Predict (EDF file)** | `POST /api/v1/predict-edf/` |  Working |
| **Model Info** | `GET /api/v1/model-info/` |  Working |
| **Patients** | `/patients/` |  Working |
| **Inference Studio** | `/predict/` |  Working |
| **Pipeline Status** | `/pipeline/` |  Working |

---

## Kiến trúc tổng thể

```

                      Layer 1  IoT Simulation                               
  Python Simulator  Mosquitto MQTT  AWS S3 (raw .npy) + RDS PostgreSQL    

                                    

                      Layer 2  Data & Training                              
  MNE-Python  Feature Builder (43 features)  DVC + S3  SageMaker        
                     MLflow Registry (XGBoost / LSTM / ResNet1D)           

                                    

                      Layer 3  Serving & Deployment                         
  GitHub Actions CI/CD  Docker + ECR  ECS Fargate  AWS ALB               
  Django Web App + Inference Service + ElastiCache Redis                     
  Dashboard  Patients  Inference Studio  Pipeline Status                  

                                    

                      Layer 4  Monitoring & Retraining                      
  CloudWatch  Evidently AI  Prefect  EventBridge  Auto-retrain loop     

```

---

## Dữ liệu: CAP Sleep Database

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn chính** | [PhysioNet CAP Sleep DB](https://physionet.org/content/capslpdb/1.0.0/) |
| **Nguồn thay thế** | Kaggle CAP Sleep Database (pre-processed CSV) |
| **Số bản ghi** | 108 polysomnographic recordings |
| **Kích thước** | 40.1 GB (PhysioNet) / 30+ GB (Kaggle) |
| **Tín hiệu** | EEG (F3/F4, C3/C4, O1/O2), EOG, EMG, airflow, SpO2, ECG |
| **Annotation** | Sleep stages (W, S1S4, REM) + CAP phase A subtypes (A1, A2, A3) |

### So sánh nguồn dữ liệu

| | PhysioNet (.edf) | Kaggle (.csv) |
|---|---|---|
| Format | Binary EDF + annotation .txt | Pre-segmented CSV |
| Channels | Multi-channel (EEG, EMG, ECG) | Single-channel EEG |
| Epoch | 30 giây (raw, cần xử lý) | 2 giây (1024 samples, sẵn sàng) |
| Labels | Sleep stages W/S1-S4/REM | CAP phases B/A1/A2/A3 |
| Kích thước/file | 100700 MB/file | 12 GB/file |
| Khuyến nghị | Training chính thức | Demo nhanh, prototype |

### Phân loại bệnh lý

| Prefix | Bệnh | Số lượng |
|---|---|---|
| `n1n16` | Healthy | 16 |
| `nfle1nfle40` | Nocturnal Frontal Lobe Epilepsy | 40 |
| `rbd1rbd22` | REM Behavior Disorder | 22 |
| `plm1plm10` | Periodic Leg Movements | 10 |
| `ins1ins9` | Insomnia | 9 |
| `narco1narco5` | Narcolepsy | 5 |
| `sdb1sdb4` | Sleep-Disordered Breathing | 4 |
| `brux1brux2` | Bruxism | 2 |

---

## Cấu trúc thư mục

```
sleep-disorder-mlops/

 README.md
 dvc.yaml                        # DVC pipeline stages
 params.yaml                     # Hyperparameters
 requirements.txt

 notebooks/
    kaggle_cap_training.ipynb   # Training notebook từ Kaggle CSV data

 data/                           # Managed bởi DVC
    raw/                        # File .edf gốc từ PhysioNet
    processed/                  # Epochs đã lọc (*.npz)
    features/                   # Feature parquet files
    kaggle/                     # CSV files từ Kaggle

 iot_simulation/                 # Layer 1  IoT
    simulator.py                # Đọc .edf  publish MQTT
    subscriber.py               # Nhận MQTT  lưu S3 + RDS
    mqtt_config.py
    docker-compose.mqtt.yml     # Mosquitto broker

 feature_engineering/            # Layer 2  Data Processing
    preprocess.py               # Bandpass filter, epoch 30s
    extract_features.py         # 43 spectral/statistical features
    annotation_parser.py        # Parse .txt/.st annotation
    build_dataset.py            # Tổng hợp feature + label

 training/                       # Layer 2  Model Training
    train.py                    # XGBoost training (SageMaker compatible)
    sagemaker_train.py          # Launch SageMaker job
    register_model.py           # Đăng ký lên MLflow Registry

 sleep_portal/                   # Layer 3  Django Web App
    manage.py
    sleep_portal/settings/
       base.py
       development.py
       production.py
    dashboard/
       views.py                # Home, Patients, Predict, Pipeline
       urls.py
       models.py               # Patient, EpochPrediction
       templates/dashboard/
           base.html           # Shared layout + navbar
           home.html           # Overview: KPIs, activity, quick links
           predict.html        # Inference Studio (4 tabs)
           pipeline.html       # Pipeline status + architecture
           patient_list.html   # Patient roster
           patient_detail.html # Epoch timeline + confidence
    api/
       views.py                # PredictView, PredictEDFView, ModelInfoView
       serializers.py
       urls.py
    inference/
        predictor.py            # Singleton model loader + predict()
        cache.py                # Redis cache wrapper

 docker/
    Dockerfile                  # Multi-stage Django image
    docker-compose.mqtt.yml     # Local MQTT stack

 monitoring/
    drift_detection.py          # Evidently AI drift report
    retrain_flow.py             # Prefect retraining flow
    promote_rules.py            # F1 threshold  promote

 tests/
    conftest.py              # Shared fixtures (sample_epoch, sample_features, django_client)
    test_features.py         # Feature extraction + annotation tests
    test_api.py              # REST API endpoint tests (health, predict, EDF, dashboard)
    test_inference.py        # Predictor unit tests (cache, batch, class counts)

 infrastructure/              # Terraform IaC
    main.tf                  # Provider + VPC + subnets + NAT gateway
    variables.tf             # All configurable variables
    security_groups.tf       # ALB / ECS / RDS security groups
    ecr.tf                   # ECR repository + lifecycle policy
    alb.tf                   # ALB + target group + HTTP listener
    ecs.tf                   # ECS cluster + task definition + service
    rds.tf                   # RDS PostgreSQL instance
    iam.tf                   # Task execution role + task role
    cloudwatch.tf            # 4 CloudWatch alarms
    outputs.tf               # ALB DNS, ECR URL, RDS endpoint
    terraform.tfvars.example # Variable values template

 .github/workflows/
     ci.yml                      # Test  Build  Push ECR  Deploy ECS
     monitoring.yml              # Weekly drift detection
     retrain.yml                 # Manual retrain trigger
```

---

## Hướng dẫn triển khai

### Bước 1  Cài đặt môi trường

```bash
git clone https://github.com/<your-username>/sleep-disorder-mlops.git
cd sleep-disorder-mlops

python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

### Bước 2  Tải dữ liệu

**Tùy chọn A  PhysioNet (chính thức):**
```bash
# Tải vài file để test
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.edf data/raw/
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.txt data/raw/

# Tải toàn bộ (40 GB)
aws s3 sync --no-sign-request s3://physionet-open/capslpdb/1.0.0/ data/raw/
```

**Tùy chọn B  Kaggle CSV (nhanh hơn):**
```bash
pip install kaggle
# Đặt ~/.kaggle/kaggle.json
kaggle datasets download -d <dataset-slug> -p data/kaggle/
unzip data/kaggle/*.zip -d data/kaggle/

# Mở notebook training
jupyter notebook notebooks/kaggle_cap_training.ipynb
```

### Bước 3  Feature Engineering (PhysioNet path)

```bash
python feature_engineering/preprocess.py \
    --input-dir data/raw --output-dir data/processed

python feature_engineering/extract_features.py \
    --input-dir data/processed --output-dir data/features

python feature_engineering/build_dataset.py \
    --features-dir data/features \
    --annotations-dir data/raw \
    --output data/features/dataset_labeled.parquet
```

### Bước 4  Training

**Local:**
```bash
python training/train.py \
    --data-dir data/features \
    --model-type xgboost \
    --model-dir models/

mlflow ui --backend-store-uri mlruns
#  http://localhost:5000
```

**AWS SageMaker:**
```bash
python training/sagemaker_train.py \
    --model-type xgboost \
    --data-uri s3://sleep-mlops-data/features/ \
    --instance-type ml.m5.xlarge
```

### Bước 5  Django Web App

```bash
cd sleep_portal
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py runserver
#  http://localhost:8000
```

**Các trang web:**

| URL | Trang | Mô tả |
|---|---|---|
| `/` | Overview | KPIs, activity feed, quick links |
| `/patients/` | Patients | Danh sách bệnh nhân + diagnosis |
| `/patients/<id>/` | Patient Detail | Epoch timeline, confidence |
| `/predict/` | Inference Studio | Single vector / Batch CSV / EDF upload / JSON API |
| `/pipeline/` | Pipeline Status | Model registry, CI/CD workflows, architecture |

### Bước 6  Provision AWS infrastructure (Terraform)

```bash
cd infrastructure

# Copy biến mẫu và điền giá trị thực
cp terraform.tfvars.example terraform.tfvars
# Chỉnh sửa terraform.tfvars: điền db_password, django_secret_key

# Hoặc dùng biến môi trường
export TF_VAR_db_password="your-db-password"
export TF_VAR_django_secret_key="your-django-secret-key"

terraform init
terraform plan
terraform apply
# Output: alb_dns_name, ecr_repository_url, rds_endpoint
```

> **Lưu ý**: Nếu đã có hạ tầng chạy sẵn, bỏ qua bước này và dùng `terraform import` để import vào state.

### Bước 7  Docker & ECR

```bash
docker build -f docker/Dockerfile -t sleep-portal:latest .

# Test local
docker run -p 8000:8000 \
    -e DJANGO_SETTINGS_MODULE=sleep_portal.settings.production \
    -e DATABASE_URL=postgresql://user:pass@host:5432/db \
    sleep-portal:latest

# Push ECR
aws ecr get-login-password --region ap-southeast-1 | \
    docker login --username AWS --password-stdin \
    651709558967.dkr.ecr.ap-southeast-1.amazonaws.com

docker tag sleep-portal:latest \
    651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
docker push \
    651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
```

### Bước 8  ECS Deploy

```bash
aws ecs update-service \
    --cluster sleep-portal-cluster \
    --service sleep-portal-service \
    --force-new-deployment \
    --region ap-southeast-1
```

### Bước 9  IoT Simulation

```bash
docker-compose -f docker/docker-compose.mqtt.yml up -d

python iot_simulation/subscriber.py

python iot_simulation/simulator.py \
    --edf data/raw/n1.edf \
    --patient-id n1
```

---

## API Reference

### POST /api/v1/predict/

Gửi một hoặc nhiều epochs (mỗi epoch = 43 features) và nhận dự đoán giai đoạn giấc ngủ.

```bash
curl -X POST http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,0.4,0.5,10.0,5.0,8.0,3.0,15.0,1.0,0.5,1.0,0.5,1.0,1.5,5.0,0.8,0.5,0.9,0.7,2.0,1.5]]}'
```

Response:
```json
{
  "predicted_class": "Wake",
  "predictions": ["Wake"],
  "prediction_count": 1,
  "class_counts": {"Wake": 1},
  "cached": false
}
```

### POST /api/v1/predict-edf/

Upload file EDF trực tiếp. Server chạy full MNE pipeline: bandpass filter  epoch 30s  extract 43 features  predict.

```bash
curl -X POST http://sleep-portal-alb-.../api/v1/predict-edf/ \
  -F "file=@data/raw/n1.edf"
```

Response:
```json
{
  "predictions": ["Wake", "S1", "S2", "S2", "REM"],
  "n_epochs": 5,
  "sfreq": 512,
  "duration_sec": 150,
  "channels_used": ["F3-A2", "C3-A2", "O1-A2"]
}
```

### GET /api/v1/model-info/

```bash
curl http://sleep-portal-alb-.../api/v1/model-info/
```

Response:
```json
{
  "model_name": "sleep-disorder-classifier",
  "model_stage": "None",
  "tracking_uri": "mlruns",
  "loaded": true
}
```

### Feature Schema (43 features)

| Index | Feature | Mô tả |
|---|---|---|
| 04 | `ch0_{band}_rel` | Relative band power channel 0 (delta/theta/alpha/beta/gamma) |
| 59 | `ch1_{band}_rel` | Relative band power channel 1 |
| 1014 | `ch2_{band}_rel` | Relative band power channel 2 |
| 1519 | `ch3_{band}_rel` | Relative band power channel 3 |
| 2024 | `ch4_{band}_rel` | Relative band power channel 4 |
| 2529 | `{band}_power_mean` | Mean absolute band power across channels |
| 3034 | `{band}_power_std` | Std of band power across channels |
| 35 | `spectral_entropy_mean` | Signal complexity measure |
| 36 | `peak_frequency_mean` | Dominant frequency (Hz) |
| 37 | `mean_frequency_mean` | Power-weighted mean frequency |
| 38 | `amplitude_mean` | Mean absolute amplitude |
| 39 | `amplitude_std` | Amplitude standard deviation |
| 40 | `rms` | Root mean square amplitude |
| 41 | `delta_beta_ratio` | Drowsiness marker |
| 42 | `theta_alpha_ratio` | Sleepiness marker |

**Label Mapping:**

| Code | Label | Mô tả |
|---|---|---|
| 0 | Wake | Thức |
| 1 | S1 | NREM stage 1 (ngủ nhẹ) |
| 2 | S2 | NREM stage 2 |
| 3 | S3 | NREM stage 3 (sóng chậm) |
| 4 | S4 | NREM stage 4 (sóng chậm sâu) |
| 5 | REM | Rapid Eye Movement |

---

## AWS Infrastructure

| Resource | Value |
|---|---|
| **ECS Cluster** | `sleep-portal-cluster` (ap-southeast-1) |
| **ECS Service** | `sleep-portal-service` (Fargate, 0.5 vCPU, 1 GB RAM) |
| **ECR** | `651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal` |
| **RDS PostgreSQL** | `sleep-portal-db.cjmguig029vh.ap-southeast-1.rds.amazonaws.com` (db.t3.micro) |
| **ALB** | `sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com` |
| **CloudWatch** | 3 alarms: 5xx rate, latency, CPU |

### GitHub Secrets cần thiết

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM deploy key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret |
| `ECR_REGISTRY` | `651709558967.dkr.ecr.ap-southeast-1.amazonaws.com` |
| `AWS_REGION` | `ap-southeast-1` |
| `ECR_REPOSITORY` | `sleep-portal` |
| `ECS_CLUSTER` | `sleep-portal-cluster` |
| `ECS_SERVICE` | `sleep-portal-service` |
| `DATABASE_URL` | `postgresql://sleepAdmin:<pw>@sleep-portal-db.cjmguig029vh.ap-southeast-1.rds.amazonaws.com:5432/sleep_portal` |

---

## CI/CD Pipeline

```
Push to main
     [1] pytest (unit tests)
     [2] docker build + push ECR
     [3] ECS force-new-deployment

Weekly (Monday 03:00 UTC)
     monitoring.yml
         drift_detection.py  Evidently AI report
             if drift  trigger retrain.yml
                 SageMaker training job
                     promote_rules.py (F1  0.75  Production)
```

---

## Cách hệ thống vận hành (end-to-end)

```
1. IoT Simulation
   simulator.py đọc file .edf từ CAP Sleep DB
    lọc bandpass 0.540 Hz, cắt epochs 30 giây
    publish lên MQTT topic: sleep/sensor/<patient_id>

2. Data Ingestion
   subscriber.py nhận MQTT message
    lưu raw epoch array (.npy) lên AWS S3
    ghi metadata vào PostgreSQL

3. Feature Engineering
   preprocess.py: bandpass, notch filter, artifact rejection
   extract_features.py: tính 43 features/epoch
   build_dataset.py: ghép features + sleep stage annotations
    dataset_labeled.parquet

4. Model Training
   train.py: XGBoost với class-weight balancing
    log metrics + model vào MLflow
   promote_rules.py: F1_weighted  0.75  Production

5. Serving
   Django load model từ MLflow Registry (singleton predictor)
   POST /api/v1/predict/  model.predict()  cache Redis (TTL 1h)
   Label mapping: 0Wake, 1S1, 2S2, 3S3, 4S4, 5REM

6. EDF Pipeline
   POST /api/v1/predict-edf/ nhận file .edf (max 500 MB)
    MNE: load + bandpass filter
    epoch 30s, extract 43 features/epoch
    predict() cho tất cả epochs  trả list predictions + metadata

7. Monitoring
   CloudWatch: CPU, memory, 5xx từ ECS/ALB
   Evidently AI: so sánh distribution mới vs baseline hàng tuần
    nếu drift  trigger retrain  promote nếu F1 pass
```

---

## Demo hệ thống

### Yêu cầu nhanh — kiểm tra toàn bộ hệ thống

```bash
ALB="http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com"

# 1. Health check
curl $ALB/api/v1/health/
# → {"status":"ok","timestamp":"..."}

# 2. Model metadata
curl $ALB/api/v1/model-info/
# → {"model_name":"sleep-disorder-classifier","model_stage":"None","loaded":...}

# 3. Single epoch prediction
curl -X POST $ALB/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features":[[0.12,0.05,0.31,0.22,0.08,0.14,0.07,0.29,0.18,0.11,0.09,0.06,0.24,0.15,0.10,0.11,0.08,0.19,0.14,0.09,0.13,0.06,0.21,0.16,0.08,10.2,5.4,8.1,3.3,15.6,1.1,0.6,1.2,0.5,1.1,1.4,5.3,0.7,0.6,0.88,0.72,2.1,1.4]]}'
# → {"predicted_class":"...","predictions":[...],"prediction_count":1,...}

# 4. Batch prediction (3 epochs)
curl -X POST $ALB/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features":[
    [0.12,0.05,0.31,0.22,0.08,0.14,0.07,0.29,0.18,0.11,0.09,0.06,0.24,0.15,0.10,0.11,0.08,0.19,0.14,0.09,0.13,0.06,0.21,0.16,0.08,10.2,5.4,8.1,3.3,15.6,1.1,0.6,1.2,0.5,1.1,1.4,5.3,0.7,0.6,0.88,0.72,2.1,1.4],
    [0.08,0.12,0.19,0.25,0.14,0.09,0.11,0.22,0.17,0.13,0.07,0.08,0.18,0.21,0.12,0.08,0.10,0.16,0.19,0.11,0.07,0.09,0.17,0.20,0.10,9.1,4.8,7.3,2.9,12.4,0.9,0.5,1.0,0.4,0.9,1.2,4.7,0.6,0.5,0.82,0.68,1.8,1.2],
    [0.15,0.08,0.27,0.19,0.11,0.16,0.09,0.25,0.14,0.13,0.11,0.07,0.22,0.18,0.09,0.13,0.09,0.20,0.12,0.10,0.14,0.07,0.23,0.17,0.09,11.3,6.1,9.2,3.8,17.1,1.2,0.7,1.4,0.6,1.3,1.6,5.9,0.9,0.7,0.91,0.75,2.4,1.6]
  ]}'
# → {"prediction_count":3,"predictions":["...","...","..."],"class_counts":{...}}

# 5. EDF file upload (nếu có file .edf)
curl -X POST $ALB/api/v1/predict-edf/ -F "file=@data/raw/n1.edf"
# → {"predictions":[...],"n_epochs":...,"sfreq":...}
```

### Chạy test suite

```bash
cd sleep_portal
pytest ../tests/ -v --tb=short
# test_features.py   — feature extraction + annotation parser
# test_api.py        — REST endpoints + dashboard pages
# test_inference.py  — predictor unit tests
```

### IoT simulation end-to-end

```bash
# Khởi động MQTT broker
docker-compose -f docker/docker-compose.mqtt.yml up -d

# Terminal 1: subscriber (nhận → lưu S3 + RDS)
python iot_simulation/subscriber.py

# Terminal 2: simulator (phát tín hiệu từ EDF)
python iot_simulation/simulator.py --edf data/raw/n1.edf --patient-id n1
# Kết quả: dữ liệu xuất hiện ở dashboard /patients/n1/
```

### Monitoring weekly drift check

```bash
# Manual trigger (thường chạy tự động mỗi thứ Hai 03:00 UTC)
python monitoring/drift_detection.py \
  --reference-data data/features/baseline.parquet \
  --current-data data/features/current_week.parquet \
  --output-report reports/

# Xem kết quả
cat reports/drift_summary_*.json
# → {"drift_detected":false,"drift_share":0.05,...}
```

---

## Chi phí AWS (ước tính)

| Dịch vụ | Tier | Chi phí/tháng |
|---|---|---|
| ECS Fargate | 0.5 vCPU  1 GB | ~$12 |
| RDS PostgreSQL | db.t3.micro, 20 GB | ~$15 |
| ALB | ~50 LCU | ~$20 |
| ECR | ~5 GB | ~$0.5 |
| CloudWatch | 3 alarms | ~$0.9 |
| **Tổng** | | **~$4855/tháng** |

---

## Tech Stack

| Công nghệ | Vai trò |
|---|---|
| Python 3.11 | Ngôn ngữ chính |
| MNE-Python | Xử lý tín hiệu EEG/PSG |
| XGBoost | ML classifier |
| MLflow | Experiment tracking + Registry |
| DVC | Data versioning |
| Django 4.2 + DRF | Web app + REST API |
| Bootstrap 5.3 | Frontend UI |
| Redis | Prediction cache |
| paho-mqtt + Mosquitto | IoT messaging |
| Evidently AI | Data/model drift detection |
| Prefect | Workflow orchestration |
| Docker | Containerization |
| AWS ECS Fargate | Serverless hosting |
| AWS RDS | PostgreSQL database |
| AWS ECR | Container registry |
| AWS ALB | Load balancer |
| AWS CloudWatch | Monitoring + alarms |
| GitHub Actions | CI/CD |

---

## Trạng thái hiện tại

###  Đã hoàn thành

- [x] Web app live trên ECS Fargate (HTTP)
- [x] Dashboard overview với KPIs và activity feed
- [x] Inference Studio: single vector, batch CSV, EDF upload, JSON editor
- [x] Pipeline Status page: model registry, CI/CD workflows, architecture diagram
- [x] Patients page: roster + diagnosis breakdown
- [x] Patient Detail: epoch timeline
- [x] REST API: `/predict/`, `/predict-edf/`, `/health/`, `/model-info/`
- [x] CI/CD pipeline: GitHub Actions → ECR → ECS
- [x] Monitoring workflows: `monitoring.yml` (weekly drift), `retrain.yml` (manual)
- [x] MLflow local tracking + model registry
- [x] Training notebook từ Kaggle CSV data (`notebooks/kaggle_cap_training.ipynb`)
- [x] CloudWatch: 4 alarms (5xx rate, P99 latency, ECS CPU, RDS CPU)
- [x] Evidently AI drift detection (`monitoring/drift_detection.py`) — full report + JSON summary
- [x] IaC Terraform (`infrastructure/`) — VPC, ALB, ECS Fargate, RDS, ECR, CloudWatch, IAM
- [x] Test suite mở rộng — `conftest.py`, `test_api.py`, `test_inference.py`

###  Chưa hoàn thành

- [ ] HTTPS → cần CloudFront distribution (tài khoản AWS chưa xác minh)
- [ ] Training đầy đủ → cần chạy full CAP dataset (hiện chỉ có vài file test)
- [ ] MLflow server riêng trên EC2 (hiện dùng `mlruns/` cục bộ trong Docker)
- [ ] AWS SageMaker training job (hiện train cục bộ)
- [ ] Prefect server/agent setup cho workflow orchestration
- [ ] AWS EventBridge auto-retrain trigger
