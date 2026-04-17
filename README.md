# Sleep Disorder Detection — MLOps Pipeline

> **CAP Sleep Database · LightGBM · Django · AWS ECS · MLflow · Evidently AI**

Hệ thống MLOps end-to-end phát hiện **7 loại rối loạn giấc ngủ** từ tín hiệu EEG, triển khai trên AWS ECS Fargate với Django REST API, MLflow Model Registry, Redis cache và CI/CD tự động qua GitHub Actions.

---

## Live Deployment

| Endpoint | URL |
|---|---|
| **Web Dashboard** | `http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/` |
| **Health Check** | `GET /api/v1/health/` |
| **Predict** | `POST /api/v1/predict/` |
| **Predict EDF** | `POST /api/v1/predict-edf/` |
| **Model Info** | `GET /api/v1/model-info/` |
| **Patients** | `/patients/` |
| **Inference Studio** | `/predict/` |
| **Pipeline Status** | `/pipeline/` |

---

## Kiến trúc tổng thể

```
┌─────────────────── Layer 1: IoT Simulation ────────────────────────┐
│  Python Simulator (EEG tổng hợp) → HTTP REST → Django /ingest/     │
│  5 bệnh nhân song song · 24 features/epoch · Lưu vào PostgreSQL    │
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
| **Algorithm** | LightGBM (best of 3: XGBoost / LightGBM / RandomForest) |
| **MLflow name** | `sleep-disorder-classifier` |
| **Validation F1** | 0.5929 |
| **Validation Accuracy** | 59.1% |
| **Features** | 24 (spectral + time-domain, single EEG channel) |
| **Window size** | 2 giây / 1024 mẫu (512 Hz) |
| **Training data** | CAP Sleep Database — Kaggle CSV (~140k epochs) |

### 7 Classes (Sleep Disorders)

| Label | Bệnh lý |
|---|---|
| `healthy` | Không rối loạn giấc ngủ |
| `nfle` | Nocturnal Frontal Lobe Epilepsy |
| `rbd` | REM Behavior Disorder |
| `plm` | Periodic Leg Movements |
| `insomnia` | Insomnia |
| `narcolepsy` | Narcolepsy |
| `sdb` | Sleep-Disordered Breathing |

### 24 Features

| # | Tên | Mô tả |
|---|---|---|
| 0 | `delta_power` | Công suất băng Delta (0.5–4 Hz) |
| 1 | `delta_rel` | Công suất tương đối Delta |
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

---

## Cấu trúc thư mục

```
sleep-disorder-mlops/
│
├── README.md
├── requirements.txt
├── requirements-prod.txt          # Production dependencies (Docker)
├── requirements-local.txt
│
├── models/                        # Model artifacts (embedded trong Docker)
│   ├── label_encoder.pkl          # Sklearn LabelEncoder cho 7 classes
│   ├── feature_names.json         # 24 tên features theo đúng thứ tự
│   └── metadata.json              # F1, accuracy, classes info
│   # model.pkl + model.ubj → gitignored (lớn, dùng DVC hoặc tải từ Kaggle)
│
├── mlruns/                        # MLflow local tracking (gitignored, embedded trong Docker)
│   └── 660434770358903185/        # CAP Sleep experiment
│       └── models/sleep-disorder-classifier/
│
├── notebooks/
│   └── kaggle_cap_training.ipynb  # Training notebook (Kaggle GPU)
│
├── iot_simulation/
│   ├── simulator.py               # Đọc .edf → publish MQTT
│   ├── subscriber.py              # Nhận MQTT → S3 + PostgreSQL
│   └── demo_local.py              # Demo không cần MQTT/S3 — gọi thẳng API
│
├── feature_engineering/
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── annotation_parser.py
│   └── build_dataset.py
│
├── training/
│   ├── train.py
│   ├── sagemaker_train.py
│   └── register_model.py
│
├── sleep_portal/                  # Django Web App
│   ├── manage.py
│   ├── sleep_portal/settings/
│   │   ├── base.py
│   │   ├── development.py         # SQLite, local mlruns, DEBUG=True
│   │   └── production.py          # ECS settings, env-vars driven
│   ├── dashboard/
│   │   ├── views.py
│   │   ├── models.py              # Patient, EpochPrediction
│   │   └── templates/dashboard/
│   │       ├── base.html
│   │       ├── home.html          # KPIs + model status
│   │       ├── predict.html       # Inference Studio (4 tabs)
│   │       ├── pipeline.html      # Model registry + CI/CD status
│   │       ├── patient_list.html
│   │       └── patient_detail.html
│   ├── api/
│   │   ├── views.py               # PredictView, PredictEDFView, Health, ModelInfo
│   │   ├── serializers.py
│   │   └── urls.py
│   └── inference/
│       └── predictor.py           # Singleton model loader + predict() + Redis cache
│
├── docker/
│   ├── Dockerfile                 # Multi-stage: builder → production
│   └── docker-compose.local.yml   # Django + Redis + PostgreSQL + MQTT
│
├── monitoring/
│   ├── drift_detection.py         # Evidently AI drift report
│   ├── retrain_flow.py            # Prefect retraining orchestration
│   └── promote_rules.py           # Auto-promote nếu F1 ≥ threshold
│
├── infrastructure/                # Terraform IaC
│   ├── main.tf                    # VPC + subnets + NAT gateway
│   ├── ecr.tf, ecs.tf, alb.tf, rds.tf, iam.tf, cloudwatch.tf
│   ├── variables.tf, outputs.tf
│   └── terraform.tfvars.example
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py                # 20 API tests
│   ├── test_features.py           # 9 feature extraction tests
│   └── test_inference.py          # 12 predictor unit tests
│
└── .github/workflows/
    ├── ci.yml                     # Push → Test → Build ECR → Deploy ECS
    ├── monitoring.yml             # Weekly drift detection
    └── retrain.yml                # Manual retrain trigger
```

---

## Hướng dẫn chạy hệ thống

### Bước 1 — Clone và cài đặt môi trường

```powershell
git clone https://github.com/wsunicorn/sleep-disorder-mlops.git
cd sleep-disorder-mlops

python -m venv venv
.\venv\Scripts\Activate.ps1       # Windows
# source venv/bin/activate        # Linux/Mac

pip install -r requirements.txt
```

### Bước 2 — Chạy Django local (phát triển)

```powershell
cd sleep_portal
python manage.py migrate
python manage.py runserver
# Mở http://localhost:8000
```

### Bước 3 — Chạy Multi-Patient IoT Demo (Đẩy dữ liệu lên AWS)

`multi_patient_demo.py` mô phỏng 5 thiết bị EEG IoT đồng thời, tạo EEG tổng hợp, trích xuất 24 features, gửi predict và lưu bệnh nhân qua `/api/v1/ingest/`.

```powershell
# Chạy 5 bệnh nhân song song đẩy lên AWS ALB (production)
python iot_simulation/multi_patient_demo.py --epochs 20 --delay 0.1

# Tùy chỉnh:
python iot_simulation/multi_patient_demo.py \
  --url http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com \
  --epochs 50 --batch-size 5 --delay 0.05 --workers 5

# Sau khi chạy, kiểm tra trên web:
# http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/patients/
```

### Bước 4 — Chạy Single-Patient IoT Demo

`demo_local.py` mô phỏng thiết bị EEG IoT: tạo dữ liệu EEG tổng hợp, tính 24 features, gửi batch đến API, hiển thị kết quả chẩn đoán.

```powershell
# Demo với server trực tiếp trên AWS (mặc định)
python iot_simulation/demo_local.py --disorder insomnia --epochs 30

# Các loại bệnh lý có thể demo:
python iot_simulation/demo_local.py --disorder healthy    --epochs 30
python iot_simulation/demo_local.py --disorder insomnia   --epochs 30
python iot_simulation/demo_local.py --disorder narcolepsy --epochs 30
python iot_simulation/demo_local.py --disorder nfle       --epochs 30
python iot_simulation/demo_local.py --disorder rbd        --epochs 30
python iot_simulation/demo_local.py --disorder plm        --epochs 30
python iot_simulation/demo_local.py --disorder sdb        --epochs 30

# Trỏ vào server local:
python iot_simulation/demo_local.py --disorder insomnia --url http://localhost:8000

# Tùy chỉnh đầy đủ:
python iot_simulation/demo_local.py \
  --patient-id "patient_001" \
  --disorder insomnia \
  --epochs 50 \
  --batch-size 10 \
  --delay 0.2 \
  --url http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com
```

**Output mẫu:**
```
✅ API health check: OK
📊 Model: sleep-disorder-classifier | Features: 24 | Ready: True
────────────────────────────────────────────────────────────
Epoch 001/20 | Patient: patient_insomnia | Predicted: nfle  | Cached: False
Epoch 002/20 | Patient: patient_insomnia | Predicted: nfle  | Cached: False
...
────────────────────────────────────────────────────────────
📈 KẾT QUẢ PHÂN TÍCH GIẤC NGỦ — Patient: patient_insomnia
   Tổng epochs phân tích: 20
   nfle : 16 epochs ( 80.0%) ████████████████
   plm  :  4 epochs ( 20.0%) ████
   → Chẩn đoán chính: NFLE
────────────────────────────────────────────────────────────
```

### Bước 4 — Chạy full stack với Docker Compose

```powershell
# Build image (mlruns/ và models/ được embed bên trong)
docker build -f docker/Dockerfile -t sleep-portal:latest .

# Khởi động: Django + Redis + PostgreSQL + MQTT
docker-compose -f docker/docker-compose.local.yml up -d

# Xem logs
docker-compose -f docker/docker-compose.local.yml logs -f web

# Mở http://localhost:8000
```

### Bước 5 — Chạy IoT Simulation đầy đủ (cần EDF file)

```powershell
# Tải file EDF mẫu từ PhysioNet
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.edf data/raw/

# Khởi động MQTT broker
docker-compose -f docker/docker-compose.local.yml up -d mqtt

# Terminal 1: Subscriber (nhận MQTT → S3 + DB)
python iot_simulation/subscriber.py

# Terminal 2: Simulator (đọc EDF → publish MQTT)
python iot_simulation/simulator.py \
  --edf data/raw/n1.edf \
  --patient-id n1 \
  --delay 0.1
```

---

## Hướng dẫn Deploy lên AWS

### Bước 1 — Provision Infrastructure (Terraform)

```powershell
cd infrastructure
cp terraform.tfvars.example terraform.tfvars
# Chỉnh sửa: db_password, django_secret_key

terraform init
terraform plan
terraform apply
# Output: alb_dns_name, ecr_repository_url, rds_endpoint
```

> Nếu đã có hạ tầng sẵn (như trường hợp hiện tại), bỏ qua bước này.

### Bước 2 — Build và push Docker image lên ECR

```powershell
# Login ECR
aws ecr get-login-password --region ap-southeast-1 | `
  docker login --username AWS --password-stdin `
  651709558967.dkr.ecr.ap-southeast-1.amazonaws.com

# Build (embed mlruns/ và models/ bên trong image)
docker build -f docker/Dockerfile -t sleep-portal:latest .

# Tag và push
docker tag sleep-portal:latest `
  651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
docker push `
  651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
```

### Bước 3 — Deploy ECS

```powershell
aws ecs update-service `
  --cluster sleep-portal-cluster `
  --service sleep-portal-service `
  --force-new-deployment `
  --region ap-southeast-1
```

### Bước 4 — Kiểm tra sau deploy

```powershell
# Chờ ~2-3 phút, kiểm tra status
aws ecs describe-services `
  --cluster sleep-portal-cluster `
  --services sleep-portal-service `
  --region ap-southeast-1 `
  --query "services[0].{status:status,running:runningCount,desired:desiredCount}"

# Smoke test
curl http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/health/
# {"status": "ok"}

# Kiểm tra model
curl http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/model-info/
# {"ready": true, "feature_count": 24, ...}
```

### CI/CD tự động

Mọi push lên nhánh `main` sẽ tự động:

```
git push → GitHub Actions ci.yml
  ├── Job 1: test (pytest 41 tests)
  └── Job 2: build-and-deploy (sau khi tests pass)
        ├── docker build -f docker/Dockerfile
        ├── docker push ECR
        └── aws ecs update-service --force-new-deployment
```

---

## Trải nghiệm Web Dashboard

Truy cập `http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com`

| Trang | URL | Mô tả |
|---|---|---|
| **Home / Overview** | `/` | KPIs tổng quan, model status chip (24 features, Ready), activity feed |
| **Inference Studio** | `/predict/` | 4 tabs: Single vector · Batch CSV · EDF upload · JSON API |
| **Patients** | `/patients/` | Danh sách bệnh nhân + diagnosis đã ghi nhận |
| **Patient Detail** | `/patients/<id>/` | Timeline epochs + confidence chart |
| **Pipeline Status** | `/pipeline/` | Model registry versions, CI/CD workflow status, architecture |
| **API Health** | `/api/v1/health/` | JSON: `{"status": "ok"}` |
| **API Model Info** | `/api/v1/model-info/` | JSON: model name, stage, 24 features, ready status |
| **API Ingest** | `POST /api/v1/ingest/` | Nhận dữ liệu IoT: lưu bệnh nhân + epoch predictions |

### Inference Studio — 4 tabs

1. **Single Vector**: Nhập 24 số float (cách nhau bởi dấu phẩy), nhấn Predict
2. **Batch CSV**: Upload file CSV — mỗi hàng là 1 epoch (24 cột), nhận predictions cho toàn bộ
3. **EDF Upload**: Upload file EDF trực tiếp — server tự trích xuất features và predict
4. **JSON API**: Xem ví dụ curl command, copy và chạy trực tiếp

---

## API Reference

### POST /api/v1/ingest/

Nhận dữ liệu từ thiết bị IoT: upsert bệnh nhân + lưu epoch predictions.

```bash
curl -X POST \
  http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/ingest/ \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-001",
    "diagnosis": "insomnia",
    "age": 42,
    "gender": "F",
    "epochs": [
      {"epoch_index": 0, "predicted_class": "insomnia", "confidence": 0.87, "timestamp": "2025-01-01T00:00:00Z"}
    ]
  }'
```

---

### POST /api/v1/predict/

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

Response:
```json
{
  "predicted_class": "healthy",
  "predictions": ["healthy"],
  "prediction_count": 1,
  "class_counts": {"healthy": 1},
  "cached": false
}
```

### Batch prediction

```bash
curl -X POST \
  http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": [
    [0.0012,0.35,0.0008,0.22,0.0003,0.09,0.0002,0.05,0.00005,0.01,1.85,3.2,4.1,0.000015,0.000022,0.000018,0.8,1.4,0.12,2.8,0.045,0.0000003,0.31,2.1],
    [0.0025,0.40,0.0015,0.30,0.0001,0.05,0.00015,0.03,0.00003,0.008,2.10,2.8,3.9,0.000020,0.000030,0.000025,1.2,1.8,0.08,3.1,0.038,0.0000004,0.28,2.4]
  ]}'
```

### GET /api/v1/model-info/

```bash
curl http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/model-info/
```

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

---

## Chạy tests

```powershell
cd sleep_portal
python -m pytest ../tests/ -v
# Expected: 41 passed (test_api: 20, test_features: 9, test_inference: 12)
```

---

## Training lại từ đầu (Kaggle)

1. Mở [Kaggle](https://www.kaggle.com/) → New Notebook → GPU T4 x2
2. Upload `notebooks/kaggle_cap_training.ipynb`
3. Attach dataset: `shrutimurarka/cap-sleep-unbalanced-dataset`
4. Run All → download `sleep_model_export.zip` từ Output tab
5. Giải nén và update project:

```powershell
$extract = "C:\temp\sleep_model_export"

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

6. Fix mlruns paths (nếu Kaggle hardcode `/kaggle/working/`):

```powershell
Get-ChildItem -Recurse mlruns -Filter "meta.yaml" | ForEach-Object {
    (Get-Content $_.FullName) -replace '/kaggle/working/mlruns', '/app/mlruns' |
    Set-Content $_.FullName
}
```

7. Rebuild Docker và redeploy (xem phần Deploy)

---

## AWS Infrastructure

| Resource | Giá trị |
|---|---|
| **ECS Cluster** | `sleep-portal-cluster` (ap-southeast-1) |
| **ECS Service** | `sleep-portal-service` (Fargate, 0.5 vCPU, 1 GB RAM) |
| **ECR** | `651709558967.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal` |
| **ALB** | `sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com` |
| **RDS** | PostgreSQL db.t3.micro (ap-southeast-1) |
| **CloudWatch** | Alarms: 5xx rate, latency p99, CPU > 80% |

### GitHub Secrets cần thiết (cho CI/CD)

| Secret | Giá trị |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM deploy key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret |
| `ECR_REGISTRY` | `651709558967.dkr.ecr.ap-southeast-1.amazonaws.com` |
| `AWS_REGION` | `ap-southeast-1` |
| `ECR_REPOSITORY` | `sleep-portal` |
| `ECS_CLUSTER` | `sleep-portal-cluster` |
| `ECS_SERVICE` | `sleep-portal-service` |
| `DATABASE_URL` | `postgresql://user:pass@rds-endpoint:5432/sleep_portal` |
| `DJANGO_SECRET_KEY` | Django production secret key |

---

## Environment Variables

Copy `.env.example` thành `.env` và điền giá trị:

```env
DJANGO_SECRET_KEY=<strong-random-key>
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=<your-alb-dns>,localhost

DATABASE_URL=postgresql://<user>:<pass>@<rds-host>:5432/sleep_portal
REDIS_URL=redis://<elasticache-host>:6379/0

AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_DEFAULT_REGION=ap-southeast-1
S3_BUCKET=sleep-mlops-651709

# MLflow (model embedded trong Docker — không cần remote server)
MLFLOW_TRACKING_URI=mlruns
MLFLOW_MODEL_NAME=sleep-disorder-classifier
MLFLOW_MODEL_STAGE=None
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
