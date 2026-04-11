# Sleep Disorder Detection — MLOps Pipeline

> **CAP Sleep Database · Django · AWS · MLflow · Evidently AI · Prefect**

Hệ thống MLOps end-to-end phát hiện rối loạn giấc ngủ từ tín hiệu EEG/PSG, mô phỏng luồng dữ liệu IoT thời gian thực, huấn luyện model trên AWS SageMaker, và triển khai qua Django REST API trên AWS ECS Fargate với tự động retraining khi phát hiện data drift.

---

## Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IoT Simulation Layer (Tầng 1)                          │
│  Python Simulator → Mosquitto MQTT → AWS S3 (raw .edf) + RDS PostgreSQL    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Data & Training Layer (Tầng 2)                           │
│  MNE-Python → Feature Builder → DVC + S3 → SageMaker → MLflow Registry    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Serving & Deployment Layer (Tầng 3)                        │
│  GitHub Actions → Docker + ECR → ECS Fargate → ALB                         │
│  Django Web App + Inference Service + ElastiCache Redis + CloudFront CDN   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Monitoring & Retraining Layer (Tầng 4)                      │
│  CloudWatch → Evidently AI → Prefect → EventBridge → Auto-retrain loop     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dữ liệu: CAP Sleep Database

| Thuộc tính | Chi tiết |
|---|---|
| Nguồn | [PhysioNet CAP Sleep DB](https://physionet.org/content/capslpdb/1.0.0/) |
| Số bản ghi | 108 polysomnographic recordings |
| Kích thước | 40.1 GB (toàn bộ) |
| Tín hiệu | EEG (F3/F4, C3/C4, O1/O2), EOG, EMG (cằm + chân), airflow, SpO2, ECG |
| Annotation | Sleep stages (W, S1–S4, REM) + CAP phase A subtypes (A1, A2, A3) |
| Định dạng | `.edf` (waveform) + `.txt` / `.st` (annotation) |

### Phân loại bệnh lý (Label cho classification)

| Prefix file | Bệnh | Số lượng |
|---|---|---|
| `n1`–`n16` | Healthy (controls) | 16 |
| `nfle1`–`nfle40` | Nocturnal Frontal Lobe Epilepsy | 40 |
| `rbd1`–`rbd22` | REM Behavior Disorder | 22 |
| `plm1`–`plm10` | Periodic Leg Movements | 10 |
| `ins1`–`ins9` | Insomnia | 9 |
| `narco1`–`narco5` | Narcolepsy | 5 |
| `sdb1`–`sdb4` | Sleep-Disordered Breathing | 4 |
| `brux1`–`brux2` | Bruxism | 2 |

---

## Cấu trúc thư mục dự án

```
sleep-disorder-mlops/
│
├── README.md
├── .gitignore
├── .dvcignore
├── dvc.yaml                        # DVC pipeline stages
├── dvc.lock
├── requirements.txt
├── requirements-dev.txt
│
├── data/                           # Managed bởi DVC (không commit lên Git)
│   ├── raw/                        # File .edf gốc từ PhysioNet
│   ├── processed/                  # Epochs đã lọc
│   └── features/                   # Feature parquet files
│
├── iot_simulation/                 # Tầng 1 — IoT
│   ├── simulator.py                # Đọc .edf → publish MQTT
│   ├── subscriber.py               # Nhận MQTT → lưu S3 + RDS
│   ├── mqtt_config.py
│   └── docker-compose.mqtt.yml     # Mosquitto broker
│
├── feature_engineering/            # Tầng 2 — Data Processing
│   ├── preprocess.py               # Bandpass filter, epoch 30s
│   ├── extract_features.py         # Band power, entropy, HRV
│   ├── annotation_parser.py        # Parse .txt/.st annotation
│   └── build_dataset.py            # Tổng hợp feature + label → parquet
│
├── training/                       # Tầng 2 — Model Training
│   ├── train.py                    # Training script (SageMaker compatible)
│   ├── models/
│   │   ├── resnet1d.py
│   │   ├── lstm_model.py
│   │   └── xgboost_model.py
│   ├── sagemaker_train.py          # Launch SageMaker training job
│   └── register_model.py           # Đăng ký model lên MLflow Registry
│
├── sleep_portal/                   # Tầng 3 — Django Web App
│   ├── manage.py
│   ├── sleep_portal/               # Django project settings
│   │   ├── settings/
│   │   │   ├── base.py
│   │   │   ├── development.py
│   │   │   └── production.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── dashboard/                  # App hiển thị kết quả
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── templates/
│   │   └── static/
│   ├── api/                        # REST API endpoints
│   │   ├── views.py
│   │   ├── serializers.py
│   │   └── urls.py
│   └── inference/                  # Load model + predict
│       ├── model_loader.py         # Load từ MLflow Registry
│       ├── predictor.py
│       └── cache.py                # Redis cache
│
├── docker/
│   ├── Dockerfile                  # Django app image
│   ├── Dockerfile.simulator        # IoT simulator image
│   └── docker-compose.local.yml    # Local development
│
├── infrastructure/                 # AWS Infrastructure as Code
│   ├── terraform/                  # (optional) Terraform configs
│   ├── ecs-task-definition.json
│   ├── alb-config.json
│   └── cloudwatch-alarms.json
│
├── monitoring/                     # Tầng 4 — Monitoring
│   ├── drift_detection.py          # Evidently AI drift report
│   ├── retrain_flow.py             # Prefect retraining flow
│   └── promote_rules.py            # F1 threshold check → promote model
│
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   ├── test_inference.py
│   └── conftest.py
│
└── .github/
    └── workflows/
        ├── ci.yml                  # Test + Build + Push ECR
        └── retrain.yml             # Manual retrain trigger
```

---

## Hướng dẫn triển khai từng bước

### Bước 1 — Chuẩn bị môi trường

#### 1.1. Cài đặt dependencies

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Cài packages
pip install -r requirements.txt
```

#### 1.2. Tải dữ liệu CAP Sleep DB

```bash
# Tải qua AWS CLI (nhanh hơn, miễn phí)
aws s3 sync --no-sign-request s3://physionet-open/capslpdb/1.0.0/ data/raw/

# Hoặc wget (chỉ tải một vài file nhỏ để test trước)
wget -r -N -c -np https://physionet.org/files/capslpdb/1.0.0/n1.edf -P data/raw/
wget -r -N -c -np https://physionet.org/files/capslpdb/1.0.0/n1.txt -P data/raw/
```

#### 1.3. Khởi tạo Git + DVC

```bash
git init
git remote add origin https://github.com/<your-username>/sleep-disorder-mlops.git

# Khởi tạo DVC
dvc init

# Cấu hình DVC remote trỏ vào S3
dvc remote add -d myremote s3://sleep-mlops-data/dvc-store
dvc remote modify myremote region ap-southeast-1

# Track data với DVC
dvc add data/raw/
git add data/raw/.gitignore data/raw.dvc .dvc/config
git commit -m "chore: init DVC with S3 remote"
dvc push
```

#### 1.4. Cấu hình AWS credentials

```bash
aws configure
# AWS Access Key ID: <your-key>
# AWS Secret Access Key: <your-secret>
# Default region: ap-southeast-1
# Default output format: json
```

---

### Bước 2 — IoT Simulation (Tầng 1)

#### 2.1. Chạy Mosquitto MQTT Broker

```bash
docker-compose -f docker/docker-compose.mqtt.yml up -d
# Broker chạy tại localhost:1883
```

#### 2.2. Chạy subscriber (nhận → S3 + RDS)

```bash
# Cấu hình biến môi trường
cp .env.example .env
# Sửa .env: DATABASE_URL, S3_BUCKET, AWS_REGION

python iot_simulation/subscriber.py
```

#### 2.3. Chạy simulator (publish epochs)

```bash
# Publish epoch từ file n1.edf
python iot_simulation/simulator.py --edf data/raw/n1.edf --patient-id n1
```

Simulator sẽ đọc từng epoch 30 giây, serialize sang JSON/bytes, và publish lên topic `sleep/sensor/<patient_id>`.

---

### Bước 3 — Feature Engineering & Training (Tầng 2)

#### 3.1. Tiền xử lý tín hiệu EEG

```bash
python feature_engineering/preprocess.py \
    --input-dir data/raw/ \
    --output-dir data/processed/

# DVC track processed data
dvc add data/processed/
dvc push
```

Pipeline xử lý:
- Bandpass filter 0.5–40 Hz (butterworth order 4)
- Notch filter 50 Hz (loại bỏ nhiễu điện lưới)
- Cắt epoch 30 giây (theo annotation)
- Reject artifact (biên độ > 150 µV)

#### 3.2. Trích xuất đặc trưng

```bash
python feature_engineering/extract_features.py \
    --input-dir data/processed/ \
    --output-dir data/features/

dvc add data/features/
dvc push
```

Các đặc trưng được trích xuất:

| Nhóm | Đặc trưng |
|---|---|
| **Band power** | Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz) |
| **Spectral** | Spectral entropy, Peak frequency, Mean frequency |
| **HRV** | RMSSD, SDNN, LF/HF ratio (từ ECG) |
| **Nonlinear** | Sample entropy, Hurst exponent |
| **Relative power** | Tỉ lệ power từng band / total power |

#### 3.3. Huấn luyện trên AWS SageMaker

```bash
# Launch SageMaker training job
python training/sagemaker_train.py \
    --model-type xgboost \
    --data-uri s3://sleep-mlops-data/features/ \
    --instance-type ml.m5.xlarge \
    --mlflow-tracking-uri http://<ec2-mlflow-ip>:5000

# Xem kết quả tại MLflow UI
# http://<ec2-mlflow-ip>:5000
```

Model được huấn luyện:
- **XGBoost** — trên tabular features (baseline, nhanh, dễ interpret)
- **LSTM** — trên chuỗi epoch theo thời gian
- **ResNet1D** — trên raw EEG signal

#### 3.4. Đăng ký model lên MLflow Registry

```bash
python training/register_model.py \
    --run-id <mlflow-run-id> \
    --model-name sleep-disorder-classifier \
    --stage Staging
```

Sau khi validate trên test set (F1 ≥ 0.85), promote lên Production:

```bash
# Trong MLflow UI hoặc:
python -c "
import mlflow
client = mlflow.tracking.MlflowClient('http://<ec2>:5000')
client.transition_model_version_stage(
    name='sleep-disorder-classifier',
    version=1,
    stage='Production'
)
"
```

---

### Bước 4 — Django Web App (Tầng 3)

#### 4.1. Chạy local development

```bash
cd sleep_portal
pip install -r requirements.txt

# Migrate database
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Chạy server
python manage.py runserver
# → http://localhost:8000
```

#### 4.2. API Endpoints chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/v1/predict/` | Gửi epoch EEG → nhận prediction |
| `GET` | `/api/v1/patients/` | Danh sách bệnh nhân |
| `GET` | `/api/v1/patients/<id>/hypnogram/` | Hypnogram của bệnh nhân |
| `GET` | `/api/v1/models/` | Danh sách model trong Registry |
| `POST` | `/api/v1/retrain/` | Trigger retraining thủ công |

#### 4.3. Build Docker image

```bash
docker build -f docker/Dockerfile -t sleep-portal:latest .

# Test local
docker run -p 8000:8000 \
    -e DJANGO_SETTINGS_MODULE=sleep_portal.settings.production \
    -e DATABASE_URL=postgresql://... \
    -e REDIS_URL=redis://... \
    -e MLFLOW_TRACKING_URI=http://... \
    sleep-portal:latest
```

#### 4.4. Push lên AWS ECR

```bash
# Login ECR
aws ecr get-login-password --region ap-southeast-1 | \
    docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com

# Tag + push
docker tag sleep-portal:latest \
    <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
docker push \
    <account-id>.dkr.ecr.ap-southeast-1.amazonaws.com/sleep-portal:latest
```

#### 4.5. Deploy lên AWS ECS Fargate

```bash
# Update ECS service với image mới
aws ecs update-service \
    --cluster sleep-portal-cluster \
    --service sleep-portal-service \
    --force-new-deployment
```

---

### Bước 5 — CI/CD với GitHub Actions (Tầng 3)

Pipeline tự động kích hoạt mỗi khi push lên nhánh `main`:

```
Push to main
    │
    ├─→ [1] pytest (unit tests)
    ├─→ [2] Build Docker image
    ├─→ [3] Push image lên AWS ECR
    └─→ [4] Trigger ECS deploy (force-new-deployment)
```

Cấu hình GitHub Secrets cần thiết:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY
ECS_CLUSTER
ECS_SERVICE
MLFLOW_TRACKING_URI
DATABASE_URL
REDIS_URL
```

---

### Bước 6 — Monitoring & Auto-Retraining (Tầng 4)

#### 6.1. CloudWatch Metrics

ECS tự động gửi metrics về CloudWatch:
- CPU/Memory utilization
- Request count, latency (từ ALB)
- 5xx error rate

Tạo alarm khi error rate > 5%:

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name "SleepPortal-HighErrorRate" \
    --metric-name HTTPCode_Target_5XX_Count \
    --namespace AWS/ApplicationELB \
    --threshold 10 \
    --alarm-actions arn:aws:sns:...:notify-team
```

#### 6.2. Evidently AI — Data Drift Detection

Chạy hàng tuần so sánh phân phối dữ liệu mới với baseline:

```bash
python monitoring/drift_detection.py \
    --reference-data s3://sleep-mlops-data/baseline/features.parquet \
    --current-data s3://sleep-mlops-data/production/features_week_$(date +%Y%W).parquet \
    --output-report reports/drift_$(date +%Y%W).html
```

Nếu phát hiện drift (p-value < 0.05 hoặc F1 drop > 5%), Evidently gửi cảnh báo.

#### 6.3. Prefect — Retrain Orchestration

```bash
# Deploy Prefect flow
prefect deployment build monitoring/retrain_flow.py:retrain_pipeline \
    --name "weekly-retrain" \
    --schedule "0 2 * * 1"  # Mỗi thứ Hai 2:00 AM

prefect deployment apply retrain_pipeline-deployment.yaml
```

#### 6.4. AWS EventBridge — Auto-trigger khi drift

EventBridge rule kích hoạt Lambda → trigger Prefect flow khi nhận được CloudWatch alarm:

```json
{
  "source": ["aws.cloudwatch"],
  "detail-type": ["CloudWatch Alarm State Change"],
  "detail": {
    "alarmName": ["SleepPortal-DataDrift"],
    "state": { "value": ["ALARM"] }
  }
}
```

---

## Cài đặt nhanh (Quick Start)

```bash
# Clone repo
git clone https://github.com/<your-username>/sleep-disorder-mlops.git
cd sleep-disorder-mlops

# Cài dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Pull dữ liệu từ DVC
dvc pull

# Tải 2 file test nhỏ
mkdir -p data/raw
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.edf data/raw/
aws s3 cp --no-sign-request s3://physionet-open/capslpdb/1.0.0/n1.txt data/raw/

# Chạy pipeline cục bộ
python feature_engineering/preprocess.py --input-dir data/raw --output-dir data/processed
python feature_engineering/extract_features.py --input-dir data/processed --output-dir data/features

# Chạy Django local
cd sleep_portal
python manage.py migrate
python manage.py runserver
```

---

## Tech Stack

| Công nghệ | Vai trò |
|---|---|
| **Python 3.11** | Ngôn ngữ chính |
| **MNE-Python** | Xử lý tín hiệu EEG/PSG |
| **scikit-learn / XGBoost** | ML models |
| **PyTorch** | Deep learning (LSTM, ResNet1D) |
| **MLflow** | Experiment tracking + Model Registry |
| **DVC** | Data versioning |
| **Django 4.x + DRF** | Web app + REST API |
| **Redis** | Prediction cache |
| **paho-mqtt** | MQTT client |
| **Mosquitto** | MQTT broker |
| **Evidently AI** | Data/model drift detection |
| **Prefect** | Workflow orchestration |
| **Docker** | Containerization |
| **AWS S3** | Raw data + feature storage |
| **AWS RDS (PostgreSQL)** | Patient metadata |
| **AWS SageMaker** | Model training |
| **AWS ECR** | Container registry |
| **AWS ECS Fargate** | Serverless container hosting |
| **AWS ALB** | Load balancer |
| **AWS ElastiCache (Redis)** | Distributed cache |
| **AWS CloudFront** | CDN cho static files |
| **AWS CloudWatch** | Logs + metrics |
| **AWS EventBridge** | Event-driven automation |
| **GitHub Actions** | CI/CD pipeline |

---

## Yêu cầu AWS (Chi phí ước tính)

| Dịch vụ | Tier | Chi phí/tháng (ước tính) |
|---|---|---|
| EC2 (MLflow server) | t3.medium | ~$30 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| ECS Fargate | 0.25 vCPU, 0.5 GB | ~$10 |
| ElastiCache Redis | cache.t3.micro | ~$15 |
| S3 | 50 GB | ~$1.2 |
| SageMaker Training | ml.m5.xlarge × 2h | ~$0.46/job |
| CloudFront | 10 GB/month | ~$0.9 |
| **Tổng** | | **~$75/tháng** |

> Dùng AWS Free Tier và spot instances để giảm chi phí khi học tập.

---

## Tài liệu tham khảo

- [CAP Sleep Database — PhysioNet](https://physionet.org/content/capslpdb/1.0.0/)
- [MNE-Python Documentation](https://mne.tools/stable/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [Prefect Docs](https://docs.prefect.io/)
- [Django REST Framework](https://www.django-rest-framework.org/)
- [AWS ECS Fargate Developer Guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)

---

## License

Dữ liệu CAP Sleep DB được cấp phép theo [Open Data Commons Attribution License v1.0](https://physionet.org/content/capslpdb/view-license/1.0.0/).

Code trong repo này: MIT License.
