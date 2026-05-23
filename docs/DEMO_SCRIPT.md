# Kịch bản demo project Sleep Disorder Detection MLOps

Tài liệu này dùng để chuẩn bị lời nói và thao tác khi demo project. Mục tiêu là giúp người trình bày không chỉ bấm đúng màn hình, mà còn giải thích được ý nghĩa của từng thành phần trong hệ thống.

## 1. Thông tin nhanh cần nhớ

Tên đề tài:

- Phân tích dữ liệu cảm biến giấc ngủ để phát hiện rối loạn.

Giảng viên hướng dẫn:

- TS. Bùi Thanh Hùng.

Sinh viên tham gia:

- Nguyễn Ngọc Lân.
- Đoàn Vũ Thiên Ban.

Production URL:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
```

MLflow URL:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com:5000
```

Model production:

- Tên model: `sleep-disorder-classifier`.
- Stage phục vụ: `Production`.
- Schema đầu vào: 24 đặc trưng EEG.
- Số lớp dự đoán: 7 lớp gồm `healthy`, `insomnia`, `narcolepsy`, `nfle`, `plm`, `rbd`, `sdb`.

Câu tóm tắt một câu:

> Project này biến notebook huấn luyện từ dữ liệu CAP Sleep thành một hệ thống MLOps production: có web app, REST API, MLflow Model Registry, feature store trên S3, monitoring drift, retraining và CI/CD deploy lên AWS ECS Fargate.

## 2. Mục tiêu demo

Khi demo xong, người xem cần hiểu được 5 ý chính:

1. Dữ liệu EEG giấc ngủ được chuyển thành 24 đặc trưng theo notebook chuẩn `notebooks/kaggle_cap_training.ipynb`.
2. Model không chỉ nằm trong notebook, mà đã được đóng gói để phục vụ qua Django REST API.
3. Web dashboard có thể dự đoán, xem hồ sơ bệnh nhân, xem timeline epoch và quản lý dữ liệu demo.
4. Hệ thống có vòng lặp MLOps: ingest dữ liệu mới -> lưu feature store -> monitoring drift -> retrain -> MLflow Registry -> redeploy.
5. CI/CD trên GitHub Actions có thể build Docker image, push ECR, deploy ECS, migrate DB và smoke test production.

## 3. Kịch bản theo thời lượng

Nếu chỉ có 5-7 phút:

1. Mở dashboard tổng quan.
2. Mở trang MLOps Pipeline.
3. Mở Studio Suy luận và chạy một dự đoán.
4. Mở Hồ sơ bệnh nhân và một bệnh nhân mixed.
5. Mở MLflow Model Registry.

Nếu có 10-15 phút:

1. Làm toàn bộ phần 5-7 phút.
2. Chạy script IoT demo để tạo dữ liệu mới.
3. Quay lại dashboard chứng minh bệnh nhân/epoch xuất hiện.
4. Chỉ GitHub Actions CI/CD và Monitoring.

Nếu có 20-25 phút:

1. Làm toàn bộ phần 10-15 phút.
2. Chạy hoặc mở workflow Monitoring.
3. Chạy hoặc mở workflow Retrain.
4. Giải thích điều kiện promote model và redeploy.

## 4. Chuẩn bị trước khi demo

### 4.1. Mở sẵn các tab trình duyệt

Nên mở sẵn các tab sau để demo mượt:

1. Web app:

   ```text
   http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
   ```

2. Trang hồ sơ bệnh nhân:

   ```text
   http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/patients/
   ```

3. Trang suy luận:

   ```text
   http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/predict/
   ```

4. Trang MLOps Pipeline:

   ```text
   http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/pipeline/
   ```

5. MLflow:

   ```text
   http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com:5000
   ```

6. GitHub Actions của repository.

### 4.2. Kiểm tra nhanh production trước khi demo

Chạy:

```powershell
Invoke-RestMethod -Uri "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/api/v1/health/" -TimeoutSec 20
```

Kỳ vọng:

```text
status: ok
```

Chạy:

```powershell
Invoke-RestMethod -Uri "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/api/v1/model-info/" -TimeoutSec 30
```

Kỳ vọng cần nói được:

- `ready=True`: model đã load được.
- `model_stage=Production`: API đang dùng model production.
- `feature_count=24`: đúng schema notebook.
- `tracking_uri`: trỏ về MLflow server production.

### 4.3. Nếu AWS service đã bị tắt để tiết kiệm chi phí

Chạy lại MLflow server:

```powershell
gh workflow run mlflow.yml --ref main -f reason="Demo MLflow production"
```

Chạy lại web app:

```powershell
gh workflow run ci.yml --ref main -f reason="Demo web app production"
```

Kiểm tra trạng thái:

```powershell
gh run list --workflow ci.yml --limit 3
gh run list --workflow mlflow.yml --limit 3
```

Khi cả hai workflow xanh, mở lại health/model-info như mục 4.2.

## 5. Cấu trúc lời mở đầu

Lời thoại gợi ý:

> Em xin trình bày project phát hiện rối loạn giấc ngủ từ dữ liệu cảm biến, trọng tâm là tín hiệu EEG trong CAP Sleep Database. Điểm chính của project không chỉ là train model trong notebook, mà là đưa model đó vào một hệ thống MLOps hoàn chỉnh: có API phục vụ dự đoán, dashboard xem bệnh nhân, MLflow quản lý version model, S3 lưu feature/artifact, GitHub Actions tự động deploy lên AWS ECS và workflow monitoring/retraining khi có dữ liệu mới.

Nếu cần nói ngắn hơn:

> Notebook là nơi chuẩn hóa dữ liệu và huấn luyện. Còn project này là phần production hóa notebook đó thành hệ thống MLOps chạy được trên AWS.

## 6. Demo 1 - Dashboard tổng quan

Mở:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
```

Thao tác:

1. Chỉ vào các KPI: số bệnh nhân, số epoch, độ tin cậy trung bình.
2. Chỉ vào phân bố chẩn đoán.
3. Chỉ phần dự đoán gần đây.

Lời thoại gợi ý:

> Đây là dashboard tổng quan của hệ thống production. Dữ liệu ở đây không phải số tĩnh trong HTML, mà được đọc từ database PostgreSQL sau khi API `/api/v1/ingest/` nhận dữ liệu mô phỏng IoT hoặc dữ liệu dự đoán. Mỗi bệnh nhân có nhiều epoch, tức nhiều cửa sổ EEG 2 giây, và dashboard gom lại thành thống kê tổng quan.

Điểm cần nhấn:

- Đây là dữ liệu đã ingest vào hệ thống.
- Một bệnh nhân có nhiều epoch.
- Mỗi epoch tương ứng một đoạn tín hiệu EEG đã được trích 24 feature.

## 7. Demo 2 - Trang MLOps Pipeline

Mở:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/pipeline/
```

Thao tác:

1. Chỉ phần Serving model:
   - Model name.
   - Stage `Production`.
   - Feature schema 24 đặc trưng.
   - Tracking URI MLflow.
2. Chỉ phần GitHub Actions:
   - CI/CD.
   - MLflow.
   - Monitoring.
   - Retrain.
3. Chỉ sơ đồ 6 lớp:
   - Data Source & IoT Simulation.
   - Feature Engineering & Training.
   - MLflow Model Management.
   - Serving & Web Application.
   - CI/CD Deployment.
   - Monitoring & Retraining.

Lời thoại gợi ý:

> Trang này là bản đồ vận hành của project. Lớp đầu tiên là dữ liệu CAP Sleep và giả lập IoT. Sau đó dữ liệu được xử lý thành 24 đặc trưng, train model, log vào MLflow. Khi model được promote lên Production, web app trên ECS Fargate sẽ load model từ MLflow Registry, fallback artifact S3 nếu cần. Dữ liệu mới từ API được ghi vào feature store để Evidently kiểm tra drift, rồi workflow retrain có thể chạy lại và kích hoạt CI/CD redeploy.

Nếu bị hỏi “MLOps nằm ở đâu?”:

> MLOps nằm ở các phần tự động hóa và quản lý vòng đời model: MLflow tracking/registry, feature store S3, drift monitoring, retrain workflow, promotion rule và CI/CD redeploy. Nếu chỉ có notebook train model thì chưa gọi là MLOps hoàn chỉnh.

## 8. Demo 3 - REST API health và model-info

Mở trực tiếp trên trình duyệt:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/api/v1/health/
```

Lời thoại:

> Endpoint này dùng cho ALB và CI/CD smoke test để biết service còn sống.

Mở:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/api/v1/model-info/
```

Lời thoại:

> Endpoint này cho biết API đang dùng model nào, có sẵn sàng không, model được load từ MLflow hay artifact S3, và số feature đầu vào có đúng 24 không. Đây là cách kiểm tra production model trước khi demo dự đoán.

Các điểm cần đọc ra nếu thấy JSON:

- `ready`: trạng thái model.
- `model_name`: `sleep-disorder-classifier`.
- `model_stage`: `Production`.
- `feature_count`: `24`.
- `model_type`: thường là `PyFuncModel`.

## 9. Demo 4 - Studio Suy luận bằng vector đặc trưng

Mở:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/predict/
```

### 9.1. Vector đơn lẻ

Thao tác:

1. Chọn tab vector đơn lẻ.
2. Bấm tải dữ liệu mẫu nếu giao diện có nút mẫu.
3. Bấm dự đoán.
4. Đọc kết quả: nhãn dự đoán và trạng thái model.

Lời thoại:

> Đây là luồng inference nhanh nhất. Người dùng gửi một vector gồm 24 đặc trưng EEG. API kiểm tra schema, chuyển thành DataFrame đúng tên feature và gọi model production. Cách này phù hợp khi dữ liệu đã được trích feature từ thiết bị hoặc pipeline bên ngoài.

Nếu muốn gọi bằng PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/api/v1/predict/" `
  -ContentType "application/json" `
  -Body (Get-Content -Raw demo_web_iot\predict_single_healthy.json)
```

### 9.2. CSV theo lô

Thao tác:

1. Chọn tab CSV.
2. Upload file:

   ```text
   demo_web_iot/predict_batch.csv
   ```

3. Bấm dự đoán.
4. Chỉ bảng kết quả từng dòng.

Lời thoại:

> CSV batch dùng để demo nhiều epoch hoặc nhiều mẫu cùng lúc. Về bản chất, mỗi dòng vẫn là một vector 24 feature. API hỗ trợ batch nên có thể dự đoán nhiều epoch trong một request.

### 9.3. EDF upload

Thao tác:

1. Chọn tab EDF.
2. Upload một file `.edf` nhỏ hoặc file CAP local nếu có.
3. Bấm xử lý và dự đoán.

Lời thoại:

> Với EDF, server đọc tín hiệu thô bằng PyEDFlib, chọn kênh EEG, lọc băng thông 0.5-40 Hz, chia epoch 2 giây, trích 24 feature rồi đưa vào model. Vì request web đi qua ALB, bản demo chỉ xử lý một số epoch đầu để phản hồi ổn định; với full-night EDF nên dùng batch pipeline hoặc simulator IoT.

Điểm cần nhấn:

- Web demo giới hạn epoch để tránh timeout.
- Full EDF dài nên đi qua pipeline nền/S3/job queue.
- Đây là quyết định production hợp lý, không để người dùng chờ một request quá dài.

## 10. Demo 5 - Giả lập IoT và ingest dữ liệu mới

Có hai mức demo.

### 10.1. Demo nhanh

Chạy:

```powershell
.\demo_web_iot\run_demo_rest.ps1 `
  -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
```

Script này làm gì:

1. Gọi `/api/v1/health/`.
2. Gọi `/api/v1/model-info/`.
3. Gọi `/api/v1/predict/` với một mẫu.
4. Gọi batch predict.
5. Gửi dữ liệu bệnh nhân demo vào `/api/v1/ingest/`.

Lời thoại:

> Script này mô phỏng một thiết bị hoặc service trung gian gửi dữ liệu epoch về API. Sau khi ingest, dữ liệu xuất hiện trong dashboard bệnh nhân và đồng thời được ghi vào feature store phục vụ monitoring/retrain.

### 10.2. Demo đẹp hơn với nhiều bệnh nhân

Chạy:

```powershell
.\demo_web_iot\run_rich_demo.ps1 `
  -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
```

Kết quả mặc định:

- 24 bệnh nhân.
- 1152 epoch.
- Đủ 7 nhóm bệnh.
- Có bệnh nhân mixed để biểu đồ timeline có nhiều nhãn.

Muốn tạo bộ lớn hơn:

```powershell
.\demo_web_iot\run_rich_demo.ps1 `
  -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com `
  -Regenerate `
  -PatientsPerClass 5 `
  -MixedPatients 5 `
  -EpochsPerPatient 96
```

Lời thoại:

> Bộ rich demo giúp mô phỏng tình huống hệ thống nhận dữ liệu từ nhiều bệnh nhân khác nhau. Các bệnh nhân healthy, insomnia, narcolepsy, nfle, plm, rbd, sdb giúp dashboard có phân bố lớp đầy đủ. Các ca mixed dùng để trình bày timeline nhiều nhãn theo từng epoch.

### 10.3. Demo từ EDF thật bằng simulator

Chạy:

```powershell
python iot_simulation/simulator.py `
  --edf data\raw\your_recording.edf `
  --api-base http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com `
  --patient-id demo-edf-001 `
  --diagnosis insomnia `
  --max-epochs 32
```

Lời thoại:

> Simulator đọc EDF, trích feature theo cùng schema notebook, gọi API predict, sau đó gửi kết quả vào ingest endpoint. Đây là cách mô phỏng thiết bị IoT hoặc gateway gửi dữ liệu giấc ngủ về hệ thống.

## 11. Demo 6 - Hồ sơ bệnh nhân

Mở:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/patients/
```

Thao tác:

1. Tìm `demo-rich-mixed-01`.
2. Chỉ các cột:
   - Mã bệnh nhân.
   - Nguồn.
   - Chẩn đoán.
   - Epoch.
   - Độ tin cậy trung bình.
   - Tuổi/giới tính.
3. Mở chi tiết một bệnh nhân mixed.
4. Chỉ biểu đồ timeline epoch.
5. Chỉ biểu đồ phân bố lớp.

Lời thoại:

> Trang hồ sơ là nơi chứng minh dữ liệu mới đã được ghi vào database. Mỗi bệnh nhân có ID gốc để API upsert ổn định, đồng thời UI tạo mã hiển thị chuẩn hóa như `DEMO-RICH-INS-01` hoặc `CAP-PT-001` để danh sách dễ đọc khi demo/báo cáo.

Nếu mở bệnh nhân mixed:

> Với bệnh nhân mixed, các epoch có thể thuộc nhiều nhãn khác nhau. Timeline giúp nhìn sự thay đổi dự đoán theo thời gian thay vì chỉ xem một nhãn tổng.

### 11.1. Demo xóa bệnh nhân

Thao tác an toàn:

1. Chỉ nút `Xóa` trên danh sách.
2. Bấm thử với bệnh nhân test nếu có.
3. Đọc modal xác nhận.
4. Không xóa các bệnh nhân demo chính nếu chưa cần.

Lời thoại:

> Chức năng xóa dùng POST và CSRF token, không cho xóa bằng GET. Khi xóa bệnh nhân, các epoch liên quan được xóa cascade trong database. Đây là thao tác quản trị dữ liệu demo, không ảnh hưởng đến model artifact hay MLflow run.

Nếu cần tạo bệnh nhân tạm để demo xóa:

```powershell
$payload = @{
  patient_id = "demo-delete-temp-001"
  diagnosis = "healthy"
  age = 30
  gender = "F"
  epochs = @(
    @{
      epoch_index = 0
      predicted_class = "healthy"
      confidence = 0.95
      timestamp = "2026-05-23T09:00:00Z"
      features = @(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3)
      label = "healthy"
    }
  )
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com/api/v1/ingest/" `
  -ContentType "application/json" `
  -Body $payload
```

Sau đó mở `/patients/`, tìm `demo-delete-temp-001`, bấm xóa và xác nhận.

## 12. Demo 7 - MLflow Tracking và Model Registry

Mở:

```text
http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com:5000
```

Thao tác:

1. Vào experiment `sleep-disorder-kaggle`.
2. Chỉ các run train/retrain.
3. Chỉ metric:
   - `val_f1_weighted`.
   - `val_accuracy`.
   - các metric khác nếu có.
4. Mở artifact của run.
5. Mở Model Registry.
6. Chỉ model `sleep-disorder-classifier` stage `Production`.

Lời thoại:

> MLflow là trung tâm quản lý vòng đời model. Mỗi lần train hoặc retrain sẽ tạo một run, log tham số, metric và artifact. Model tốt nhất được đăng ký vào Model Registry. Web app không hard-code model trong code, mà ưu tiên load model từ Registry stage `Production`.

Nếu bị hỏi “vì sao cần MLflow server riêng?”:

> Vì production cần một nơi tập trung để lưu run, metric, artifact và quyết định version model nào đang phục vụ. Nếu chỉ lưu local `mlruns`, CI/CD và ECS task sẽ không nhìn thấy cùng một registry.

## 13. Demo 8 - GitHub Actions CI/CD

Mở GitHub Actions workflow:

```text
CI/CD — Build, Test, Deploy
```

Thao tác:

1. Mở run mới nhất đang xanh.
2. Chỉ job `test`.
3. Chỉ job `build-and-push`.
4. Chỉ job `deploy`.
5. Trong deploy, chỉ các bước:
   - Ensure public ALB.
   - Ensure ECS task role.
   - Register ECS task definition.
   - Deploy to ECS.
   - Run Django migrate.
   - Smoke test deployed service.

Lời thoại:

> Khi push lên main, GitHub Actions tự chạy test, build Docker image, push lên Amazon ECR, cập nhật ECS task definition, deploy lên ECS Fargate, chạy migrate và smoke test qua ALB. Đây là CI/CD thật chứ không phải deploy thủ công.

Nếu bị hỏi “tắt ALB/EC2 để tiết kiệm chi phí thì sao?”:

> Project có script `scripts/ensure_aws_alb.sh` để kiểm tra hoặc tạo lại ALB/target group/listener khi deploy. Vì vậy khi bật lại bằng workflow, hệ thống có thể tự khôi phục endpoint public ở mức cần thiết cho demo.

## 14. Demo 9 - Monitoring drift

Mở GitHub Actions workflow:

```text
Monitoring - Drift Check
```

Chạy thủ công nếu muốn:

```powershell
gh workflow run monitoring.yml --ref main
```

Hoặc nhập rõ input:

```powershell
gh workflow run monitoring.yml --ref main `
  -f reference_data="s3://sleep-mlops-651709/features/reference/features.parquet" `
  -f current_data="s3://sleep-mlops-651709/monitoring/current"
```

Lời thoại:

> Monitoring dùng Evidently để so sánh dữ liệu reference với dữ liệu current được ingest từ production. Nếu phân phối feature thay đổi mạnh, workflow ghi báo cáo drift và có thể kích hoạt retrain.

Kết quả kỳ vọng:

- Có artifact drift report.
- Có file summary JSON.
- Nếu không có current data, workflow không fail vô ích mà ghi `skipped=true`.
- Nếu `alert=true`, workflow gọi retrain.

Nếu bị hỏi “khi nào retrain?”:

> Không nên retrain chỉ vì có một vài dòng dữ liệu mới. Production thường retrain khi có đủ dữ liệu mới, khi drift vượt ngưỡng, khi metric giảm, hoặc khi có nhãn mới đã được kiểm duyệt. Project này có monitoring định kỳ và có thể kích hoạt retrain khi drift alert.

## 15. Demo 10 - Retrain, promote và redeploy

Mở GitHub Actions workflow:

```text
Retrain - Promote - Redeploy
```

Chạy thủ công nếu muốn:

```powershell
gh workflow run retrain.yml --ref main `
  -f reason="Manual demo retrain" `
  -f training_data="s3://sleep-mlops-651709/features/reference/features.parquet" `
  -f extra_data="s3://sleep-mlops-651709/monitoring/current" `
  -f artifact_s3_uri="s3://sleep-mlops-651709/models" `
  -f model_type="all" `
  -f deploy_after_success="true"
```

Lời thoại:

> Retrain workflow đọc dữ liệu huấn luyện, có thể nối thêm dữ liệu current, train lại nhiều mô hình như LightGBM, XGBoost, RandomForest, chọn model có weighted F1 tốt nhất, log vào MLflow, upload artifact lên S3 và promote model nếu vượt ngưỡng. Nếu `deploy_after_success=true`, workflow gọi lại CI/CD để redeploy web app.

Điểm cần nói rõ:

- Promote không nên tùy tiện.
- Project dùng `MODEL_PROMOTE_THRESHOLD`, ví dụ `0.55`.
- Nếu metric không đạt ngưỡng, model không nên thay model production.

## 16. Câu chuyện kỹ thuật nên kể theo thứ tự

Bạn có thể kể project theo luồng này:

1. Notebook Kaggle là nguồn chuẩn:
   - đọc dữ liệu CAP/Balanced CAP;
   - chia epoch;
   - trích 24 đặc trưng;
   - train 7 lớp.
2. Code production dùng lại schema đó:
   - `feature_engineering/cap_features.py`;
   - `training/train.py`;
   - `sleep_portal/api/views.py`;
   - `inference/predictor.py`.
3. Model được quản lý bằng MLflow:
   - log run;
   - log metric;
   - log artifact;
   - registry stage `Production`.
4. Web app phục vụ inference:
   - Django + DRF;
   - dashboard HTML;
   - API predict/ingest.
5. Cloud deployment:
   - Docker;
   - ECR;
   - ECS Fargate;
   - ALB;
   - RDS PostgreSQL;
   - S3.
6. MLOps automation:
   - GitHub Actions CI/CD;
   - Evidently drift check;
   - retrain workflow;
   - redeploy tự động.

## 17. Các câu hỏi thường gặp và cách trả lời

### 17.1. Project này có phải MLOps chưa?

Trả lời:

> Có, ở mức prototype production MLOps. Project có model registry, artifact store, feature store, CI/CD, monitoring drift, retrain workflow và deploy lên ECS. Điểm còn cần nâng cấp cho production dài hạn là hạ tầng IaC đầy đủ bằng Terraform, HTTPS/domain riêng, auth cho MLflow và quy trình kiểm duyệt nhãn y khoa.

### 17.2. Vì sao chọn 24 đặc trưng thay vì deep learning trực tiếp trên EEG?

Trả lời:

> Vì dữ liệu hiện tại phù hợp để demo pipeline ổn định với feature engineering rõ ràng. 24 đặc trưng gồm band power, relative power, entropy, tần số, thống kê biên độ và Hjorth. Cách này dễ giải thích, chạy nhanh, phù hợp với MLOps prototype. Deep learning có thể là hướng phát triển sau.

### 17.3. Vì sao dùng LightGBM/XGBoost/RandomForest?

Trả lời:

> Đây là các mô hình mạnh cho dữ liệu dạng bảng sau khi trích feature. Chúng train nhanh, dễ so sánh metric, dễ log vào MLflow và deploy nhẹ hơn so với deep learning.

### 17.4. Vì sao metric khoảng 0.59 vẫn demo được?

Trả lời:

> Metric này cho thấy model chưa đủ để dùng lâm sàng thật. Tuy nhiên mục tiêu project là chứng minh pipeline MLOps end-to-end: từ dữ liệu, model, serving, monitoring đến retrain/deploy. Phần mô hình có thể tiếp tục cải thiện bằng split theo bệnh nhân, nhiều kênh tín hiệu, label chất lượng hơn và deep learning.

### 17.5. Vì sao không retrain mỗi khi có một bệnh nhân mới?

Trả lời:

> Retrain ngay khi có một ít dữ liệu mới có thể làm model nhiễu, tốn chi phí và dễ học nhầm pseudo-label. Production nên retrain khi đủ dữ liệu, có drift, metric giảm hoặc có nhãn được kiểm duyệt. Project có monitoring để quyết định khi nào cần retrain.

### 17.6. Vì sao cần S3 feature store?

Trả lời:

> Dữ liệu ingest mới cần được lưu lại ngoài database giao diện để workflow monitoring và retrain đọc được. S3 phù hợp vì rẻ, bền, dễ dùng từ GitHub Actions/ECS và lưu được Parquet.

### 17.7. Vì sao cần RDS PostgreSQL?

Trả lời:

> RDS lưu dữ liệu nghiệp vụ của web app: bệnh nhân, epoch, lịch sử dự đoán. Đây là dữ liệu cần query nhanh cho dashboard, khác với artifact/model/feature batch lưu trên S3.

### 17.8. Vì sao MLflow chạy server riêng?

Trả lời:

> Vì nhiều môi trường cần nhìn chung một nơi: training workflow, retrain workflow và web serving. MLflow server riêng với backend/artifact store giúp quản lý run, metric, artifact và registry nhất quán.

### 17.9. Nếu web app bị tắt để tiết kiệm chi phí thì bật lại thế nào?

Trả lời:

> Chạy workflow MLflow nếu cần, sau đó chạy CI/CD workflow. Script deploy sẽ kiểm tra ALB, target group, ECS service, task definition và smoke test. Sau khi workflow xanh, mở lại health/model-info.

## 18. Checklist trong lúc demo

Trước khi nói:

- Web app mở được.
- `/api/v1/health/` trả `ok`.
- `/api/v1/model-info/` trả `ready=True`.
- MLflow mở được.
- GitHub Actions run gần nhất xanh.
- Có dữ liệu bệnh nhân trong `/patients/`.

Trong lúc demo:

- Luôn giải thích “màn hình này chứng minh điều gì”.
- Không chỉ bấm, hãy nối màn hình vào pipeline MLOps.
- Nếu chạy script IoT, nói rõ script đang gọi API nào.
- Nếu mở MLflow, nói rõ model production được chọn như thế nào.
- Nếu mở GitHub Actions, nói rõ bước deploy nào biến code thành app chạy trên AWS.

Sau khi demo:

- Nếu đã tạo bệnh nhân tạm, xóa bệnh nhân tạm.
- Nếu đã trigger workflow tốn chi phí, kiểm tra lại service cần tắt hay giữ.
- Không xóa model/artifact production nếu chưa chắc.

## 19. Kịch bản lời thoại 3 phút

Bạn có thể đọc gần như nguyên văn:

> Project của em bắt đầu từ dữ liệu CAP Sleep, trong đó tín hiệu EEG được chia thành các epoch 2 giây. Từ mỗi epoch, hệ thống trích ra 24 đặc trưng như band power, relative power, spectral entropy, các thống kê biên độ và Hjorth. Notebook `kaggle_cap_training.ipynb` là quy chuẩn cho phần feature và training.
>
> Phần em xây dựng tiếp theo là production hóa notebook thành một hệ thống MLOps. Django REST API nhận vector feature hoặc file EDF, gọi model production để dự đoán 7 nhóm rối loạn giấc ngủ, sau đó dashboard hiển thị bệnh nhân, timeline epoch và phân bố dự đoán.
>
> Model không được quản lý thủ công trong code. Mỗi lần train hoặc retrain đều được log vào MLflow, gồm metric, artifact và model registry. Web app ưu tiên load model từ MLflow stage `Production`, nếu cần có fallback artifact từ S3.
>
> Khi có dữ liệu mới từ mô phỏng IoT, API `/ingest/` lưu bệnh nhân và epoch vào PostgreSQL, đồng thời ghi feature batch lên S3 để monitoring đọc. Workflow Monitoring dùng Evidently kiểm tra drift giữa dữ liệu reference và dữ liệu current. Nếu drift vượt ngưỡng hoặc cần chạy thủ công, workflow Retrain sẽ train lại, log vào MLflow, promote model nếu metric đạt ngưỡng và kích hoạt CI/CD redeploy.
>
> Toàn bộ web app được deploy bằng GitHub Actions: test, build Docker image, push Amazon ECR, deploy ECS Fargate qua ALB, chạy migrate và smoke test. Vì vậy project này không chỉ là model ML, mà là một pipeline MLOps end-to-end. Tuy nhiên metric hiện mới ở mức prototype nên hệ thống phù hợp để demo kỹ thuật và nghiên cứu, chưa dùng làm công cụ chẩn đoán y tế thật.

## 20. Kịch bản thao tác nhanh

Thứ tự bấm khi cần demo gọn:

1. Mở `/`.
2. Nói: dashboard tổng quan đọc dữ liệu từ PostgreSQL.
3. Mở `/pipeline/`.
4. Nói: đây là kiến trúc MLOps 6 lớp.
5. Mở `/api/v1/model-info/`.
6. Nói: model production đang ready, schema 24 feature.
7. Mở `/predict/`.
8. Chạy vector mẫu hoặc CSV batch.
9. Chạy:

   ```powershell
   .\demo_web_iot\run_rich_demo.ps1 `
     -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com `
     -MaxFiles 5
   ```

10. Mở `/patients/`.
11. Mở một bệnh nhân mixed.
12. Mở MLflow port `5000`.
13. Mở GitHub Actions CI/CD.
14. Kết luận: đây là vòng lặp MLOps end-to-end.

## 21. Khi có lỗi trong lúc demo

Nếu web app không mở:

```powershell
gh workflow run ci.yml --ref main -f reason="Restart production app for demo"
```

Nếu MLflow không mở:

```powershell
gh workflow run mlflow.yml --ref main -f reason="Restart MLflow for demo"
```

Nếu health lỗi:

```powershell
gh run list --workflow ci.yml --limit 3
```

Sau đó mở run mới nhất để xem job nào fail.

Nếu EDF upload lâu:

- Nói rõ web demo chỉ xử lý số epoch giới hạn.
- Chuyển sang vector/CSV demo hoặc IoT simulator.
- Không cố upload file EDF quá lớn trong lúc demo trực tiếp.

Nếu monitoring bị skipped:

- Giải thích đây là hành vi đúng khi chưa có current data.
- Chạy `run_rich_demo.ps1` để tạo current data rồi chạy lại monitoring.

Nếu retrain không promote:

- Giải thích promotion rule bảo vệ production.
- Model chỉ lên `Production` nếu metric vượt ngưỡng.

## 22. Kết luận nên nói ở cuối demo

Lời kết gợi ý:

> Tóm lại, project đã hoàn thành một luồng MLOps đầy đủ cho bài toán phát hiện rối loạn giấc ngủ: từ dữ liệu EEG, feature engineering, model training, MLflow registry, API serving, dashboard, ingest dữ liệu mới, monitoring drift, retrain và CI/CD deploy cloud. Hạn chế hiện tại là metric model còn ở mức prototype, dữ liệu chủ yếu là EEG và chưa đủ tiêu chuẩn lâm sàng. Hướng phát triển là tăng chất lượng dữ liệu, split theo bệnh nhân, thêm tín hiệu SpO2/ECG/airflow, cải thiện model và bổ sung bảo mật production như HTTPS, authentication và IaC bằng Terraform.
