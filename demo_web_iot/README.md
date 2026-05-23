# Bộ dữ liệu demo IoT cho web app

Thư mục này dùng để demo từng phần của website Sleep Disorder MLOps mà không cần chuẩn bị lại dữ liệu từ đầu.

## Các file chính

- `feature_names.json`: danh sách 24 đặc trưng EEG đúng theo notebook `kaggle_cap_training.ipynb`.
- `predict_single_healthy.json`: payload mẫu cho `POST /api/v1/predict/`.
- `predict_batch.csv`: batch CSV để tải trực tiếp lên tab "CSV theo lô" trên trang Studio Suy luận.
- `ingest_patient_healthy.json`: phiên IoT giả lập cho bệnh nhân bình thường.
- `ingest_patient_insomnia.json`: phiên IoT giả lập cho bệnh nhân mất ngủ.
- `ingest_patient_mixed.json`: phiên IoT giả lập có nhiều nhãn dự đoán để demo biểu đồ timeline.
- `run_demo_rest.ps1`: script gọi lần lượt health, model-info, predict single, predict batch và ingest.

## Chạy nhanh bằng PowerShell

Chạy với server local:

```powershell
.\demo_web_iot\run_demo_rest.ps1 -BaseUrl http://127.0.0.1:8000
```

Chạy với production ALB:

```powershell
.\demo_web_iot\run_demo_rest.ps1 -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
```

Sau khi chạy xong, mở:

- `/` để xem KPI tổng quan.
- `/patients/` để xem các bệnh nhân demo.
- `/patients/demo-iot-mixed-001/` để xem timeline epoch.
- `/pipeline/` để trình bày luồng MLOps.

## Demo bằng giao diện

1. Mở `/predict/`.
2. Tab "Vector đơn lẻ": dán nội dung trong `predict_single_healthy.json`, hoặc bấm "Tải dữ liệu mẫu".
3. Tab "CSV theo lô": tải file `predict_batch.csv`.
4. Tab "JSON / API": dùng curl mẫu hoặc chạy script `run_demo_rest.ps1`.
5. Mở `/patients/` để xác nhận dữ liệu IoT đã được ingest vào hệ thống.
