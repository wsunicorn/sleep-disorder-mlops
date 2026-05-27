# Bộ demo IoT cho web app

Thư mục này dùng để demo các luồng API và giao diện của hệ thống Sleep Disorder MLOps mà không phải xử lý lại EDF lớn trong lúc trình bày.

## Các mức demo

### 1. Demo nhanh bằng file tĩnh

Các file `predict_single_healthy.json`, `predict_batch.csv` và `ingest_patient_*.json` phù hợp khi cần kiểm thử nhanh REST API.

```powershell
.\demo_web_iot\run_demo_rest.ps1 `
  -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com
```

Script sẽ gọi:

- `GET /api/v1/health/`
- `GET /api/v1/model-info/`
- `POST /api/v1/predict/`
- `POST /api/v1/ingest/`

### 2. Demo nạp nhiều bệnh nhân

`generate_rich_iot_demo.py` sinh bộ dữ liệu lớn từ `feature_stats.json`: đủ 7 lớp bệnh, nhiều bệnh nhân, nhiều epoch và vài ca mixed để biểu đồ timeline sinh động.

```powershell
.\demo_web_iot\run_rich_demo.ps1 `
  -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com `
  -Regenerate `
  -PatientsPerClass 3 `
  -MixedPatients 3 `
  -EpochsPerPatient 48
```

Kịch bản này phù hợp để làm dashboard và danh sách bệnh nhân có nhiều dữ liệu ngay lập tức.

### 3. Demo realtime IoT

`realtime_iot_stream.py` mô phỏng nhiều gateway IoT gửi dữ liệu theo từng nhịp nhỏ. Mỗi chu kỳ sẽ:

1. Sinh 24 đặc trưng EEG cho từng epoch.
2. Gọi model production qua `POST /api/v1/predict/`.
3. Lấy nhãn dự đoán trả về.
4. Gửi kết quả dự đoán và feature vào `POST /api/v1/ingest/`.
5. Ghi state vào `demo_web_iot/runtime/realtime_state.json` để lần sau chạy tiếp epoch mới.

Chạy demo production:

```powershell
.\demo_web_iot\run_realtime_iot.ps1 `
  -BaseUrl http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com `
  -SessionId live-demo `
  -PatientsPerClass 1 `
  -MixedPatients 1 `
  -Cycles 8 `
  -EpochsPerCycle 4 `
  -Interval 1.5 `
  -Workers 2 `
  -Retries 5
```

Script không gọi preflight `/api/v1/health/` mặc định để tránh tốn quota throttle trước khi demo. Nếu cần kiểm tra health riêng trước khi chạy, thêm `-CheckApi`.

Chạy lại cùng `SessionId` sẽ tiếp tục tăng `epoch_index`, giống thiết bị IoT gửi thêm dữ liệu mới. Nếu muốn bắt đầu lại từ epoch 0:

```powershell
.\demo_web_iot\run_realtime_iot.ps1 `
  -SessionId live-demo `
  -ResetSession
```

Nếu production trả về `429 Too Many Requests`, nghĩa là API đang giới hạn tần suất request. Hãy giảm nhịp gửi hoặc số luồng:

```powershell
.\demo_web_iot\run_realtime_iot.ps1 `
  -SessionId live-demo `
  -Workers 1 `
  -Interval 3 `
  -Retries 8
```

Chạy kiểm tra không gửi API:

```powershell
.\demo_web_iot\run_realtime_iot.ps1 `
  -DryRun `
  -Cycles 2 `
  -EpochsPerCycle 2
```

Chạy liên tục cho tới khi bấm `Ctrl+C`:

```powershell
.\demo_web_iot\run_realtime_iot.ps1 `
  -SessionId live-demo `
  -Cycles 0 `
  -Interval 2
```

### 4. Dọn dữ liệu demo/bệnh nhân

`delete_patients.py` và `delete_patients.ps1` dùng route xóa có sẵn của dashboard để xóa bệnh nhân cùng toàn bộ epoch liên quan. Mặc định script chỉ liệt kê hoặc dry-run; muốn xóa thật phải thêm `-Yes`.

Liệt kê các bệnh nhân có diagnosis không thuộc 7 lớp chính thức:

```powershell
.\demo_web_iot\delete_patients.ps1 -UnknownDiagnosis -List
```

Xóa các bệnh nhân có diagnosis lạ:

```powershell
.\demo_web_iot\delete_patients.ps1 -UnknownDiagnosis -Yes
```

Xóa các ca demo mixed cũ:

```powershell
.\demo_web_iot\delete_patients.ps1 -MixedDemo -Yes
```

Xóa toàn bộ dữ liệu demo đã ingest, giữ lại các hồ sơ mẫu CAP `PT-*`:

```powershell
.\demo_web_iot\delete_patients.ps1 -AllDemo -List
.\demo_web_iot\delete_patients.ps1 -AllDemo -Yes
```

Xóa riêng từng nhóm demo:

```powershell
.\demo_web_iot\delete_patients.ps1 -DemoRich -Yes
.\demo_web_iot\delete_patients.ps1 -RealtimeDemo -Yes
.\demo_web_iot\delete_patients.ps1 -QuickDemo -Yes
```

Xóa theo mã cụ thể hoặc prefix:

```powershell
.\demo_web_iot\delete_patients.ps1 -PatientId demo-iot-mixed-001 -Yes
.\demo_web_iot\delete_patients.ps1 -IdPrefix demo-rich- -Yes
```

## Ý nghĩa dữ liệu realtime

- `patient_id` có dạng `iot-<session>-<label>-<index>` để dễ tìm trên trang `/patients/`.
- Với ca `mixed`, chữ `mixed` chỉ nằm trong mã bệnh nhân để nhận biết kịch bản timeline nhiều nhãn; trường `disorder` vẫn luôn thuộc 7 lớp chính thức: `healthy`, `insomnia`, `narcolepsy`, `nfle`, `plm`, `rbd`, `sdb`.
- `device_id` có dạng `gw-<session>-...`, mô phỏng gateway hoặc thiết bị đeo gửi dữ liệu.
- `label` trong payload là nhãn mô phỏng chưa xác thực, dùng cho monitoring/drift và demo timeline.
- Script không gửi `training_approved=true`, vì dữ liệu IoT demo không nên tự động trở thành ground truth huấn luyện.
- Nếu muốn dữ liệu mới được retrain dùng thật, cần có quy trình xác thực nhãn và thêm cờ `training_approved` hoặc `label_verified` cho từng epoch.

## Trang nên mở khi demo

- Dashboard: `/`
- Studio suy luận: `/predict/`
- Hồ sơ bệnh nhân: `/patients/`
- Một bệnh nhân realtime: `/patients/iot-live-demo-healthy-01/`
- Một ca mixed realtime: `/patients/iot-live-demo-mixed-01/`
- Pipeline MLOps: `/pipeline/`
