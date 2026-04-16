"""
IoT Local Demo — Mô phỏng đầy đủ pipeline không cần EDF, MQTT hay AWS.

Luồng:
  Synthetic EEG generator
    → Tính 24 features (giống hệt API)
    → POST /api/v1/predict/  (local hoặc production)
    → In kết quả dự đoán theo thời gian thực

Chạy:
  python iot_simulation/demo_local.py
  python iot_simulation/demo_local.py --url http://localhost:8000
  python iot_simulation/demo_local.py --patient-id p1 --epochs 20 --delay 0.5
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import requests
from loguru import logger

# ── Cấu hình mặc định ──────────────────────────────────────────────────────
DEFAULT_URL = "http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com"
SFREQ = 512          # Hz
WINDOW_SEC = 2.0     # giây / epoch
WINDOW_SAMPLES = int(SFREQ * WINDOW_SEC)  # 1024 mẫu

# Profile EEG mô phỏng các bệnh lý khác nhau
DISORDER_PROFILES = {
    "healthy":    {"delta": 0.3, "theta": 0.2, "alpha": 0.3, "beta": 0.15, "gamma": 0.05, "noise": 0.3},
    "insomnia":   {"delta": 0.1, "theta": 0.3, "alpha": 0.4, "beta": 0.15, "gamma": 0.05, "noise": 0.5},
    "narcolepsy": {"delta": 0.5, "theta": 0.2, "alpha": 0.1, "beta": 0.10, "gamma": 0.10, "noise": 0.2},
    "nfle":       {"delta": 0.4, "theta": 0.3, "alpha": 0.1, "beta": 0.15, "gamma": 0.05, "noise": 0.6},
    "rbd":        {"delta": 0.2, "theta": 0.2, "alpha": 0.2, "beta": 0.25, "gamma": 0.15, "noise": 0.4},
    "plm":        {"delta": 0.35, "theta": 0.25, "alpha": 0.2, "beta": 0.15, "gamma": 0.05, "noise": 0.4},
    "sdb":        {"delta": 0.45, "theta": 0.2, "alpha": 0.15, "beta": 0.10, "gamma": 0.10, "noise": 0.7},
}


def generate_eeg_epoch(profile: dict, sfreq: float = 512, n_samples: int = 1024) -> np.ndarray:
    """
    Tạo tín hiệu EEG giả dựa trên profile bệnh lý.
    Kết hợp các sóng sin theo tần số đặc trưng của từng băng tần.
    """
    t = np.linspace(0, n_samples / sfreq, n_samples)
    signal = np.zeros(n_samples)

    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 35.0}
    for band, amp in profile.items():
        if band == "noise":
            signal += amp * np.random.randn(n_samples) * 1e-5
        else:
            freq = band_freqs[band]
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase) * 1e-5

    return signal.astype(np.float32)


def extract_24_features(signal: np.ndarray, sfreq: float = 512) -> list:
    """
    Tính 24 features từ tín hiệu EEG 1 chiều (giống hệt PredictEDFView).
    """
    from scipy import signal as scipy_signal
    from scipy.stats import entropy as scipy_entropy, skew, kurtosis as kurt

    FREQ_BANDS = {
        "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0), "gamma": (30.0, 40.0),
    }

    def bandpower(psd, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(psd[idx], freqs[idx])) if idx.sum() > 0 else 0.0

    nperseg = min(256, len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=sfreq, nperseg=nperseg)
    total_power = bandpower(psd, freqs, 0.5, 40.0) + 1e-12

    feats = []
    band_powers = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        bp = bandpower(psd, freqs, fmin, fmax)
        band_powers[band_name] = bp
        feats.append(bp)
        feats.append(bp / total_power)

    psd_norm = psd / (psd.sum() + 1e-12)
    feats.append(float(scipy_entropy(psd_norm + 1e-12)))
    feats.append(float(freqs[np.argmax(psd)]))
    feats.append(float(np.sum(freqs * psd) / (psd.sum() + 1e-12)))
    feats.append(float(np.mean(np.abs(signal))))
    feats.append(float(np.std(signal)))
    feats.append(float(np.sqrt(np.mean(signal ** 2))))
    feats.append(band_powers["delta"] / (band_powers["beta"] + 1e-12))
    feats.append(band_powers["theta"] / (band_powers["alpha"] + 1e-12))
    feats.append(float(skew(signal)))
    feats.append(float(kurt(signal)))
    zcr = float(np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal)))
    feats.append(zcr)
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = float(np.var(signal))
    mobility = float(np.sqrt(np.var(diff1) / (activity + 1e-12)))
    complexity = float(np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12))
    feats.extend([activity, mobility, complexity])

    return feats  # 24 features


def predict_via_api(url: str, features_batch: list) -> dict:
    """Gửi batch features lên REST API và trả về kết quả."""
    resp = requests.post(
        f"{url.rstrip('/')}/api/v1/predict/",
        json={"features": features_batch},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def run_demo(
    patient_id: str,
    disorder: str,
    n_epochs: int,
    delay: float,
    api_url: str,
    batch_size: int,
):
    profile = DISORDER_PROFILES[disorder]
    logger.info(f"🚀 IoT Demo bắt đầu")
    logger.info(f"   Patient: {patient_id} | Disorder profile: {disorder}")
    logger.info(f"   API: {api_url}")
    logger.info(f"   Epochs: {n_epochs} | Batch size: {batch_size} | Delay: {delay}s")
    logger.info("─" * 60)

    # Kiểm tra API sẵn sàng
    try:
        health = requests.get(f"{api_url.rstrip('/')}/api/v1/health/", timeout=5)
        health.raise_for_status()
        logger.info("✅ API health check: OK")
    except Exception as e:
        logger.error(f"❌ API không phản hồi: {e}")
        logger.error("Hãy chạy Django local: cd sleep_portal && python manage.py runserver")
        return

    # Lấy model info
    try:
        info = requests.get(f"{api_url.rstrip('/')}/api/v1/model-info/", timeout=5).json()
        logger.info(f"📊 Model: {info.get('model_name')} | Features: {info.get('feature_count')} | Ready: {info.get('ready')}")
    except Exception:
        pass

    logger.info("─" * 60)

    all_predictions = []
    epoch_idx = 0
    batch_buffer = []
    batch_start_idx = 0

    while epoch_idx < n_epochs:
        # Tạo tín hiệu EEG giả
        signal = generate_eeg_epoch(profile, sfreq=SFREQ, n_samples=WINDOW_SAMPLES)
        features = extract_24_features(signal, sfreq=SFREQ)
        batch_buffer.append(features)

        # Gửi API khi đủ batch hoặc epoch cuối
        if len(batch_buffer) >= batch_size or epoch_idx == n_epochs - 1:
            timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
            try:
                result = predict_via_api(api_url, batch_buffer)
                predictions = result.get("predictions", [])
                for i, pred in enumerate(predictions):
                    ep = batch_start_idx + i
                    all_predictions.append(pred)
                    logger.info(
                        f"[{timestamp}] Epoch {ep+1:03d}/{n_epochs} | "
                        f"Patient: {patient_id} | "
                        f"Predicted: {pred:15s} | "
                        f"Cached: {result.get('cached', False)}"
                    )
            except Exception as e:
                logger.error(f"[{timestamp}] Epoch {epoch_idx+1}: API error — {e}")

            batch_start_idx += len(batch_buffer)
            batch_buffer = []

        epoch_idx += 1
        time.sleep(delay)

    # Thống kê kết quả
    if all_predictions:
        from collections import Counter
        counts = Counter(all_predictions)
        logger.info("─" * 60)
        logger.info(f"📈 KẾT QUẢ PHÂN TÍCH GIẤC NGỦ — Patient: {patient_id}")
        logger.info(f"   Tổng epochs phân tích: {len(all_predictions)}")
        for disorder_class, count in counts.most_common():
            pct = count / len(all_predictions) * 100
            bar = "█" * int(pct / 5)
            logger.info(f"   {disorder_class:15s}: {count:4d} epochs ({pct:5.1f}%)  {bar}")
        dominant = counts.most_common(1)[0][0]
        logger.info(f"   → Chẩn đoán chính: {dominant.upper()}")
        logger.info("─" * 60)


def main():
    parser = argparse.ArgumentParser(description="IoT Sleep Disorder Demo")
    parser.add_argument("--patient-id", default="demo_patient", help="ID bệnh nhân")
    parser.add_argument(
        "--disorder", default="healthy",
        choices=list(DISORDER_PROFILES.keys()),
        help="Loại rối loạn giả lập",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Số epochs gửi lên API")
    parser.add_argument("--delay", type=float, default=0.2, help="Độ trễ giữa các epoch (giây)")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL của Django API")
    parser.add_argument("--batch-size", type=int, default=5, help="Số epochs mỗi lần gọi API")
    args = parser.parse_args()

    run_demo(
        patient_id=args.patient_id,
        disorder=args.disorder,
        n_epochs=args.epochs,
        delay=args.delay,
        api_url=args.url,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
