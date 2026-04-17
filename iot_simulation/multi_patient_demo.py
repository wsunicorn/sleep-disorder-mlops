"""
Multi-Patient IoT Demo
======================
Mô phỏng nhiều thiết bị EEG IoT gửi dữ liệu đồng thời:
  1. Sinh tín hiệu EEG tổng hợp theo profile bệnh lý
  2. Tính 24 features / epoch
  3. POST /api/v1/predict/  → nhận predictions
  4. POST /api/v1/ingest/   → lưu Patient + EpochPrediction vào DB

Chạy:
  python iot_simulation/multi_patient_demo.py
  python iot_simulation/multi_patient_demo.py --url http://localhost:8000
  python iot_simulation/multi_patient_demo.py --epochs 40 --delay 0.05
"""

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from loguru import logger

# ─── Config mặc định ────────────────────────────────────────────────────────
DEFAULT_URL = "http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com"
SFREQ = 512
WINDOW_SAMPLES = int(SFREQ * 2.0)  # 1024 mẫu / 2 giây

# ─── Profile EEG theo bệnh lý ───────────────────────────────────────────────
DISORDER_PROFILES = {
    "healthy":    {"delta": 0.30, "theta": 0.20, "alpha": 0.30, "beta": 0.15, "gamma": 0.05, "noise": 0.30},
    "insomnia":   {"delta": 0.10, "theta": 0.30, "alpha": 0.40, "beta": 0.15, "gamma": 0.05, "noise": 0.50},
    "narcolepsy": {"delta": 0.50, "theta": 0.20, "alpha": 0.10, "beta": 0.10, "gamma": 0.10, "noise": 0.20},
    "nfle":       {"delta": 0.40, "theta": 0.30, "alpha": 0.10, "beta": 0.15, "gamma": 0.05, "noise": 0.60},
    "rbd":        {"delta": 0.20, "theta": 0.20, "alpha": 0.20, "beta": 0.25, "gamma": 0.15, "noise": 0.40},
    "plm":        {"delta": 0.35, "theta": 0.25, "alpha": 0.20, "beta": 0.15, "gamma": 0.05, "noise": 0.40},
    "sdb":        {"delta": 0.45, "theta": 0.20, "alpha": 0.15, "beta": 0.10, "gamma": 0.10, "noise": 0.70},
}

# ─── Danh sách bệnh nhân demo ────────────────────────────────────────────────
DEFAULT_PATIENTS = [
    {"patient_id": "PT-001", "disorder": "insomnia",   "age": 42, "gender": "F"},
    {"patient_id": "PT-002", "disorder": "nfle",       "age": 28, "gender": "M"},
    {"patient_id": "PT-003", "disorder": "healthy",    "age": 35, "gender": "F"},
    {"patient_id": "PT-004", "disorder": "sdb",        "age": 55, "gender": "M"},
    {"patient_id": "PT-005", "disorder": "narcolepsy", "age": 22, "gender": "M"},
]


# ─── EEG generator ──────────────────────────────────────────────────────────
def _generate_eeg(profile: dict) -> np.ndarray:
    t = np.linspace(0, WINDOW_SAMPLES / SFREQ, WINDOW_SAMPLES)
    sig = np.zeros(WINDOW_SAMPLES)
    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 35.0}
    for band, amp in profile.items():
        if band == "noise":
            sig += amp * np.random.randn(WINDOW_SAMPLES) * 1e-5
        else:
            phase = np.random.uniform(0, 2 * np.pi)
            sig += amp * np.sin(2 * np.pi * band_freqs[band] * t + phase) * 1e-5
    return sig.astype(np.float32)


# ─── Feature extraction ──────────────────────────────────────────────────────
def _extract_features(signal: np.ndarray) -> list:
    from scipy import signal as scipy_signal
    from scipy.stats import entropy as scipy_entropy, skew, kurtosis as kurt

    BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
             "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}

    def bp(psd, freqs, lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(psd[idx], freqs[idx])) if idx.any() else 0.0

    nperseg = min(256, len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=SFREQ, nperseg=nperseg)
    total = bp(psd, freqs, 0.5, 40.0) + 1e-12

    feats, band_pows = [], {}
    for name, (lo, hi) in BANDS.items():
        p = bp(psd, freqs, lo, hi)
        band_pows[name] = p
        feats += [p, p / total]

    psd_norm = psd / (psd.sum() + 1e-12)
    feats += [
        float(scipy_entropy(psd_norm + 1e-12)),
        float(freqs[np.argmax(psd)]),
        float(np.sum(freqs * psd) / (psd.sum() + 1e-12)),
        float(np.mean(np.abs(signal))),
        float(np.std(signal)),
        float(np.sqrt(np.mean(signal ** 2))),
        band_pows["delta"] / (band_pows["beta"] + 1e-12),
        band_pows["theta"] / (band_pows["alpha"] + 1e-12),
        float(skew(signal)),
        float(kurt(signal)),
    ]
    zcr = float(np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal)))
    feats.append(zcr)
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    act = float(np.var(signal))
    mob = float(np.sqrt(np.var(d1) / (act + 1e-12)))
    cmp = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mob + 1e-12))
    feats += [act, mob, cmp]
    return feats  # 24 features


# ─── API helpers ─────────────────────────────────────────────────────────────
def _predict(api_url: str, features_batch: list) -> list:
    resp = requests.post(
        f"{api_url.rstrip('/')}/api/v1/predict/",
        json={"features": features_batch},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("predictions", [])


def _ingest(api_url: str, patient_id: str, disorder: str,
            age: Optional[int], gender: Optional[str],
            epoch_records: list, retries: int = 4) -> dict:
    payload = {
        "patient_id": patient_id,
        "disorder": disorder,
        "age": age,
        "gender": gender,
        "epochs": epoch_records,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/api/v1/ingest/",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt < retries - 1:
                wait = 0.5 * (2 ** attempt) + np.random.uniform(0, 0.3)
                time.sleep(wait)
            else:
                raise


# ─── Per-patient worker ──────────────────────────────────────────────────────
def run_patient(patient: dict, n_epochs: int, batch_size: int,
                delay: float, api_url: str) -> dict:
    pid = patient["patient_id"]
    disorder = patient["disorder"]
    age = patient.get("age")
    gender = patient.get("gender")
    profile = DISORDER_PROFILES[disorder]

    all_epoch_records = []
    all_predictions = []
    epoch_idx = 0
    buf_feats = []

    logger.info(f"[{pid}] 🟢 Bắt đầu — disorder={disorder}, epochs={n_epochs}")

    while epoch_idx < n_epochs:
        sig = _generate_eeg(profile)
        feats = _extract_features(sig)
        buf_feats.append(feats)

        flush = (len(buf_feats) >= batch_size) or (epoch_idx == n_epochs - 1)
        if flush:
            ts = datetime.now(tz=timezone.utc)
            try:
                preds = _predict(api_url, buf_feats)
                for i, pred in enumerate(preds):
                    ep = epoch_idx - len(buf_feats) + 1 + i
                    all_predictions.append(pred)
                    all_epoch_records.append({
                        "epoch_index": ep,
                        "predicted_class": pred,
                        "confidence": None,
                        "timestamp": ts.isoformat(),
                    })
                    logger.info(f"  [{pid}] epoch {ep+1:03d}/{n_epochs} → {pred}")
            except Exception as exc:
                logger.error(f"  [{pid}] predict error: {exc}")
            buf_feats = []

        epoch_idx += 1
        time.sleep(delay)

    # Xác định chẩn đoán chính từ majority vote
    counts = Counter(all_predictions)
    dominant = counts.most_common(1)[0][0] if counts else disorder

    # Lưu vào DB qua API ingest
    ingest_result = {}
    try:
        ingest_result = _ingest(api_url, pid, dominant, age, gender, all_epoch_records)
        logger.info(
            f"[{pid}] 💾 Đã lưu — diagnosis={dominant}, "
            f"epochs_saved={ingest_result.get('epochs_saved')}, "
            f"patient_created={ingest_result.get('patient_created')}"
        )
    except Exception as exc:
        logger.error(f"[{pid}] ingest error: {exc}")

    # In thống kê
    logger.info(f"[{pid}] 📈 Kết quả ({len(all_predictions)} epochs):")
    for cls, cnt in counts.most_common():
        pct = cnt / len(all_predictions) * 100
        bar = "█" * max(1, int(pct / 5))
        logger.info(f"  {cls:15s} {cnt:3d} ({pct:5.1f}%)  {bar}")
    logger.info(f"[{pid}] → Chẩn đoán chính: {dominant.upper()}")

    return {
        "patient_id": pid,
        "dominant": dominant,
        "counts": dict(counts),
        "ingest": ingest_result,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-Patient IoT Demo")
    parser.add_argument("--url", default=DEFAULT_URL, help="Django API base URL")
    parser.add_argument("--epochs", type=int, default=20, help="Số epochs mỗi bệnh nhân")
    parser.add_argument("--batch-size", type=int, default=5, help="Số epochs / lần gọi API")
    parser.add_argument("--delay", type=float, default=0.1, help="Độ trễ giữa các epoch (giây)")
    parser.add_argument("--workers", type=int, default=3, help="Số bệnh nhân chạy song song")
    args = parser.parse_args()

    api_url = args.url

    # Kiểm tra API
    logger.info("=" * 60)
    logger.info("🏥 MULTI-PATIENT IoT SLEEP MONITORING DEMO")
    logger.info("=" * 60)
    try:
        health = requests.get(f"{api_url.rstrip('/')}/api/v1/health/", timeout=5)
        health.raise_for_status()
        logger.info(f"✅ API: {api_url} → OK")
    except Exception as e:
        logger.error(f"❌ API không phản hồi: {e}")
        logger.error("Chạy local: cd sleep_portal && python manage.py runserver")
        return

    try:
        info = requests.get(f"{api_url.rstrip('/')}/api/v1/model-info/", timeout=5).json()
        logger.info(f"📊 Model: {info.get('model_name')} | Features: {info.get('feature_count')} | Ready: {info.get('ready')}")
    except Exception:
        pass

    logger.info(f"👥 Bệnh nhân: {len(DEFAULT_PATIENTS)} | Epochs/patient: {args.epochs} | Workers: {args.workers}")
    logger.info("=" * 60)

    # Chạy song song các bệnh nhân
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_patient, p, args.epochs, args.batch_size, args.delay, api_url
            ): p["patient_id"]
            for p in DEFAULT_PATIENTS
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"[{pid}] worker error: {exc}")

    # Tổng kết
    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 TỔNG KẾT TOÀN BỘ BỆNH NHÂN")
    logger.info("=" * 60)
    for r in sorted(results, key=lambda x: x["patient_id"]):
        pid = r["patient_id"]
        diag = r["dominant"].upper()
        saved = r["ingest"].get("epochs_saved", "?")
        new = "✨ MỚI" if r["ingest"].get("patient_created") else "  cập nhật"
        logger.info(f"  {pid}  →  {diag:15s}  ({saved} epochs lưu)  {new}")
    logger.info("=" * 60)
    logger.info(f"✅ Hoàn thành! Xem bệnh nhân tại: {api_url.rstrip('/')}/patients/")


if __name__ == "__main__":
    main()
