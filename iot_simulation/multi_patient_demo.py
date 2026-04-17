"""
Multi-Patient IoT Demo
======================
Mô phỏng nhiều thiết bị EEG IoT gửi dữ liệu đồng thời:
  1. Sample 24 features trực tiếp từ phân phối thực (mean/std tính từ CAP Sleep DB)
  2. POST /api/v1/predict/  → nhận predictions
  3. POST /api/v1/ingest/   → lưu Patient + EpochPrediction vào DB

Features được sample từ Gaussian(mean, std) của từng rối loạn, lấy từ dữ liệu huấn luyện
thực tế (data/raw/balanced_CAP/) — không sinh EEG tổng hợp nữa, nên predictions chính xác hơn.

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

FEATURE_NAMES = [
    "delta_power", "delta_rel", "theta_power", "theta_rel",
    "alpha_power", "alpha_rel", "beta_power", "beta_rel",
    "gamma_power", "gamma_rel", "spectral_entropy", "peak_frequency",
    "mean_frequency", "amplitude_mean", "amplitude_std", "rms",
    "delta_beta_ratio", "theta_alpha_ratio", "skewness", "kurtosis",
    "zero_crossing_rate", "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
]

# ─── Phân phối feature thực từ CAP Sleep Database ────────────────────────────
# mean và std tính từ 500 mẫu/bệnh lý trong data/raw/balanced_CAP/
# Thứ tự khớp với FEATURE_NAMES (24 features)
FEATURE_STATS = {
    "healthy": {
        "mean": [81.008, 0.4393, 52.207, 0.3245, 14.656, 0.1022, 11.517, 0.0871,
                 0.5203, 0.0060, 1.7777, 2.5400, 5.2055, 14.741, 18.259, 18.518,
                 10.126, 4.2181, -0.0198, 0.0224, 0.02205, 394.63, 0.07342, 4.2062],
        "std":  [79.376, 0.1395, 36.308, 0.0721, 10.229, 0.0554, 10.923, 0.0851,
                 0.5186, 0.0094, 0.3400, 1.6746, 2.3060, 6.2953, 7.8256, 7.8616,
                 9.1802, 2.6434, 0.4527, 0.8752, 0.01197, 380.29, 0.03575, 1.2706],
    },
    "insomnia": {
        "mean": [178.39, 0.4043, 114.54, 0.2805, 80.745, 0.2039, 22.458, 0.0653,
                 2.3323, 0.0075, 1.8509, 2.2880, 5.4460, 27.580, 33.558, 34.141,
                 10.648, 1.6257, 0.0985, -0.2066, 0.02103, 1247.5, 0.07016, 4.7672],
        "std":  [112.63, 0.1287, 44.663, 0.0514, 38.712, 0.0801, 12.956, 0.0551,
                 2.3382, 0.0101, 0.3071, 1.3730, 1.8149, 9.2155, 11.016, 11.151,
                 9.0894, 0.8270, 0.4143, 0.6313, 0.01071, 856.96, 0.03105, 1.4064],
    },
    "narcolepsy": {
        "mean": [43.897, 0.3011, 47.463, 0.3427, 24.143, 0.1745, 12.698, 0.1052,
                 1.0059, 0.0125, 2.0455, 3.4240, 6.7097, 11.101, 13.868, 14.006,
                 4.2600, 2.5198, 0.0273, 0.0457, 0.03096, 216.14, 0.09834, 2.9375],
        "std":  [42.945, 0.1034, 36.390, 0.0758, 21.287, 0.0868, 10.244, 0.0587,
                 0.9490, 0.0160, 0.2151, 2.6359, 1.6826, 3.8684, 4.8809, 4.8756,
                 3.8810, 1.5067, 0.3661, 0.6616, 0.00832, 192.13, 0.02293, 0.6890],
    },
    "nfle": {
        "mean": [65.398, 0.3331, 45.798, 0.2788, 19.542, 0.1454, 16.327, 0.1571,
                 2.2965, 0.0197, 2.1343, 2.7720, 7.4714, 12.795, 16.143, 16.447,
                 3.9245, 2.4741, 0.1513, 0.3793, 0.02943, 341.86, 0.10537, 3.5507],
        "std":  [115.19, 0.1249, 64.604, 0.0851, 18.845, 0.0649, 13.722, 0.0889,
                 7.9932, 0.0351, 0.3717, 2.1991, 3.1397, 6.8064, 9.0143, 9.0866,
                 6.0370, 1.8999, 0.4937, 1.0514, 0.01277, 475.90, 0.04495, 1.0681],
    },
    "plm": {
        "mean": [42.320, 0.3335, 41.411, 0.3458, 21.909, 0.1892, 8.1334, 0.0791,
                 0.7360, 0.0080, 1.9780, 3.0240, 6.1211, 11.433, 14.260, 14.432,
                 6.5617, 2.2719, 0.0235, 0.0687, 0.02728, 230.05, 0.08773, 3.9643],
        "std":  [37.797, 0.1169, 26.438, 0.0655, 15.931, 0.0856, 6.1750, 0.0451,
                 1.0277, 0.0078, 0.2369, 2.1070, 1.7109, 4.0995, 5.1682, 5.2033,
                 6.0017, 1.2388, 0.4281, 0.7079, 0.00934, 193.48, 0.02515, 1.0196],
    },
    "rbd": {
        "mean": [29.316, 0.4143, 21.329, 0.3485, 7.8520, 0.1325, 3.5448, 0.0646,
                 0.4731, 0.0083, 1.8017, 2.6520, 5.2563, 11.680, 13.731, 14.374,
                 10.002, 3.0535, 0.0565, -0.2322, 0.01850, 216.99, 0.06568, 6.0104],
        "std":  [24.316, 0.1489, 13.402, 0.1009, 5.8843, 0.0544, 3.7927, 0.0409,
                 1.2398, 0.0099, 0.2951, 1.4963, 1.7701, 4.5471, 5.3332, 5.4303,
                 8.3657, 1.4147, 0.4397, 0.7495, 0.00946, 176.62, 0.02764, 1.7939],
    },
    "sdb": {
        "mean": [43.621, 0.2239, 53.029, 0.2941, 30.077, 0.1989, 27.731, 0.1980,
                 4.7930, 0.0318, 2.3316, 5.1280, 9.4636, 13.093, 16.135, 16.474,
                 2.8898, 1.8718, 0.0392, 0.1181, 0.04036, 291.47, 0.13174, 3.2566],
        "std":  [47.663, 0.1398, 42.213, 0.1105, 15.358, 0.0785, 26.601, 0.1360,
                 10.002, 0.0320, 0.3979, 4.1598, 4.0163, 4.6116, 5.5781, 5.6981,
                 3.8571, 1.3318, 0.3632, 0.8059, 0.01763, 233.95, 0.05390, 1.0757],
    },
}

# Giới hạn dưới hợp lệ cho từng feature (power/amplitude không thể âm)
FEATURE_CLIP_LOW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # band powers + rel
    0.0, 0.0, 0.0, 0.0, 0.0,                   # gamma_power, gamma_rel, spectral_entropy, peak_freq, mean_freq
    0.0, 0.0, 0.0,                              # amplitude_mean, amplitude_std, rms
    0.0, 0.0, None, None,                       # ratios, skewness, kurtosis
    0.0, 0.0, 0.0, 0.0,                         # zcr, hjorth x3
]

# ─── Danh sách bệnh nhân demo ────────────────────────────────────────────────
DEFAULT_PATIENTS = [
    {"patient_id": "PT-001", "disorder": "insomnia",   "age": 42, "gender": "F"},
    {"patient_id": "PT-002", "disorder": "nfle",       "age": 28, "gender": "M"},
    {"patient_id": "PT-003", "disorder": "healthy",    "age": 35, "gender": "F"},
    {"patient_id": "PT-004", "disorder": "sdb",        "age": 55, "gender": "M"},
    {"patient_id": "PT-005", "disorder": "narcolepsy", "age": 22, "gender": "M"},
]


# ─── Feature sampler — dựa trên phân phối thực từ data ───────────────────────
def _sample_features(disorder: str) -> list:
    """
    Sample 1 epoch features từ phân phối Gaussian(mean, std) của disorder.
    Thực tế hơn nhiều so với sinh EEG tổng hợp vì dùng đúng phân phối training data.
    """
    stats = FEATURE_STATS[disorder]
    mean = np.array(stats["mean"], dtype=np.float64)
    std  = np.array(stats["std"],  dtype=np.float64)

    # Sample từ Gaussian, clip các feature không thể âm
    sampled = np.random.normal(mean, std)
    for i, lo in enumerate(FEATURE_CLIP_LOW):
        if lo is not None:
            sampled[i] = max(sampled[i], lo)

    # Đảm bảo relative power sum ≈ 1 (chuẩn hóa lại)
    rel_indices = [1, 3, 5, 7, 9]  # delta_rel, theta_rel, alpha_rel, beta_rel, gamma_rel
    rel_sum = sampled[rel_indices].sum()
    if rel_sum > 0:
        sampled[rel_indices] /= rel_sum

    return sampled.tolist()


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

    all_epoch_records = []
    all_predictions = []
    epoch_idx = 0
    buf_feats = []

    logger.info(f"[{pid}] 🟢 Bắt đầu — disorder={disorder}, epochs={n_epochs}")

    while epoch_idx < n_epochs:
        feats = _sample_features(disorder)
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

    logger.info("=" * 60)
    logger.info("🏥 MULTI-PATIENT IoT SLEEP MONITORING DEMO")
    logger.info("   Features: sampled từ phân phối thực CAP Sleep DB")
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
