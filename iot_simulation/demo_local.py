"""Single-patient IoT demo using the Kaggle notebook feature schema."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime

import numpy as np
import requests
from loguru import logger

from feature_engineering.cap_features import (
    SFREQ,
    WINDOW_SAMPLES,
    extract_feature_vector,
)

DEFAULT_URL = "http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com"

DISORDER_PROFILES = {
    "healthy": {"delta": 0.3, "theta": 0.2, "alpha": 0.3, "beta": 0.15, "gamma": 0.05, "noise": 0.3},
    "insomnia": {"delta": 0.1, "theta": 0.3, "alpha": 0.4, "beta": 0.15, "gamma": 0.05, "noise": 0.5},
    "narcolepsy": {"delta": 0.5, "theta": 0.2, "alpha": 0.1, "beta": 0.10, "gamma": 0.10, "noise": 0.2},
    "nfle": {"delta": 0.4, "theta": 0.3, "alpha": 0.1, "beta": 0.15, "gamma": 0.05, "noise": 0.6},
    "rbd": {"delta": 0.2, "theta": 0.2, "alpha": 0.2, "beta": 0.25, "gamma": 0.15, "noise": 0.4},
    "plm": {"delta": 0.35, "theta": 0.25, "alpha": 0.2, "beta": 0.15, "gamma": 0.05, "noise": 0.4},
    "sdb": {"delta": 0.45, "theta": 0.2, "alpha": 0.15, "beta": 0.10, "gamma": 0.10, "noise": 0.7},
}


def generate_eeg_epoch(
    profile: dict[str, float],
    sfreq: float = SFREQ,
    n_samples: int = WINDOW_SAMPLES,
) -> np.ndarray:
    """Generate a small synthetic EEG window for API smoke demos."""
    t = np.linspace(0, n_samples / sfreq, n_samples)
    signal = np.zeros(n_samples)
    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 35.0}
    for band, amp in profile.items():
        if band == "noise":
            signal += amp * np.random.randn(n_samples) * 1e-5
        else:
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * band_freqs[band] * t + phase) * 1e-5
    return signal.astype(np.float32)


def predict_via_api(url: str, features_batch: list[list[float]]) -> dict:
    response = requests.post(
        f"{url.rstrip('/')}/api/v1/predict/",
        json={"features": features_batch},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def run_demo(
    patient_id: str,
    disorder: str,
    n_epochs: int,
    delay: float,
    api_url: str,
    batch_size: int,
) -> None:
    profile = DISORDER_PROFILES[disorder]
    logger.info("Starting single-patient IoT demo")
    logger.info(f"Patient={patient_id} | profile={disorder} | api={api_url}")

    try:
        health = requests.get(f"{api_url.rstrip('/')}/api/v1/health/", timeout=5)
        health.raise_for_status()
        logger.info("API health check: OK")
    except Exception as exc:
        logger.error(f"API is not available: {exc}")
        return

    all_predictions: list[str] = []
    batch_buffer: list[list[float]] = []
    batch_start_idx = 0

    for epoch_idx in range(n_epochs):
        signal = generate_eeg_epoch(profile)
        batch_buffer.append(extract_feature_vector(signal, sfreq=SFREQ))

        should_flush = len(batch_buffer) >= batch_size or epoch_idx == n_epochs - 1
        if should_flush:
            timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
            try:
                result = predict_via_api(api_url, batch_buffer)
                for offset, pred in enumerate(result.get("predictions", [])):
                    current_epoch = batch_start_idx + offset
                    all_predictions.append(pred)
                    logger.info(
                        f"[{timestamp}] epoch {current_epoch + 1:03d}/{n_epochs} "
                        f"patient={patient_id} predicted={pred} "
                        f"cached={result.get('cached', False)}"
                    )
            except Exception as exc:
                logger.error(f"[{timestamp}] API error: {exc}")
            batch_start_idx += len(batch_buffer)
            batch_buffer = []

        time.sleep(delay)

    if all_predictions:
        counts = Counter(all_predictions)
        dominant = counts.most_common(1)[0][0]
        logger.info(f"Summary for {patient_id}: dominant={dominant}")
        for class_name, count in counts.most_common():
            pct = count / len(all_predictions) * 100
            logger.info(f"  {class_name:<15} {count:4d} ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT Sleep Disorder Demo")
    parser.add_argument("--patient-id", default="demo_patient")
    parser.add_argument("--disorder", default="healthy", choices=list(DISORDER_PROFILES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--batch-size", type=int, default=5)
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
