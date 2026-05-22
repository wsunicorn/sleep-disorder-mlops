"""Multi-patient API demo using Balanced CAP feature statistics."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from loguru import logger

from feature_engineering.cap_features import (
    DISEASE_FILES,
    load_feature_stats,
    sample_feature_vector,
)

DEFAULT_URL = "http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com"
DEFAULT_STATS_PATH = Path("data/raw/balanced_CAP/feature_stats.json")

DEFAULT_PATIENTS = [
    {"patient_id": "PT-001", "disorder": "insomnia", "age": 42, "gender": "F"},
    {"patient_id": "PT-002", "disorder": "nfle", "age": 28, "gender": "M"},
    {"patient_id": "PT-003", "disorder": "healthy", "age": 35, "gender": "F"},
    {"patient_id": "PT-004", "disorder": "sdb", "age": 55, "gender": "M"},
    {"patient_id": "PT-005", "disorder": "narcolepsy", "age": 22, "gender": "M"},
]


def _predict(api_url: str, features_batch: list[list[float]]) -> list[str]:
    response = requests.post(
        f"{api_url.rstrip('/')}/api/v1/predict/",
        json={"features": features_batch},
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("predictions", [])


def _ingest(
    api_url: str,
    patient: dict[str, Any],
    diagnosis: str,
    epoch_records: list[dict[str, Any]],
    retries: int = 4,
) -> dict[str, Any]:
    payload = {
        "patient_id": patient["patient_id"],
        "disorder": diagnosis,
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "epochs": epoch_records,
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/api/v1/ingest/",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (2**attempt) + np.random.uniform(0, 0.3))

    return {}


def run_patient(
    patient: dict[str, Any],
    feature_stats: dict[str, dict[str, list[float]]],
    n_epochs: int,
    batch_size: int,
    delay: float,
    api_url: str,
    seed: int,
) -> dict[str, Any]:
    patient_id = patient["patient_id"]
    disorder = patient["disorder"]
    rng = np.random.default_rng(seed)
    predictions: list[str] = []
    epoch_records: list[dict[str, Any]] = []
    feature_buffer: list[list[float]] = []

    logger.info(f"[{patient_id}] start disorder={disorder}, epochs={n_epochs}")
    for epoch_index in range(n_epochs):
        feature_buffer.append(sample_feature_vector(feature_stats, disorder, rng))

        should_flush = len(feature_buffer) >= batch_size or epoch_index == n_epochs - 1
        if should_flush:
            timestamp = datetime.now(tz=timezone.utc)
            try:
                batch_predictions = _predict(api_url, feature_buffer)
                start_index = epoch_index - len(feature_buffer) + 1
                for offset, pred in enumerate(batch_predictions):
                    current_epoch = start_index + offset
                    predictions.append(pred)
                    epoch_records.append(
                        {
                            "epoch_index": current_epoch,
                            "predicted_class": pred,
                            "confidence": None,
                            "timestamp": timestamp.isoformat(),
                        }
                    )
                    logger.info(f"[{patient_id}] epoch {current_epoch + 1:03d} -> {pred}")
            except Exception as exc:
                logger.error(f"[{patient_id}] predict error: {exc}")
            feature_buffer = []

        time.sleep(delay)

    counts = Counter(predictions)
    diagnosis = counts.most_common(1)[0][0] if counts else disorder
    ingest_result: dict[str, Any] = {}
    if epoch_records:
        try:
            ingest_result = _ingest(api_url, patient, diagnosis, epoch_records)
        except Exception as exc:
            logger.error(f"[{patient_id}] ingest error: {exc}")

    logger.info(f"[{patient_id}] dominant={diagnosis}, counts={dict(counts)}")
    return {
        "patient_id": patient_id,
        "dominant": diagnosis,
        "counts": dict(counts),
        "ingest": ingest_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-patient IoT demo")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--stats", default=str(DEFAULT_STATS_PATH))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats_path = Path(args.stats)
    if not stats_path.exists():
        expected = ", ".join(DISEASE_FILES.keys())
        raise FileNotFoundError(
            f"Feature stats not found: {stats_path}. "
            f"Create it from the notebook/Balanced CAP data for classes: {expected}."
        )
    feature_stats = load_feature_stats(stats_path)

    health = requests.get(f"{args.url.rstrip('/')}/api/v1/health/", timeout=5)
    health.raise_for_status()
    logger.info(f"API OK: {args.url}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_patient,
                patient,
                feature_stats,
                args.epochs,
                args.batch_size,
                args.delay,
                args.url,
                args.seed + index,
            ): patient["patient_id"]
            for index, patient in enumerate(DEFAULT_PATIENTS)
        }
        for future in as_completed(futures):
            patient_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"[{patient_id}] worker error: {exc}")

    logger.info("Summary")
    for result in sorted(results, key=lambda item: item["patient_id"]):
        saved = result.get("ingest", {}).get("epochs_saved", "?")
        logger.info(f"{result['patient_id']} -> {result['dominant']} ({saved} epochs saved)")


if __name__ == "__main__":
    main()
