"""IoT REST simulator for the Sleep Disorder MLOps web app.

The current production path receives IoT-like batches through:
    POST /api/v1/ingest/

This simulator reads an EDF recording, extracts the same 24 EEG features used by
the notebook/API, asks the production inference endpoint for labels, then sends
patient + epoch payloads back to the ingest endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
from dotenv import load_dotenv
from loguru import logger


def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "feature_engineering" / "cap_features.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from feature_engineering.cap_features import (  # noqa: E402
    WINDOW_SEC,
    extract_feature_matrix,
)


load_dotenv()

DEFAULT_API_BASE = os.getenv("SLEEP_PORTAL_API_BASE", "http://127.0.0.1:8000")
DEFAULT_DELAY_SEC = float(os.getenv("IOT_SIMULATION_DELAY_SEC", "0.25"))
DEFAULT_BATCH_SIZE = int(os.getenv("IOT_SIMULATION_BATCH_SIZE", "8"))


def _post_json(api_base: str, endpoint: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    url = api_base.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{endpoint} failed with HTTP {exc.code}: {body}") from exc


def read_edf_feature_rows(edf_path: str, max_epochs: int | None = None) -> list[list[float]]:
    """Read EDF, pick the first EEG channel, and extract notebook-standard features."""
    logger.info(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(l_freq=0.5, h_freq=40.0, method="fir", verbose=False)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_idx = int(eeg_picks[0]) if len(eeg_picks) else 0
    channel_name = raw.ch_names[ch_idx]
    signal, _ = raw[[ch_idx], :]
    signal = signal[0]

    sfreq = float(raw.info["sfreq"])
    window_samples = int(WINDOW_SEC * sfreq)
    features = extract_feature_matrix(signal, sfreq=sfreq, window_samples=window_samples)
    if max_epochs is not None:
        features = features[:max_epochs]

    logger.info(
        f"Extracted {len(features)} epochs | channel={channel_name} | "
        f"sfreq={sfreq:g} Hz | window={WINDOW_SEC:g}s"
    )
    return features.astype(float).tolist()


def predict_epoch_classes(api_base: str, feature_rows: list[list[float]]) -> list[str]:
    """Call the serving API so the simulated IoT stream uses the deployed model."""
    response = _post_json(api_base, "/api/v1/predict/", {"features": feature_rows})
    predictions = response.get("predictions")
    if isinstance(predictions, list) and len(predictions) == len(feature_rows):
        return [str(item) for item in predictions]
    predicted_class = str(response.get("predicted_class", "unknown"))
    return [predicted_class for _ in feature_rows]


def iter_batches(items: list[list[float]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def replay_edf_to_ingest(
    *,
    edf_path: str,
    api_base: str,
    patient_id: str,
    diagnosis: str,
    age: int | None,
    gender: str | None,
    max_epochs: int | None,
    batch_size: int,
    delay: float,
    confidence: float,
) -> None:
    feature_rows = read_edf_feature_rows(edf_path, max_epochs=max_epochs)
    if not feature_rows:
        raise RuntimeError("No feature rows were extracted from the EDF file.")

    total_saved = 0
    for offset, batch_features in iter_batches(feature_rows, batch_size):
        predictions = predict_epoch_classes(api_base, batch_features)
        now = datetime.now(tz=timezone.utc)
        epochs = []
        for local_index, (features, predicted_class) in enumerate(zip(batch_features, predictions)):
            epoch_index = offset + local_index
            epochs.append(
                {
                    "epoch_index": epoch_index,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "timestamp": now.isoformat().replace("+00:00", "Z"),
                    "features": features,
                }
            )

        payload = {
            "patient_id": patient_id,
            "disorder": diagnosis,
            "age": age,
            "gender": gender,
            "epochs": epochs,
        }
        ingest_response = _post_json(api_base, "/api/v1/ingest/", payload)
        total_saved += int(ingest_response.get("epochs_saved", 0))
        logger.info(
            f"Ingested epochs {offset}-{offset + len(epochs) - 1} | "
            f"saved={ingest_response.get('epochs_saved')} | "
            f"feature_rows={ingest_response.get('feature_rows_saved')}"
        )
        if delay > 0:
            time.sleep(delay)

    logger.info(f"IoT simulation complete. Total epochs saved: {total_saved}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDF -> predict API -> ingest API simulator")
    parser.add_argument("--edf", required=True, help="Path to .edf file")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Web app base URL")
    parser.add_argument("--patient-id", required=True, help="Patient ID shown on the dashboard")
    parser.add_argument("--diagnosis", default="unknown", help="Known/demo diagnosis label")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--gender", default=None, choices=["M", "F", "O"])
    parser.add_argument("--max-epochs", type=int, default=32, help="Limit epochs for a fast demo")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC)
    parser.add_argument("--confidence", type=float, default=0.82)
    args = parser.parse_args()

    replay_edf_to_ingest(
        edf_path=args.edf,
        api_base=args.api_base,
        patient_id=args.patient_id,
        diagnosis=args.diagnosis,
        age=args.age,
        gender=args.gender,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        delay=args.delay,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
