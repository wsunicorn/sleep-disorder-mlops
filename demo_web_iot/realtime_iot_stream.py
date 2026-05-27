"""Realtime-style IoT stream demo for the Sleep Disorder MLOps web app.

The demo behaves like a small bedside gateway:
- generate plausible 24-feature EEG epochs for many patients
- call the deployed /api/v1/predict/ endpoint for each small batch
- send predicted epochs to /api/v1/ingest/
- keep a local session state so repeated runs continue epoch indexes

Rows are intentionally not marked as training-approved. They are suitable for
dashboard, monitoring, and drift demos, but retraining will ignore them until a
human/clinical process verifies labels.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError

try:
    from generate_rich_iot_demo import (
        AGE_RANGES,
        FEATURE_NAMES,
        LABELS,
        load_stats,
        sample_feature,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_rich_iot_demo import (
        AGE_RANGES,
        FEATURE_NAMES,
        LABELS,
        load_stats,
        sample_feature,
    )


DEFAULT_BASE_URL = "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com"
DEFAULT_STATS_PATH = "data/raw/balanced_CAP/feature_stats.json"
DEFAULT_STATE_PATH = "demo_web_iot/runtime/realtime_state.json"
DEFAULT_SESSION_ID = "live-demo"
MIXED_CASE_TYPE = "mixed"


@dataclass(frozen=True)
class Patient:
    patient_id: str
    device_id: str
    disorder: str
    age: int
    gender: str
    case_type: str = "single"


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def iso_z(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sessions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_path(path: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return root / resolved


def build_patients(
    *,
    session_id: str,
    patients_per_class: int,
    mixed_patients: int,
    rng: random.Random,
) -> list[Patient]:
    patients: list[Patient] = []
    clean_session = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in session_id).strip("-")
    clean_session = clean_session or DEFAULT_SESSION_ID

    for label in LABELS:
        low, high = AGE_RANGES[label]
        for index in range(1, patients_per_class + 1):
            patients.append(
                Patient(
                    patient_id=f"iot-{clean_session}-{label}-{index:02d}",
                    device_id=f"gw-{clean_session}-{label[:3]}-{index:02d}",
                    disorder=label,
                    age=rng.randint(low, high),
                    gender=rng.choice(["M", "F"]),
                )
            )

    for index in range(1, mixed_patients + 1):
        primary_label = LABELS[(index - 1) % len(LABELS)]
        low, high = AGE_RANGES[primary_label]
        patients.append(
            Patient(
                patient_id=f"iot-{clean_session}-mixed-{index:02d}",
                device_id=f"gw-{clean_session}-mix-{index:02d}",
                disorder=primary_label,
                age=rng.randint(low, high),
                gender=rng.choice(["M", "F"]),
                case_type=MIXED_CASE_TYPE,
            )
        )

    return patients


def latent_label(disorder: str, epoch_index: int, rng: random.Random, *, case_type: str = "single") -> str:
    """Return the underlying feature class for one simulated epoch."""
    if case_type == MIXED_CASE_TYPE:
        # A mixed demo patient intentionally rotates through several patterns
        # while the patient diagnosis remains one of the 7 official labels.
        return LABELS[(epoch_index // 8) % len(LABELS)]

    if disorder == "healthy":
        if epoch_index % 37 in {11, 12, 13}:
            return rng.choice(["insomnia", "sdb"])
        return "healthy" if rng.random() < 0.94 else rng.choice(["insomnia", "sdb"])

    if disorder == "insomnia":
        if epoch_index % 29 in {8, 9, 10}:
            return "healthy"
        return "insomnia" if rng.random() < 0.86 else rng.choice(["healthy", "sdb"])

    if disorder == "narcolepsy":
        return "narcolepsy" if rng.random() < 0.84 else rng.choice(["healthy", "nfle"])

    if disorder == "nfle":
        return "nfle" if rng.random() < 0.88 else rng.choice(["healthy", "rbd"])

    if disorder == "plm":
        return "plm" if rng.random() < 0.87 else rng.choice(["healthy", "rbd"])

    if disorder == "rbd":
        return "rbd" if rng.random() < 0.88 else rng.choice(["healthy", "plm"])

    if disorder == "sdb":
        return "sdb" if rng.random() < 0.9 else rng.choice(["healthy", "insomnia"])

    return disorder if disorder in LABELS else rng.choice(LABELS)


def add_sensor_noise(features: list[float], *, epoch_index: int, drift_strength: float, rng: random.Random) -> list[float]:
    """Add small deterministic noise/drift while keeping the schema valid."""
    noisy = []
    slow_wave = 1.0 + drift_strength * min(epoch_index / 240.0, 1.0)
    for idx, value in enumerate(features):
        jitter = rng.uniform(-0.035, 0.035)
        adjusted = float(value) * (1.0 + jitter)
        if idx in {0, 2, 4, 6, 8, 14, 15, 21}:  # absolute powers/std/activity
            adjusted *= slow_wave
        noisy.append(round(adjusted, 6))

    relative_indices = [1, 3, 5, 7, 9]
    rel_sum = sum(max(noisy[index], 0.0) for index in relative_indices)
    if rel_sum > 0:
        for index in relative_indices:
            noisy[index] = round(max(noisy[index], 0.0) / rel_sum, 6)

    non_negative = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23}
    for index in non_negative:
        noisy[index] = max(noisy[index], 0.0)
    return noisy


def confidence_for(predicted: str, latent: str, rng: random.Random) -> float:
    if predicted == latent:
        return round(rng.uniform(0.72, 0.93), 3)
    return round(rng.uniform(0.52, 0.74), 3)


def post_json(
    session: requests.Session,
    *,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except HTTPError as exc:
            response = exc.response
            last_error = exc
            if response is not None and response.status_code == 429 and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                delay = max(delay, 2.0 * (2 ** (attempt - 1)))
                print(f"  throttle 429 on {path}; retrying in {delay:.1f}s ({attempt}/{retries})")
                time.sleep(delay)
                continue
            if attempt == retries:
                break
            time.sleep(1.0 * (2 ** (attempt - 1)))
        except Exception as exc:  # requests may wrap JSON/HTTP/socket errors.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(1.0 * (2 ** (attempt - 1)))
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last_error}")


def simulate_patient_cycle(
    *,
    patient: Patient,
    stats: dict[str, dict[str, list[float]]],
    base_url: str,
    start_epoch: int,
    epochs_per_cycle: int,
    started_at: datetime,
    drift_strength: float,
    dry_run: bool,
    timeout: int,
    retries: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    features_batch: list[list[float]] = []
    latent_labels: list[str] = []

    for offset in range(epochs_per_cycle):
        epoch_index = start_epoch + offset
        label = latent_label(patient.disorder, epoch_index, rng, case_type=patient.case_type)
        features = sample_feature(stats, label, rng)
        features = add_sensor_noise(features, epoch_index=epoch_index, drift_strength=drift_strength, rng=rng)
        features_batch.append(features)
        latent_labels.append(label)

    with requests.Session() as session:
        if dry_run:
            predictions = latent_labels[:]
        else:
            prediction_result = post_json(
                session,
                base_url=base_url,
                path="/api/v1/predict/",
                payload={"features": features_batch},
                timeout=timeout,
                retries=retries,
            )
            predictions = list(prediction_result.get("predictions") or [])
            if len(predictions) != len(features_batch):
                raise RuntimeError(
                    f"Predict returned {len(predictions)} predictions for {len(features_batch)} epochs"
                )

        epoch_records: list[dict[str, Any]] = []
        for offset, (features, latent, predicted) in enumerate(zip(features_batch, latent_labels, predictions)):
            epoch_index = start_epoch + offset
            timestamp = started_at + timedelta(seconds=epoch_index * 2)
            epoch_records.append(
                {
                    "epoch_index": epoch_index,
                    "predicted_class": str(predicted),
                    "confidence": confidence_for(str(predicted), latent, rng),
                    "timestamp": iso_z(timestamp),
                    "label": latent,
                    "features": features,
                    "device_id": patient.device_id,
                    "sampling_rate": 512,
                    "window_seconds": 2,
                }
            )

        ingest_result: dict[str, Any] = {
            "patient_id": patient.patient_id,
            "epochs_saved": len(epoch_records),
            "feature_rows_saved": len(epoch_records),
            "dry_run": True,
        }
        if not dry_run:
            ingest_result = post_json(
                session,
                base_url=base_url,
                path="/api/v1/ingest/",
                payload={
                    "patient_id": patient.patient_id,
                    "disorder": patient.disorder,
                    "age": patient.age,
                    "gender": patient.gender,
                    "epochs": epoch_records,
                },
                timeout=timeout,
                retries=retries,
            )

    return {
        "patient_id": patient.patient_id,
        "device_id": patient.device_id,
        "start_epoch": start_epoch,
        "next_epoch": start_epoch + epochs_per_cycle,
        "latent_counts": dict(Counter(latent_labels)),
        "prediction_counts": dict(Counter(predictions)),
        "ingest": ingest_result,
    }


def resolve_session_state(state: dict[str, Any], session_id: str, reset: bool) -> dict[str, Any]:
    sessions = state.setdefault("sessions", {})
    if reset or session_id not in sessions:
        sessions[session_id] = {
            "created_at": iso_z(utc_now()),
            "patients": {},
        }
    sessions[session_id]["last_started_at"] = iso_z(utc_now())
    return sessions[session_id]


def check_api(base_url: str, timeout: int) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/api/v1/health/", timeout=timeout)
    response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime IoT stream demo")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH)
    parser.add_argument("--state-file", default=DEFAULT_STATE_PATH)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--patients-per-class", type=int, default=1)
    parser.add_argument("--mixed-patients", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=6, help="0 means run until Ctrl+C.")
    parser.add_argument("--epochs-per-cycle", type=int, default=4)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--drift-strength", type=float, default=0.12)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--reset-session", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    stats_path = normalize_path(args.stats)
    state_path = normalize_path(args.state_file)

    if args.patients_per_class < 0 or args.mixed_patients < 0:
        raise ValueError("patients-per-class and mixed-patients must be >= 0")
    if args.patients_per_class == 0 and args.mixed_patients == 0:
        raise ValueError("At least one patient is required")
    if args.epochs_per_cycle <= 0:
        raise ValueError("epochs-per-cycle must be > 0")

    stats = load_stats(stats_path)
    rng = random.Random(args.seed)
    patients = build_patients(
        session_id=args.session_id,
        patients_per_class=args.patients_per_class,
        mixed_patients=args.mixed_patients,
        rng=rng,
    )

    if args.check_api and not args.dry_run:
        check_api(base_url, args.timeout)

    state = read_state(state_path)
    session_state = resolve_session_state(state, args.session_id, args.reset_session)
    patient_state = session_state.setdefault("patients", {})
    started_at_raw = session_state.get("stream_started_at")
    if args.reset_session or not started_at_raw:
        started_at = utc_now()
        session_state["stream_started_at"] = iso_z(started_at)
    else:
        started_at = datetime.fromisoformat(str(started_at_raw).replace("Z", "+00:00"))

    print("")
    print("Realtime IoT stream demo")
    print(f"Base URL     : {base_url}")
    print(f"Session      : {args.session_id}")
    print(f"Patients     : {len(patients)}")
    print(f"Epoch/cycle  : {args.epochs_per_cycle}")
    print(f"State file   : {state_path}")
    print(f"Dry run      : {args.dry_run}")
    print("")

    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            cycle_started = time.perf_counter()
            print(f"Cycle {cycle:03d} | {iso_z(utc_now())}")

            futures = {}
            cycle_failures = 0
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for index, patient in enumerate(patients):
                    next_epoch = int(patient_state.get(patient.patient_id, {}).get("next_epoch", 0))
                    future = pool.submit(
                        simulate_patient_cycle,
                        patient=patient,
                        stats=stats,
                        base_url=base_url,
                        start_epoch=next_epoch,
                        epochs_per_cycle=args.epochs_per_cycle,
                        started_at=started_at,
                        drift_strength=max(0.0, args.drift_strength),
                        dry_run=args.dry_run,
                        timeout=args.timeout,
                        retries=args.retries,
                        seed=args.seed + cycle * 10_000 + index,
                    )
                    futures[future] = patient

                for future in as_completed(futures):
                    patient = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        cycle_failures += 1
                        print(f"  {patient.patient_id:<34} FAILED {exc}")
                        continue
                    patient_state[patient.patient_id] = {
                        "device_id": patient.device_id,
                        "disorder": patient.disorder,
                        "case_type": patient.case_type,
                        "next_epoch": result["next_epoch"],
                        "last_prediction_counts": result["prediction_counts"],
                        "updated_at": iso_z(utc_now()),
                    }
                    saved = result["ingest"].get("epochs_saved", "?")
                    features_saved = result["ingest"].get("feature_rows_saved", "?")
                    print(
                        f"  {patient.patient_id:<34} "
                        f"epoch {result['start_epoch']:04d}->{result['next_epoch'] - 1:04d} "
                        f"saved={saved} features={features_saved} "
                        f"pred={result['prediction_counts']}"
                    )

            session_state["last_completed_at"] = iso_z(utc_now())
            session_state["last_cycle_failures"] = cycle_failures
            write_state(state_path, state)

            elapsed = time.perf_counter() - cycle_started
            if cycle_failures:
                print(f"  cycle failures: {cycle_failures}; successful patients will continue next cycle.")
            if args.cycles == 0 or cycle < args.cycles:
                time.sleep(max(0.0, args.interval - elapsed))
    except KeyboardInterrupt:
        print("")
        print("Stopped by user. Session state has been kept for the next run.")
        write_state(state_path, state)

    print("")
    print("Open these pages:")
    print(f"{base_url}/")
    print(f"{base_url}/patients/")
    page_patient_ids = [patients[0].patient_id]
    mixed_patient = next((patient.patient_id for patient in patients if patient.case_type == MIXED_CASE_TYPE), None)
    if mixed_patient and mixed_patient not in page_patient_ids:
        page_patient_ids.append(mixed_patient)
    for patient_id in page_patient_ids:
        print(f"{base_url}/patients/{patient_id}/")
    print(f"{base_url}/pipeline/")


if __name__ == "__main__":
    main()
