"""Generate richer IoT demo payloads for the Sleep Disorder MLOps web app.

The generated files are intentionally deterministic so a demo can be repeated:
- multiple patients across all 7 CAP labels
- many epochs per patient
- mixed cases for timeline charts
- CSV batch for the prediction UI
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


FEATURE_NAMES = [
    "delta_power",
    "delta_rel",
    "theta_power",
    "theta_rel",
    "alpha_power",
    "alpha_rel",
    "beta_power",
    "beta_rel",
    "gamma_power",
    "gamma_rel",
    "spectral_entropy",
    "peak_frequency",
    "mean_frequency",
    "amplitude_mean",
    "amplitude_std",
    "rms",
    "delta_beta_ratio",
    "theta_alpha_ratio",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
]

LABELS = ["healthy", "insomnia", "narcolepsy", "nfle", "plm", "rbd", "sdb"]
RELATIVE_INDICES = [1, 3, 5, 7, 9]
NON_NEGATIVE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23]

AGE_RANGES = {
    "healthy": (22, 45),
    "insomnia": (32, 62),
    "narcolepsy": (18, 38),
    "nfle": (18, 45),
    "plm": (40, 72),
    "rbd": (48, 76),
    "sdb": (42, 74),
    "monitoring_case": (25, 65),
}

CONFIDENCE_RANGES = {
    "healthy": (0.78, 0.96),
    "insomnia": (0.68, 0.91),
    "narcolepsy": (0.62, 0.88),
    "nfle": (0.61, 0.86),
    "plm": (0.6, 0.85),
    "rbd": (0.59, 0.84),
    "sdb": (0.64, 0.9),
}


def load_stats(path: Path) -> dict[str, dict[str, list[float]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def round_feature(value: float) -> float:
    return round(float(value), 6)


def sample_feature(stats: dict[str, dict[str, list[float]]], label: str, rng: random.Random) -> list[float]:
    mean = stats[label]["mean"]
    std = stats[label]["std"]
    values = [rng.gauss(mu, max(sigma, 1e-9) * 0.55) for mu, sigma in zip(mean, std)]

    for index in NON_NEGATIVE_INDICES:
        values[index] = max(values[index], 0.0)

    rel_sum = sum(values[index] for index in RELATIVE_INDICES)
    if rel_sum > 0:
        for index in RELATIVE_INDICES:
            values[index] = max(values[index], 0.0) / rel_sum

    return [round_feature(value) for value in values]


def choose_prediction(primary_label: str, feature_label: str, epoch_index: int, rng: random.Random) -> str:
    if primary_label == "monitoring_case":
        return feature_label

    # Short unstable bands make the patient detail timeline less flat.
    if epoch_index % 17 in {13, 14} and primary_label != "healthy":
        return rng.choice(["healthy", primary_label])
    if epoch_index % 23 == 7:
        return rng.choice([primary_label, feature_label])
    return primary_label if rng.random() < 0.88 else feature_label


def confidence_for(label: str, predicted: str, rng: random.Random) -> float:
    lo, hi = CONFIDENCE_RANGES.get(predicted, (0.58, 0.82))
    value = rng.uniform(lo, hi)
    if predicted != label and label != "monitoring_case":
        value -= rng.uniform(0.04, 0.12)
    return round(max(0.5, min(0.98, value)), 3)


def patient_meta(label: str, index: int, rng: random.Random, prefix: str) -> dict[str, Any]:
    lo, hi = AGE_RANGES.get(label, AGE_RANGES["monitoring_case"])
    return {
        "patient_id": f"{prefix}-{label.replace('_', '-')}-{index:02d}",
        "disorder": label,
        "age": rng.randint(lo, hi),
        "gender": rng.choice(["M", "F"]),
    }


def feature_label_for_patient(primary_label: str, epoch_index: int, rng: random.Random) -> str:
    if primary_label == "monitoring_case":
        pattern = ["healthy", "insomnia", "narcolepsy", "nfle", "sdb", "rbd", "plm"]
        return pattern[(epoch_index // 6) % len(pattern)]
    if primary_label == "healthy":
        return "healthy" if rng.random() < 0.94 else rng.choice(["insomnia", "sdb"])
    return primary_label if rng.random() < 0.9 else rng.choice(["healthy", primary_label])


def build_patient_payload(
    stats: dict[str, dict[str, list[float]]],
    *,
    patient: dict[str, Any],
    epochs: int,
    start_time: datetime,
    rng: random.Random,
) -> dict[str, Any]:
    records = []
    primary_label = patient["disorder"]
    for epoch_index in range(epochs):
        feature_label = feature_label_for_patient(primary_label, epoch_index, rng)
        predicted = choose_prediction(primary_label, feature_label, epoch_index, rng)
        records.append(
            {
                "epoch_index": epoch_index,
                "predicted_class": predicted,
                "confidence": confidence_for(primary_label, predicted, rng),
                "timestamp": (start_time + timedelta(seconds=epoch_index * 2)).isoformat().replace("+00:00", "Z"),
                "label": feature_label,
                "features": sample_feature(stats, feature_label, rng),
            }
        )

    return {
        "patient_id": patient["patient_id"],
        "disorder": patient["disorder"],
        "age": patient["age"],
        "gender": patient["gender"],
        "epochs": records,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_prediction_csv(
    path: Path,
    stats: dict[str, dict[str, list[float]]],
    rows_per_label: int,
    rng: random.Random,
) -> int:
    total = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FEATURE_NAMES)
        for label in LABELS:
            for _ in range(rows_per_label):
                writer.writerow(sample_feature(stats, label, rng))
                total += 1
    return total


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    stats_path = Path(args.stats)
    if not stats_path.is_absolute():
        stats_path = root / stats_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("ingest_*.json"):
        old_file.unlink()

    rng = random.Random(args.seed)
    stats = load_stats(stats_path)
    start = datetime(2026, 5, 23, 21, 30, tzinfo=timezone.utc)

    patient_files: list[str] = []
    total_epochs = 0
    patient_number = 1
    for label in LABELS:
        for per_class_index in range(1, args.patients_per_class + 1):
            patient = patient_meta(label, patient_number, rng, args.prefix)
            patient["patient_id"] = f"{args.prefix}-{label}-{per_class_index:02d}"
            payload = build_patient_payload(
                stats,
                patient=patient,
                epochs=args.epochs_per_patient,
                start_time=start + timedelta(minutes=patient_number * 3),
                rng=rng,
            )
            filename = f"ingest_{patient['patient_id']}.json"
            write_json(output_dir / filename, payload)
            patient_files.append(filename)
            total_epochs += len(payload["epochs"])
            patient_number += 1

    for mixed_index in range(1, args.mixed_patients + 1):
        patient = patient_meta("monitoring_case", mixed_index, rng, args.prefix)
        patient["patient_id"] = f"{args.prefix}-mixed-{mixed_index:02d}"
        payload = build_patient_payload(
            stats,
            patient=patient,
            epochs=args.epochs_per_patient,
            start_time=start + timedelta(minutes=patient_number * 3),
            rng=rng,
        )
        filename = f"ingest_{patient['patient_id']}.json"
        write_json(output_dir / filename, payload)
        patient_files.append(filename)
        total_epochs += len(payload["epochs"])
        patient_number += 1

    csv_rows = write_prediction_csv(
        output_dir / "predict_batch_rich.csv",
        stats,
        rows_per_label=args.csv_rows_per_label,
        rng=rng,
    )

    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": args.seed,
        "stats_path": str(stats_path),
        "patients": len(patient_files),
        "epochs_per_patient": args.epochs_per_patient,
        "total_epochs": total_epochs,
        "csv_rows": csv_rows,
        "patient_files": patient_files,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rich IoT demo files.")
    parser.add_argument("--stats", default="data/raw/balanced_CAP/feature_stats.json")
    parser.add_argument("--output-dir", default="demo_web_iot/generated")
    parser.add_argument("--patients-per-class", type=int, default=3)
    parser.add_argument("--mixed-patients", type=int, default=3)
    parser.add_argument("--epochs-per-patient", type=int, default=48)
    parser.add_argument("--csv-rows-per-label", type=int, default=8)
    parser.add_argument("--prefix", default="demo-rich")
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    manifest = generate(args)
    print(
        "Generated "
        f"{manifest['patients']} patients, "
        f"{manifest['total_epochs']} epochs, "
        f"{manifest['csv_rows']} CSV rows."
    )
    print(f"Output: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
