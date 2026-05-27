"""Utilities for storing ingested feature batches for monitoring/retraining."""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


META_COLUMNS = {
    "patient_id",
    "epoch_index",
    "timestamp",
    "label",
    "disease",
    "diagnosis",
    "predicted_class",
    "confidence",
    "ingested_at",
    "training_approved",
    "label_verified",
    "ground_truth_verified",
    "verified_for_training",
}
VERIFIED_LABEL_COLUMNS = (
    "training_approved",
    "label_verified",
    "ground_truth_verified",
    "verified_for_training",
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in S3 URI: {uri}")
    return bucket, key.strip("/")


def _coerce_feature_row(
    epoch: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, float] | None:
    raw_features = epoch.get("features")
    if raw_features is None:
        return None
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        return None
    if len(raw_features) != len(feature_names):
        return None

    row: dict[str, float] = {}
    for name, value in zip(feature_names, raw_features):
        row[name] = float(value)
    return row


def build_ingest_feature_frame(
    *,
    patient_id: str,
    diagnosis: str,
    epochs: Iterable[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Convert `/ingest/` epoch payloads into a notebook-schema feature frame."""
    ingested_at = datetime.now(tz=timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for epoch in epochs:
        feature_row = _coerce_feature_row(epoch, feature_names)
        if feature_row is None:
            continue

        label = str(epoch.get("label") or epoch.get("disease") or diagnosis)
        feature_row.update(
            {
                "patient_id": patient_id,
                "epoch_index": int(epoch.get("epoch_index", len(rows))),
                "timestamp": str(epoch.get("timestamp") or ingested_at),
                "label": label,
                "disease": label,
                "diagnosis": diagnosis,
                "predicted_class": str(epoch.get("predicted_class", "")),
                "confidence": epoch.get("confidence"),
                "ingested_at": ingested_at,
            }
        )
        for column in VERIFIED_LABEL_COLUMNS:
            if column in epoch:
                feature_row[column] = epoch.get(column)
        rows.append(feature_row)

    return pd.DataFrame(rows)


def write_ingest_feature_batch(
    *,
    patient_id: str,
    diagnosis: str,
    epochs: Iterable[Mapping[str, Any]],
    feature_names: Sequence[str],
    local_dir: str | Path,
    s3_uri: str | None = None,
    aws_region: str | None = None,
) -> dict[str, Any]:
    """
    Persist one ingested feature batch as Parquet.

    Local files make development/testing observable; S3 files make the same
    batches available to GitHub Actions monitoring and retraining.
    """
    df = build_ingest_feature_frame(
        patient_id=patient_id,
        diagnosis=diagnosis,
        epochs=epochs,
        feature_names=feature_names,
    )
    if df.empty:
        return {"rows": 0, "local_path": None, "s3_uri": None}

    now = datetime.now(tz=timezone.utc)
    partition = now.strftime("date=%Y-%m-%d")
    filename = f"features_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{_safe_name(patient_id)}.parquet"

    output_dir = Path(local_dir) / partition
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / filename
    df.to_parquet(local_path, index=False)

    uploaded_uri = None
    if s3_uri:
        try:
            import boto3

            bucket, prefix = _parse_s3_uri(s3_uri)
            key = "/".join(part for part in [prefix, partition, filename] if part)
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            boto3.client("s3", region_name=aws_region).upload_fileobj(
                buffer,
                bucket,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            uploaded_uri = f"s3://{bucket}/{key}"
        except Exception as exc:
            logger.warning(f"Could not upload ingest feature batch to {s3_uri}: {exc}")

    return {
        "rows": int(len(df)),
        "local_path": str(local_path),
        "s3_uri": uploaded_uri,
    }
