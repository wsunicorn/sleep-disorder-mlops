"""Download model serving artifacts from object storage when configured."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


ARTIFACT_FILES = [
    "model.pkl",
    "model.ubj",
    "label_encoder.pkl",
    "feature_names.json",
    "metadata.json",
]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in S3 URI: {uri}")
    return bucket, key.strip("/")


def sync_model_artifacts_once(
    *,
    artifact_uri: str | None,
    local_dir: str | Path,
    aws_region: str | None = None,
) -> dict[str, Any]:
    """
    Sync serving artifacts from S3 into the local model directory.

    The function is intentionally best-effort. If S3 is unavailable the app can
    still fall back to artifacts baked into the Docker image.
    """
    if not artifact_uri:
        return {"enabled": False, "downloaded": [], "error": None}
    if not artifact_uri.startswith("s3://"):
        return {
            "enabled": False,
            "downloaded": [],
            "error": f"Unsupported artifact URI: {artifact_uri}",
        }

    try:
        import boto3
        from botocore.exceptions import ClientError
    except Exception as exc:
        return {"enabled": True, "downloaded": [], "error": str(exc)}

    output_dir = Path(local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket, prefix = _parse_s3_uri(artifact_uri)
    s3 = boto3.client("s3", region_name=aws_region)
    downloaded: list[str] = []

    for filename in ARTIFACT_FILES:
        key = "/".join(part for part in [prefix, filename] if part)
        target = output_dir / filename
        try:
            s3.download_file(bucket, key, str(target))
            downloaded.append(filename)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                continue
            logger.warning(f"Could not download s3://{bucket}/{key}: {exc}")
        except Exception as exc:
            logger.warning(f"Could not download s3://{bucket}/{key}: {exc}")

    marker = {
        "artifact_uri": artifact_uri,
        "downloaded": downloaded,
        "synced_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (output_dir / ".artifact_sync.json").write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )
    return {"enabled": True, "downloaded": downloaded, "error": None}
