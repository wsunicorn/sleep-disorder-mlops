"""Build and upload the reference feature dataset used by drift monitoring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering.cap_features import NOTEBOOK_MAX_PER_CLASS, load_balanced_cap_dataset


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key, got: {uri}")
    return bucket, key


def upload_parquet(local_path: Path, s3_uri: str) -> None:
    import boto3

    bucket, key = _parse_s3_uri(s3_uri)
    boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION")).upload_file(
        str(local_path),
        bucket,
        key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reference CAP features")
    parser.add_argument("--data-dir", default="data/raw/balanced_CAP")
    parser.add_argument("--output", default="data/features/reference/features.parquet")
    parser.add_argument("--s3-uri", default=os.getenv("DRIFT_REFERENCE_DATA", ""))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    df = load_balanced_cap_dataset(
        args.data_dir,
        max_per_class=NOTEBOOK_MAX_PER_CLASS,
        synthetic_if_missing=args.allow_synthetic,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} reference rows to {output}")

    if args.s3_uri:
        upload_parquet(output, args.s3_uri)
        print(f"Uploaded reference data to {args.s3_uri}")


if __name__ == "__main__":
    main()
