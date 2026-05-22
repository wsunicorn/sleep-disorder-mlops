"""Build and upload the reference feature dataset used by drift monitoring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from feature_engineering.cap_features import (
    DISEASE_FILES,
    FEATURE_NAMES,
    NOTEBOOK_MAX_PER_CLASS,
    load_balanced_cap_dataset,
    load_feature_stats,
    sample_feature_vector,
)


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


def build_from_feature_stats(
    stats_path: str | Path,
    samples_per_class: int,
    random_seed: int,
) -> pd.DataFrame:
    """Create a lightweight 7-class reference set from notebook-derived stats."""
    stats = load_feature_stats(stats_path)
    rng = np.random.default_rng(random_seed)
    rows = []
    for disease in DISEASE_FILES:
        if disease not in stats:
            raise KeyError(f"No statistics for class '{disease}' in {stats_path}")
        for _ in range(samples_per_class):
            values = sample_feature_vector(stats, disease, rng)
            rows.append(dict(zip(FEATURE_NAMES, values)) | {"disease": disease})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reference CAP features")
    parser.add_argument("--data-dir", default="data/raw/balanced_CAP")
    parser.add_argument("--output", default="data/features/reference/features.parquet")
    parser.add_argument("--s3-uri", default=os.getenv("DRIFT_REFERENCE_DATA", ""))
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--stats-path",
        default="",
        help="Optional feature_stats.json path for a fast 24-feature reference export.",
    )
    parser.add_argument(
        "--stats-samples-per-class",
        type=int,
        default=1000,
        help="Rows per class when --stats-path is provided.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if args.stats_path:
        df = build_from_feature_stats(
            args.stats_path,
            samples_per_class=args.stats_samples_per_class,
            random_seed=args.random_seed,
        )
    else:
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
