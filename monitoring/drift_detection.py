"""
Monitoring — Evidently AI data drift detection.
So sánh phân phối features mới với baseline, phát hiện data drift.
"""

import os
import argparse
import pandas as pd
import boto3
from pathlib import Path
from datetime import datetime
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "sleep-mlops-data")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
DRIFT_THRESHOLD = 0.3  # Tỉ lệ features drift > 30% → cảnh báo


def load_parquet_from_s3_or_local(path: str) -> pd.DataFrame:
    """Load parquet từ S3 URI hoặc local path."""
    if path.startswith("s3://"):
        import io
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket, key = path[5:].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    return pd.read_parquet(path)


def run_drift_detection(reference_path: str, current_path: str, output_dir: str) -> dict:
    """
    Chạy Evidently drift detection.
    Returns dict với thông tin drift.
    """
    logger.info(f"Loading reference data: {reference_path}")
    reference_df = load_parquet_from_s3_or_local(reference_path)

    logger.info(f"Loading current data: {current_path}")
    current_df = load_parquet_from_s3_or_local(current_path)

    # Chỉ lấy feature columns (bỏ metadata)
    meta_cols = ["epoch_index", "subject_id", "label"]
    feature_cols = [c for c in reference_df.columns if c not in meta_cols]

    ref_features = reference_df[feature_cols]
    cur_features = current_df[feature_cols]

    logger.info(f"Reference: {len(ref_features)} rows | Current: {len(cur_features)} rows")
    logger.info(f"Features: {len(feature_cols)}")

    # Tạo Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=ref_features, current_data=cur_features)

    # Lưu HTML report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = output_dir / report_name
    report.save_html(str(report_path))
    logger.info(f"Report saved: {report_path}")

    # Lấy kết quả drift
    result = report.as_dict()
    dataset_drift = result["metrics"][2]["result"]
    drift_detected = dataset_drift.get("dataset_drift", False)
    drift_share = dataset_drift.get("share_of_drifted_columns", 0.0)

    logger.info(f"Drift detected: {drift_detected}")
    logger.info(f"Share of drifted features: {drift_share:.2%}")

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "report_path": str(report_path),
        "n_reference": len(ref_features),
        "n_current": len(cur_features),
        "alert": drift_share > DRIFT_THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser(description="Evidently Drift Detection")
    parser.add_argument("--reference-data", required=True, help="Path to reference parquet")
    parser.add_argument("--current-data", required=True, help="Path to current parquet")
    parser.add_argument("--output-report", default="reports/", help="Output directory")
    args = parser.parse_args()

    result = run_drift_detection(
        args.reference_data,
        args.current_data,
        args.output_report,
    )

    if result["alert"]:
        logger.warning(
            f"ALERT: Data drift exceeds threshold! "
            f"Drift share = {result['drift_share']:.2%} > {DRIFT_THRESHOLD:.2%}"
        )

    return result


if __name__ == "__main__":
    main()
