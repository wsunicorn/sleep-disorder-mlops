"""
Monitoring — Prefect retraining flow.
Tự động retrain model khi phát hiện drift hoặc F1 drop.
"""

import os
import subprocess
from datetime import datetime
from loguru import logger
from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "sleep-mlops-data")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sleep-disorder-classifier")
F1_PROMOTE_THRESHOLD = 0.80  # Model mới phải đạt F1 ≥ 0.80 để promote


@task(name="check-drift-threshold", retries=2)
def check_drift_threshold(drift_share: float, f1_current: float) -> bool:
    """Kiểm tra xem có cần retrain không."""
    pf_logger = get_run_logger()
    needs_retrain = drift_share > 0.3 or f1_current < F1_PROMOTE_THRESHOLD
    pf_logger.info(
        f"Drift share: {drift_share:.2%} | F1: {f1_current:.4f} | "
        f"Retrain needed: {needs_retrain}"
    )
    return needs_retrain


@task(name="run-feature-engineering", retries=1)
def run_feature_engineering() -> str:
    """Chạy lại feature extraction với data mới nhất."""
    pf_logger = get_run_logger()
    pf_logger.info("Running feature engineering on new data...")

    result = subprocess.run(
        [
            "python", "feature_engineering/extract_features.py",
            "--input-dir", "data/processed",
            "--output-dir", "data/features",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Feature engineering failed:\n{result.stderr}")

    pf_logger.info("Feature engineering completed.")
    features_path = f"s3://{S3_BUCKET}/features/features.parquet"
    return features_path


@task(name="run-training", retries=1)
def run_training(features_path: str) -> str:
    """Chạy training và trả về MLflow run_id."""
    pf_logger = get_run_logger()
    pf_logger.info(f"Starting training with data: {features_path}")

    result = subprocess.run(
        [
            "python", "training/train.py",
            "--data-dir", f"s3://{S3_BUCKET}/features/",
            "--model-type", "xgboost",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")

    pf_logger.info("Training completed.")
    # Lấy run_id từ output
    for line in result.stdout.split("\n"):
        if "Run ID:" in line:
            run_id = line.split("Run ID:")[-1].strip()
            return run_id

    return "unknown"


@task(name="evaluate-and-promote")
def evaluate_and_promote(run_id: str) -> bool:
    """Kiểm tra F1 của model mới, nếu đạt → promote lên Staging."""
    pf_logger = get_run_logger()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if run_id == "unknown":
        pf_logger.warning("Could not determine run_id. Skipping promotion.")
        return False

    run = client.get_run(run_id)
    f1 = run.data.metrics.get("val_f1_weighted", 0.0)
    pf_logger.info(f"New model F1: {f1:.4f} (threshold: {F1_PROMOTE_THRESHOLD})")

    if f1 >= F1_PROMOTE_THRESHOLD:
        # Tìm version mới nhất của model
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if versions:
            version = versions[0].version
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=version,
                stage="Staging",
            )
            pf_logger.info(f"Model v{version} promoted to Staging (F1={f1:.4f})")
            return True
    else:
        pf_logger.warning(f"Model F1={f1:.4f} below threshold. Not promoting.")
        return False


@flow(name="sleep-disorder-retrain-pipeline", log_prints=True)
def retrain_pipeline(drift_share: float = 0.0, f1_current: float = 1.0):
    """
    Main Prefect flow: kiểm tra điều kiện → feature engineering → train → evaluate.
    """
    needs_retrain = check_drift_threshold(drift_share, f1_current)

    if not needs_retrain:
        print("No retraining needed. Exiting.")
        return

    features_path = run_feature_engineering()
    run_id = run_training(features_path)
    promoted = evaluate_and_promote(run_id)

    if promoted:
        print(f"Retraining complete. New model in Staging. Run ID: {run_id}")
    else:
        print("Retraining complete but model did not meet promotion threshold.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-share", type=float, default=0.35)
    parser.add_argument("--f1-current", type=float, default=0.75)
    args = parser.parse_args()

    retrain_pipeline(drift_share=args.drift_share, f1_current=args.f1_current)
