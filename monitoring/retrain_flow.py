"""
Monitoring — Prefect retraining flow.
Tự động retrain model khi phát hiện drift hoặc F1 drop.
"""

import os
import subprocess
import re
from pathlib import Path
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv
from monitoring.promote_rules import evaluate_run_and_promote

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

    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

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
    features_path = str(features_dir)
    return features_path


@task(name="run-training", retries=1)
def run_training(features_path: str) -> str:
    """Chạy training và trả về MLflow run_id."""
    pf_logger = get_run_logger()
    pf_logger.info(f"Starting training with data: {features_path}")

    result = subprocess.run(
        [
            "python", "training/train.py",
            "--data-dir", features_path,
            "--model-type", "xgboost",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")

    pf_logger.info("Training completed.")
    # Lấy run_id từ output logger của training script.
    match = re.search(r"Run ID:\s*([a-f0-9]{32})", result.stdout)
    if match:
        return match.group(1)

    return "unknown"


@task(name="evaluate-and-promote")
def evaluate_and_promote(run_id: str) -> bool:
    """Kiểm tra F1 của model mới, nếu đạt → promote lên Staging."""
    pf_logger = get_run_logger()

    if run_id == "unknown":
        pf_logger.warning("Could not determine run_id. Skipping promotion.")
        return False

    promoted, details = evaluate_run_and_promote(
        run_id=run_id,
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MLFLOW_MODEL_NAME,
        threshold=F1_PROMOTE_THRESHOLD,
        stage="Staging",
    )
    pf_logger.info(details)
    return promoted


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
