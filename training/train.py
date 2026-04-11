"""
Training — Script huấn luyện model (tương thích AWS SageMaker).
Đọc feature parquet, train XGBoost/LSTM/ResNet1D, log vào MLflow.
"""

import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")  # local folder mặc định
MLFLOW_EXPERIMENT = "sleep-disorder-detection"

STAGE_NAMES = {0: "Wake", 1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "REM", 6: "Movement"}


def load_features(data_dir: str) -> tuple:
    """Load feature parquet, trả về X, y."""
    # Ưu tiên dataset_labeled (có sleep stage labels)
    for fname in ("dataset_labeled.parquet", "features.parquet"):
        feats_path = Path(data_dir) / fname
        if feats_path.exists():
            break
    else:
        # SageMaker path fallback
        feats_path = Path(data_dir) / "train" / "dataset_labeled.parquet"

    logger.info(f"Loading features from {feats_path}")
    df = pd.read_parquet(feats_path)

    # Xác định cột nhãn
    if "sleep_stage" in df.columns:
        label_col = "sleep_stage"
    else:
        label_col = "label"

    # Drop metadata columns
    meta_cols = ["epoch_index", "subject_id", "label", "sleep_stage", "stage_name"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y_raw = df[label_col].values.astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    logger.info(f"Classes: {list(le.classes_)}")
    return X, y, le, feature_cols


def train_xgboost(X_train, y_train, X_val, y_val, label_encoder):
    """Train XGBoost classifier."""
    from xgboost import XGBClassifier

    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, class_weights))
    sample_weights = np.array([weight_map[y] for y in y_train])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")
    report = classification_report(
        y_val, y_pred,
        target_names=label_encoder.classes_,
    )
    return model, f1, report


def main():
    parser = argparse.ArgumentParser(description="Sleep Disorder Classifier Training")
    parser.add_argument("--data-dir", default="/opt/ml/input/data", help="Data directory")
    parser.add_argument("--model-dir", default="/opt/ml/model", help="Model output directory")
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y, label_encoder, feature_cols = load_features(args.data_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y if np.bincount(y).min() >= 2 else None,
    )

    with mlflow.start_run(run_name=f"{args.model_type}_run") as run:
        mlflow.log_params({
            "model_type": args.model_type,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": X.shape[1],
            "n_classes": len(label_encoder.classes_),
            "classes": list(label_encoder.classes_),
        })

        if args.model_type == "xgboost":
            model, f1, report = train_xgboost(
                X_train, y_train, X_val, y_val, label_encoder
            )

        logger.info(f"\nClassification Report:\n{report}")
        mlflow.log_metric("val_f1_weighted", f1)
        mlflow.log_text(report, "classification_report.txt")

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="sleep-disorder-classifier",
        )

        # Save label encoder
        import pickle
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact(str(model_dir / "label_encoder.pkl"))

        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"F1 (weighted): {f1:.4f}")
        logger.info(f"MLflow UI: {MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}")


if __name__ == "__main__":
    main()
