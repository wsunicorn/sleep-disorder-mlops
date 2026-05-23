"""Train the sleep disorder classifier using the Kaggle notebook standard.

Source of truth: notebooks/kaggle_cap_training.ipynb
- Balanced CAP CSV input
- 24 handcrafted features from 1024-sample EEG windows
- 7 disease classes
- XGBoost, LightGBM, and RandomForest comparison
- MLflow tracking plus exportable artifacts in models/
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Callable

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from feature_engineering.cap_features import (
        DISEASE_FILES,
        FEATURE_NAMES,
        NOTEBOOK_MAX_PER_CLASS,
        SFREQ,
        WINDOW_SAMPLES,
        load_balanced_cap_dataset,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from feature_engineering.cap_features import (
        DISEASE_FILES,
        FEATURE_NAMES,
        NOTEBOOK_MAX_PER_CLASS,
        SFREQ,
        WINDOW_SAMPLES,
        load_balanced_cap_dataset,
    )


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "sleep-disorder-kaggle")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sleep-disorder-classifier")


def _training_threads() -> int:
    value = os.getenv("TRAINING_NUM_THREADS") or os.getenv("OMP_NUM_THREADS") or "2"
    try:
        return max(1, int(value))
    except ValueError:
        return 2


for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, str(_training_threads()))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in S3 URI: {uri}")
    return bucket, key.strip("/")


def _load_s3_parquet(uri: str) -> pd.DataFrame:
    import boto3

    bucket, key = _parse_s3_uri(uri)
    s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))

    if key.endswith(".parquet"):
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    prefix = key.rstrip("/")
    if prefix:
        prefix = f"{prefix}/"
    frames: list[pd.DataFrame] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            object_key = item["Key"]
            if not object_key.endswith(".parquet"):
                continue
            obj = s3.get_object(Bucket=bucket, Key=object_key)
            frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))
    if not frames:
        raise FileNotFoundError(f"No parquet files found under {uri}")
    return pd.concat(frames, ignore_index=True)


def _load_parquet_or_directory(path: str | Path) -> pd.DataFrame:
    path_str = str(path)
    if path_str.startswith("s3://"):
        logger.info(f"Loading feature parquet from {path_str}")
        return _load_s3_parquet(path_str)

    local_path = Path(path)
    if local_path.is_dir():
        parquet_files = sorted(local_path.rglob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} parquet files from {local_path}")
            return pd.concat(
                [pd.read_parquet(parquet_file) for parquet_file in parquet_files],
                ignore_index=True,
            )
    return pd.read_parquet(local_path)


def detect_devices() -> tuple[str, str]:
    """Return XGBoost and LightGBM device names following the notebook."""
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda", "gpu"
    except Exception:
        return "cpu", "cpu"


def _parse_class_limits(args: argparse.Namespace) -> dict[str, int | None]:
    limits = dict(NOTEBOOK_MAX_PER_CLASS)
    for label in DISEASE_FILES:
        value = getattr(args, f"max_{label}")
        if value is not None:
            limits[label] = None if value <= 0 else int(value)
    return limits


def _load_feature_frame(
    data_dir: str,
    class_limits: dict[str, int | None],
    allow_synthetic: bool,
) -> pd.DataFrame:
    if data_dir.startswith("s3://"):
        return _load_parquet_or_directory(data_dir)

    data_path = Path(data_dir)
    if data_path.is_dir() and list(data_path.rglob("*.parquet")):
        return _load_parquet_or_directory(data_path)

    parquet_candidates = [
        data_path if data_path.suffix == ".parquet" else None,
        data_path / "features.parquet",
        data_path / "train" / "features.parquet",
    ]
    for candidate in parquet_candidates:
        if candidate and candidate.exists():
            logger.info(f"Loading precomputed features from {candidate}")
            return _load_parquet_or_directory(candidate)

    logger.info(f"Extracting features from Balanced CAP CSVs in {data_path}")
    return load_balanced_cap_dataset(
        data_path,
        max_per_class=class_limits,
        synthetic_if_missing=allow_synthetic,
    )


def load_training_data(
    data_dir: str,
    class_limits: dict[str, int | None],
    allow_synthetic: bool = False,
    extra_data: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame]:
    """Load or build a 24-feature disease-classification dataset."""
    df = _load_feature_frame(data_dir, class_limits, allow_synthetic)
    for extra_path in extra_data or []:
        if not extra_path:
            continue
        extra_df = _load_feature_frame(extra_path, class_limits, False)
        logger.info(f"Appending extra training data from {extra_path}: {len(extra_df)} rows")
        df = pd.concat([df, extra_df], ignore_index=True)

    label_col = "disease" if "disease" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError("Training data must contain a 'disease' or 'label' column.")

    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        raise ValueError(
            "Feature data does not match the Kaggle notebook schema. "
            f"Missing columns: {missing}"
        )

    df = df[df[label_col].isin(DISEASE_FILES.keys())].copy()
    if df.empty:
        raise ValueError("No 7-class sleep disorder labels found in training data.")

    x = df[FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str).to_numpy())

    logger.info(
        f"Dataset: {x.shape[0]} samples, {x.shape[1]} features, "
        f"{len(label_encoder.classes_)} classes"
    )
    logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info(f"Label distribution:\n{df[label_col].value_counts().to_string()}")
    return x, y, label_encoder, df


def build_models(model_type: str, random_seed: int) -> dict[str, Callable[[], object]]:
    """Create lazy model factories from the notebook.

    Keeping the factories lazy avoids importing XGBoost and LightGBM native
    runtimes at the same time before the first model trains. That is safer on
    small CI runners where OpenMP-backed libraries can otherwise segfault.
    """
    device, lgb_device = detect_devices()
    threads = _training_threads()
    models: dict[str, Callable[[], object]] = {}

    if model_type in {"all", "xgboost"}:
        def make_xgboost() -> object:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1.0,
                device=device,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=threads,
                random_state=random_seed,
            )

        models["XGBoost"] = make_xgboost

    if model_type in {"all", "lightgbm"}:
        def make_lightgbm() -> object:
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=500,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                device_type=lgb_device,
                n_jobs=threads,
                random_state=random_seed,
                verbose=-1,
            )

        models["LightGBM"] = make_lightgbm

    if model_type in {"all", "randomforest"}:
        def make_randomforest() -> object:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=10,
                class_weight="balanced",
                n_jobs=threads,
                random_state=random_seed,
            )

        models["RandomForest"] = make_randomforest

    return models


def train_and_evaluate(
    models: dict[str, Callable[[], object]],
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
    sample_weights: np.ndarray,
) -> tuple[str, dict[str, dict[str, object]]]:
    """Train each candidate model and return the best weighted-F1 model."""
    results: dict[str, dict[str, object]] = {}

    for name, model_factory in models.items():
        logger.info(f"Training {name}")
        with mlflow.start_run(run_name=name) as run:
            logger.info(f"Run ID: {run.info.run_id}")
            model = model_factory()
            fit_kwargs: dict[str, object] = {}
            if name in {"XGBoost", "LightGBM"}:
                fit_kwargs["sample_weight"] = sample_weights
            if name == "XGBoost":
                fit_kwargs["eval_set"] = [(x_val, y_val)]
                fit_kwargs["verbose"] = 100
            if name == "LightGBM":
                import lightgbm as lgb

                fit_kwargs["eval_set"] = [(x_val, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(100),
                ]

            model.fit(x_train, y_train, **fit_kwargs)
            logger.info(f"{name}: generating validation predictions")
            y_pred = model.predict(x_val)
            logger.info(f"{name}: computing validation metrics")
            f1 = f1_score(y_val, y_pred, average="weighted")
            acc = accuracy_score(y_val, y_pred)
            report = classification_report(
                y_val,
                y_pred,
                labels=np.arange(len(label_encoder.classes_)),
                target_names=label_encoder.classes_,
                zero_division=0,
            )

            mlflow.log_params(
                {
                    "model": name,
                    "n_features": x_train.shape[1],
                    "n_classes": len(label_encoder.classes_),
                    "n_train": len(x_train),
                    "classes": list(label_encoder.classes_),
                }
            )
            mlflow.log_metrics({"val_f1_weighted": f1, "val_accuracy": acc})
            mlflow.log_text(report, "classification_report.txt")
            logger.info(f"{name}: logging model to MLflow")
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

            results[name] = {
                "model": model,
                "f1": f1,
                "acc": acc,
                "report": report,
                "run_id": run.info.run_id,
            }
            logger.info(f"{name}: F1={f1:.4f}, accuracy={acc:.4f}")

    best_name = max(results, key=lambda key: float(results[key]["f1"]))
    logger.info(f"Best model: {best_name} (F1={results[best_name]['f1']:.4f})")
    return best_name, results


def export_artifacts(
    model_dir: str,
    best_name: str,
    results: dict[str, dict[str, object]],
    label_encoder: LabelEncoder,
    device: str,
) -> None:
    """Export model artifacts expected by the Django serving layer."""
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model = results[best_name]["model"]
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    if best_name == "XGBoost" and hasattr(best_model, "save_model"):
        best_model.save_model(str(output_dir / "model.ubj"))

    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    (output_dir / "feature_names.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "best_model": best_name,
        "model_name": MODEL_NAME,
        "classes": list(label_encoder.classes_),
        "n_features": len(FEATURE_NAMES),
        "val_f1": round(float(results[best_name]["f1"]), 4),
        "val_accuracy": round(float(results[best_name]["acc"]), 4),
        "sfreq": SFREQ,
        "window_samples": WINDOW_SAMPLES,
        "device": device,
        "run_id": results[best_name]["run_id"],
        "model_file": "model.pkl",
        "all_results": {
            name: {
                "f1": round(float(result["f1"]), 4),
                "acc": round(float(result["acc"]), 4),
            }
            for name, result in results.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "val_f1_weighted": metadata["val_f1"],
        "val_accuracy": metadata["val_accuracy"],
        "best_model": best_name,
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    best_run_id = str(results[best_name]["run_id"])
    try:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_artifacts(str(output_dir), artifact_path="serving_artifacts")
            mlflow.log_artifact("metrics.json")
    except Exception as exc:
        logger.warning(f"Could not log serving artifacts to MLflow: {exc}")


def promote_best_model(
    best_name: str,
    results: dict[str, dict[str, object]],
    stage: str,
    threshold: float,
    require_promotion: bool,
) -> dict[str, object]:
    """Promote the registered MLflow model version that belongs to the best run."""
    if not stage:
        return {"promoted": False, "reason": "promotion stage is empty"}

    best_f1 = float(results[best_name]["f1"])
    best_run_id = str(results[best_name]["run_id"])
    if best_f1 < threshold:
        message = (
            f"Best model {best_name} F1={best_f1:.4f} is below "
            f"promotion threshold {threshold:.4f}"
        )
        if require_promotion:
            raise RuntimeError(message)
        logger.warning(message)
        return {"promoted": False, "reason": "metric_below_threshold", "f1": best_f1}

    client = mlflow.tracking.MlflowClient()
    candidates = []
    deadline = time.time() + 60
    while time.time() < deadline:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        candidates = [version for version in versions if version.run_id == best_run_id]
        if candidates:
            ready = [
                version
                for version in candidates
                if getattr(version, "status", "READY") == "READY"
            ]
            if ready:
                candidates = ready
                break
        time.sleep(3)

    if not candidates:
        message = (
            f"Could not find a registered MLflow model version for "
            f"{MODEL_NAME} run_id={best_run_id}"
        )
        if require_promotion:
            raise RuntimeError(message)
        logger.warning(message)
        return {"promoted": False, "reason": "model_version_not_found"}

    version = max(candidates, key=lambda item: int(item.version))
    logger.info(
        f"Promoting {MODEL_NAME} version {version.version} "
        f"from run {best_run_id} to stage {stage}"
    )
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage=stage,
        archive_existing_versions=True,
    )
    client.set_model_version_tag(MODEL_NAME, version.version, "best_model", best_name)
    client.set_model_version_tag(
        MODEL_NAME,
        version.version,
        "val_f1_weighted",
        f"{best_f1:.4f}",
    )
    return {
        "promoted": True,
        "model_name": MODEL_NAME,
        "version": version.version,
        "stage": stage,
        "run_id": best_run_id,
        "f1": best_f1,
    }


def upload_artifacts_to_s3(model_dir: str, artifact_s3_uri: str) -> list[str]:
    """Upload serving artifacts to S3 so CI/CD and runtime sync can consume them."""
    import boto3

    bucket, prefix = _parse_s3_uri(artifact_s3_uri)
    output_dir = Path(model_dir)
    files = [
        output_dir / "model.pkl",
        output_dir / "model.ubj",
        output_dir / "label_encoder.pkl",
        output_dir / "feature_names.json",
        output_dir / "metadata.json",
        Path("metrics.json"),
    ]
    uploaded: list[str] = []
    s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
    for path in files:
        if not path.exists():
            continue
        key = "/".join(part for part in [prefix, path.name] if part)
        s3.upload_file(str(path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    logger.info(f"Uploaded {len(uploaded)} artifacts to {artifact_s3_uri}")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sleep disorder classifier")
    parser.add_argument("--data-dir", default="data/raw/balanced_CAP")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--model-type",
        default="all",
        choices=["all", "xgboost", "lightgbm", "randomforest"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="Accepted by CI retrain jobs.")
    parser.add_argument(
        "--extra-data",
        action="append",
        default=[],
        help="Additional feature parquet file/prefix to append to the training data.",
    )
    parser.add_argument(
        "--artifact-s3-uri",
        default=os.getenv("MODEL_ARTIFACT_S3_URI", ""),
        help="Optional s3://bucket/prefix where exported serving artifacts are uploaded.",
    )
    parser.add_argument(
        "--promote-stage",
        default=os.getenv("MODEL_PROMOTE_STAGE", ""),
        help="Optional MLflow Model Registry stage to promote the best model to.",
    )
    parser.add_argument(
        "--promote-threshold",
        type=float,
        default=float(os.getenv("MODEL_PROMOTE_THRESHOLD", "0.0")),
        help="Minimum weighted F1 required before MLflow promotion.",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        default=os.getenv("MODEL_REQUIRE_PROMOTION", "false").lower() == "true",
        help="Fail the run when promotion is requested but cannot be completed.",
    )
    for label in DISEASE_FILES:
        parser.add_argument(
            f"--max-{label}",
            type=int,
            default=None,
            help="Rows to load for this class; 0 means all rows.",
        )
    args = parser.parse_args()

    class_limits = _parse_class_limits(args)
    device, _ = detect_devices()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    x, y, label_encoder, _ = load_training_data(
        args.data_dir,
        class_limits=class_limits,
        allow_synthetic=args.allow_synthetic,
        extra_data=args.extra_data,
    )

    stratify = y if np.bincount(y).min() >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify,
    )

    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, class_weights))
    sample_weights = np.asarray([weight_map[label] for label in y_train])

    models = build_models(args.model_type, random_seed=args.random_seed)
    best_name, results = train_and_evaluate(
        models,
        x_train,
        x_val,
        y_train,
        y_val,
        label_encoder,
        sample_weights,
    )
    export_artifacts(args.model_dir, best_name, results, label_encoder, device)
    if args.artifact_s3_uri:
        upload_artifacts_to_s3(args.model_dir, args.artifact_s3_uri)
    if args.promote_stage:
        promotion = promote_best_model(
            best_name=best_name,
            results=results,
            stage=args.promote_stage,
            threshold=args.promote_threshold,
            require_promotion=args.require_promotion,
        )
        logger.info(f"Promotion result: {promotion}")

    logger.info("Training complete.")
    logger.info(f"Artifacts written to {Path(args.model_dir).resolve()}")


if __name__ == "__main__":
    main()
