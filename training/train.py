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
import json
import os
import pickle
import subprocess
from pathlib import Path

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
    data_path = Path(data_dir)
    parquet_candidates = [
        data_path if data_path.suffix == ".parquet" else None,
        data_path / "features.parquet",
        data_path / "train" / "features.parquet",
    ]
    for candidate in parquet_candidates:
        if candidate and candidate.exists():
            logger.info(f"Loading precomputed features from {candidate}")
            return pd.read_parquet(candidate)

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
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame]:
    """Load or build a 24-feature disease-classification dataset."""
    df = _load_feature_frame(data_dir, class_limits, allow_synthetic)

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


def build_models(model_type: str, random_seed: int) -> dict[str, object]:
    """Create the model set from the notebook."""
    device, lgb_device = detect_devices()
    models: dict[str, object] = {}

    if model_type in {"all", "xgboost"}:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
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
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_seed,
        )

    if model_type in {"all", "lightgbm"}:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            device_type=lgb_device,
            n_jobs=-1,
            random_state=random_seed,
            verbose=-1,
        )

    if model_type in {"all", "randomforest"}:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        )

    return models


def train_and_evaluate(
    models: dict[str, object],
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
    sample_weights: np.ndarray,
) -> tuple[str, dict[str, dict[str, object]]]:
    """Train each candidate model and return the best weighted-F1 model."""
    results: dict[str, dict[str, object]] = {}

    for name, model in models.items():
        logger.info(f"Training {name}")
        with mlflow.start_run(run_name=name) as run:
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
            y_pred = model.predict(x_val)
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

    logger.info("Training complete.")
    logger.info(f"Artifacts written to {Path(args.model_dir).resolve()}")


if __name__ == "__main__":
    main()
