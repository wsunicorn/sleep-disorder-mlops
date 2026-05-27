"""Register existing serving artifacts in MLflow and optionally promote them.

This is useful when a reviewed benchmark artifact already exists in models/
and should become the production model without retraining on unverified data.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from loguru import logger
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import MODEL_NAME, upload_artifacts_to_s3  # noqa: E402


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key, got: {uri}")
    return bucket, key.strip("/")


def _upload_directory_to_s3(local_dir: Path, s3_uri: str) -> str:
    bucket, prefix = _parse_s3_uri(s3_uri)
    try:
        import boto3

        s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{relative}"
            s3.upload_file(str(path), bucket, key)
    except ModuleNotFoundError:
        subprocess.run(
            ["aws", "s3", "cp", str(local_dir), s3_uri, "--recursive"],
            check=True,
        )
    return f"s3://{bucket}/{prefix}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _install_numpy_pickle_compat() -> None:
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)
    except Exception:
        pass

    # LightGBM imports optional Dask integrations at import time. Some Windows
    # demo environments crash in that optional import path, while the model
    # itself does not need Dask to be unpickled or registered.
    for module_name in ("dask", "dask.array", "dask.dataframe", "dask.distributed"):
        sys.modules.setdefault(module_name, None)


def _wait_for_model_version(model_name: str, run_id: str):
    client = mlflow.tracking.MlflowClient()
    deadline = time.time() + 90
    while time.time() < deadline:
        versions = client.search_model_versions(f"name='{model_name}'")
        candidates = [version for version in versions if version.run_id == run_id]
        ready = [
            version
            for version in candidates
            if getattr(version, "status", "READY") == "READY"
        ]
        if ready:
            return max(ready, key=lambda item: int(item.version))
        time.sleep(3)
    raise RuntimeError(f"Could not find READY model version for run_id={run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register existing model artifacts")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "sleep-disorder-kaggle"))
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", MODEL_NAME))
    parser.add_argument("--promote-stage", default=os.getenv("MODEL_PROMOTE_STAGE", ""))
    parser.add_argument("--artifact-s3-uri", default=os.getenv("MODEL_ARTIFACT_S3_URI", ""))
    parser.add_argument(
        "--direct-s3-model-uri",
        default="",
        help=(
            "Optional S3 prefix for a complete MLflow model directory. "
            "When set, the model version source points here instead of using "
            "the MLflow HTTP artifact proxy."
        ),
    )
    parser.add_argument("--run-name", default="restore-reviewed-benchmark-artifact")
    parser.add_argument(
        "--skip-mlflow-serving-artifacts",
        action="store_true",
        help="Skip logging the whole models/ directory to MLflow; useful for quick production restore.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "metadata.json"
    model_path = model_dir / "model.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    metadata = _load_json(metadata_path)
    metrics = {
        "val_f1_weighted": float(metadata["val_f1"]),
        "val_accuracy": float(metadata["val_accuracy"]),
    }
    metrics_payload = {
        **metrics,
        "best_model": metadata.get("best_model", ""),
        "source": "reviewed_repository_artifact",
    }
    Path("metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    _install_numpy_pickle_compat()
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Registering reviewed artifact in MLflow run {run_id}")
        mlflow.log_params(
            {
                "model": metadata.get("best_model", "unknown"),
                "n_features": metadata.get("n_features", 24),
                "n_classes": len(metadata.get("classes", [])),
                "classes": metadata.get("classes", []),
                "source_run_id": metadata.get("run_id", ""),
                "data_policy": "reviewed_repository_artifact",
            }
        )
        mlflow.set_tags(
            {
                "best_model": metadata.get("best_model", "unknown"),
                "data_source_summary": json.dumps(
                    {"reviewed_repository_artifact": 1},
                    ensure_ascii=True,
                ),
                "restored_from_metadata": "true",
            }
        )
        mlflow.log_metrics(metrics)
        if not args.skip_mlflow_serving_artifacts:
            mlflow.log_artifacts(str(model_dir), artifact_path="serving_artifacts")
        mlflow.log_artifact("metrics.json")

        if args.direct_s3_model_uri:
            tmp_root = ROOT / ".tmp"
            tmp_root.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
                model_output = Path(tmp_dir) / "model"
                mlflow.sklearn.save_model(model, path=str(model_output))
                source = _upload_directory_to_s3(model_output, args.direct_s3_model_uri)

            client = mlflow.tracking.MlflowClient()
            try:
                client.get_registered_model(args.model_name)
            except MlflowException:
                client.create_registered_model(args.model_name)
            version = client.create_model_version(
                name=args.model_name,
                source=source,
                run_id=run_id,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=args.model_name,
            )
            version = _wait_for_model_version(args.model_name, run_id)

    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(args.model_name, version.version, "best_model", metadata.get("best_model", ""))
    client.set_model_version_tag(
        args.model_name,
        version.version,
        "val_f1_weighted",
        f"{metrics['val_f1_weighted']:.4f}",
    )
    client.set_model_version_tag(args.model_name, version.version, "restored_from_metadata", "true")
    if args.promote_stage:
        logger.info(f"Promoting {args.model_name} version {version.version} to {args.promote_stage}")
        client.transition_model_version_stage(
            name=args.model_name,
            version=version.version,
            stage=args.promote_stage,
            archive_existing_versions=True,
        )

    if args.artifact_s3_uri:
        upload_artifacts_to_s3(str(model_dir), args.artifact_s3_uri)

    logger.info(
        f"Registered model version {version.version} "
        f"with val_f1_weighted={metrics['val_f1_weighted']:.4f}"
    )


if __name__ == "__main__":
    main()
