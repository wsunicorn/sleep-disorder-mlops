"""Inference helpers for loading the registered model and serving predictions."""

import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from loguru import logger

from .artifact_sync import sync_model_artifacts_once


_model = None
_feature_names = None
_label_encoder = None
_using_pkl_fallback = False
_artifact_sync_status = None


def _model_dirs() -> list[Path]:
    configured = Path(getattr(settings, "MODEL_ARTIFACT_LOCAL_DIR", "models"))
    return [
        configured,
        Path(settings.BASE_DIR).parent / "models",
        Path(settings.BASE_DIR) / "models",
        Path("/app/models"),
    ]


def _sync_artifacts_if_configured() -> dict:
    """Download latest S3 model artifacts once per process when configured."""
    global _artifact_sync_status
    if _artifact_sync_status is not None:
        return _artifact_sync_status

    _artifact_sync_status = sync_model_artifacts_once(
        artifact_uri=getattr(settings, "MODEL_ARTIFACT_S3_URI", ""),
        local_dir=getattr(settings, "MODEL_ARTIFACT_LOCAL_DIR", "models"),
        aws_region=getattr(settings, "AWS_DEFAULT_REGION", None),
    )
    if _artifact_sync_status.get("downloaded"):
        logger.info(f"Synced model artifacts: {_artifact_sync_status['downloaded']}")
    elif _artifact_sync_status.get("error"):
        logger.warning(f"Model artifact sync skipped/failed: {_artifact_sync_status['error']}")
    return _artifact_sync_status


class _MetadataLabelEncoder:
    """Minimal inverse_transform fallback backed by metadata.json classes."""

    def __init__(self, classes):
        self.classes_ = list(classes)

    def inverse_transform(self, values):
        return [self.classes_[int(value)] for value in values]


def _install_numpy_pickle_compat():
    """Support artifacts pickled with NumPy 2 when serving with NumPy 1.x."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)
    except Exception:
        pass


def _load_label_encoder():
    """Load the label encoder from model artifacts when available."""
    global _label_encoder
    if _label_encoder is not None:
        return _label_encoder
    _sync_artifacts_if_configured()
    candidates = [path / "label_encoder.pkl" for path in _model_dirs()]
    for path in candidates:
        if path.exists():
            try:
                _install_numpy_pickle_compat()
                with open(path, "rb") as f:
                    _label_encoder = pickle.load(f)
                logger.info(f"Loaded label encoder from {path}")
                return _label_encoder
            except Exception as exc:
                logger.warning(f"Could not load label encoder from {path}: {exc}")

    metadata_candidates = [path / "metadata.json" for path in _model_dirs()]
    for path in metadata_candidates:
        if path.exists():
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                classes = metadata.get("classes")
                if classes:
                    _label_encoder = _MetadataLabelEncoder(classes)
                    logger.info(f"Loaded label classes from {path}")
                    return _label_encoder
            except Exception as exc:
                logger.warning(f"Could not load label classes from {path}: {exc}")
    return None


def _load_feature_names() -> list | None:
    """Load feature names from model artifacts when available."""
    global _feature_names
    if _feature_names is not None:
        return _feature_names
    _sync_artifacts_if_configured()
    candidates = [path / "feature_names.json" for path in _model_dirs()]
    for path in candidates:
        if path.exists():
            _feature_names = json.loads(path.read_text())
            logger.info(f"Loaded {len(_feature_names)} feature names from {path}")
            return _feature_names
    return None


def get_feature_count() -> int:
    """Return the feature count expected by the current model."""
    names = _load_feature_names()
    return len(names) if names else 24


def _get_model():
    """Singleton: load the model from MLflow Registry, then pkl fallback."""
    global _model, _label_encoder, _using_pkl_fallback
    if _model is not None:
        return _model
    _sync_artifacts_if_configured()

    # Try MLflow model registry first.
    try:
        model_uri = (
            f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"
        )
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.info(f"Loading model from MLflow: {model_uri}")
        _model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded via MLflow registry.")
        return _model
    except Exception as mlflow_exc:
        logger.warning(f"MLflow registry load failed ({mlflow_exc}); trying pkl fallback.")

    # Fallback: load model.pkl directly.
    candidates = [path / "model.pkl" for path in _model_dirs()]
    for pkl_path in candidates:
        if pkl_path.exists():
            logger.info(f"Loading model from pkl: {pkl_path}")
            _install_numpy_pickle_compat()
            with open(pkl_path, "rb") as f:
                _model = pickle.load(f)
            _using_pkl_fallback = True
            logger.info("Model loaded from pkl fallback.")
            return _model

    raise RuntimeError("Could not load model: MLflow registry failed and no model.pkl found.")


def get_model_status() -> dict:
    """Return lightweight serving metadata for the dashboard and health checks."""
    status = {
        "model_name": settings.MLFLOW_MODEL_NAME,
        "model_stage": settings.MLFLOW_MODEL_STAGE,
        "tracking_uri": settings.MLFLOW_TRACKING_URI,
        "artifact_s3_uri": getattr(settings, "MODEL_ARTIFACT_S3_URI", ""),
        "artifact_sync": _sync_artifacts_if_configured(),
        "feature_count": get_feature_count(),
        "feature_names": _load_feature_names(),
        "supports_batch": True,
    }

    try:
        model = _get_model()
        status.update({
            "ready": True,
            "model_type": type(model).__name__,
        })
    except Exception as exc:
        logger.error(f"Model status check failed: {exc}")
        status.update({
            "ready": False,
            "error": str(exc),
        })

    return status


def predict(features: np.ndarray) -> dict:
    """
    Run prediction for one or more feature rows.

    Args:
        features: numpy array shape (1, n_features)

    Returns:
        dict with predicted_class, predictions, prediction_count, class_counts, cached.
    """
    # Build a cache key from the feature payload.
    features_hash = hashlib.sha256(features.tobytes()).hexdigest()
    cache_key = f"pred:{features_hash}"

    # Check cache first.
    cached = cache.get(cache_key)
    if cached is not None:
        cached["cached"] = True
        return cached

    model = _get_model()
    feature_names = _load_feature_names()
    if feature_names and features.shape[1] == len(feature_names):
        model_input = pd.DataFrame(features, columns=feature_names)
    else:
        model_input = pd.DataFrame(features)
    preds = model.predict(model_input)
    raw = np.asarray(preds).reshape(-1)

    # Decode integer class indices to class names using the label encoder.
    le = _load_label_encoder()
    try:
        if le is not None and np.issubdtype(raw.dtype, np.integer):
            predictions = list(le.inverse_transform(raw))
        elif le is not None and not isinstance(raw[0], str):
            predictions = list(le.inverse_transform(raw.astype(int)))
        else:
            predictions = [str(p) for p in raw.tolist()]
    except Exception:
        predictions = [str(p) for p in raw.tolist()]

    result = {
        "predicted_class": predictions[0],
        "predictions": predictions,
        "prediction_count": len(predictions),
        "class_counts": dict(Counter(predictions)),
        "cached": False,
    }

    # Store in Redis cache for 1 hour.
    cache.set(cache_key, result, timeout=3600)
    return result
