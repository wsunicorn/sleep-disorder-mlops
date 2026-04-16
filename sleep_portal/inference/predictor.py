"""Inference helpers for loading the registered model and serving predictions."""

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from loguru import logger


_model = None
_feature_names = None
_label_encoder = None
_using_pkl_fallback = False


def _load_label_encoder():
    """Load label encoder từ models/label_encoder.pkl (nếu có)."""
    global _label_encoder
    if _label_encoder is not None:
        return _label_encoder
    candidates = [
        Path(settings.BASE_DIR).parent / "models" / "label_encoder.pkl",
        Path(settings.BASE_DIR) / "models" / "label_encoder.pkl",
        Path("/app/models/label_encoder.pkl"),
    ]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                _label_encoder = pickle.load(f)
            logger.info(f"Loaded label encoder from {path}")
            return _label_encoder
    return None


def _load_feature_names() -> list:
    """Load feature names từ models/feature_names.json (nếu có), fallback về số lượng mặc định."""
    global _feature_names
    if _feature_names is not None:
        return _feature_names
    # BASE_DIR là thư mục gốc project (parent của sleep_portal/)
    candidates = [
        Path(settings.BASE_DIR).parent / "models" / "feature_names.json",
        Path(settings.BASE_DIR) / "models" / "feature_names.json",
    ]
    for path in candidates:
        if path.exists():
            _feature_names = json.loads(path.read_text())
            logger.info(f"Loaded {len(_feature_names)} feature names from {path}")
            return _feature_names
    # fallback: trả về None → serializer dùng giá trị mặc định
    return None


def get_feature_count() -> int:
    """Trả về số features mà model hiện tại cần."""
    names = _load_feature_names()
    return len(names) if names else 24


def _get_model():
    """Singleton: load model từ MLflow Registry; fallback pkl nếu registry lỗi."""
    global _model, _label_encoder, _using_pkl_fallback
    if _model is not None:
        return _model

    # Thử load qua MLflow model registry
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

    # Fallback: load model.pkl trực tiếp
    candidates = [
        Path(settings.BASE_DIR).parent / "models" / "model.pkl",
        Path(settings.BASE_DIR) / "models" / "model.pkl",
        Path("/app/models/model.pkl"),
    ]
    for pkl_path in candidates:
        if pkl_path.exists():
            logger.info(f"Loading model from pkl: {pkl_path}")
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
    Chạy prediction cho một epoch.

    Args:
        features: numpy array shape (1, n_features)

    Returns:
        dict với keys: predicted_class, probabilities, cached
    """
    # Tạo cache key từ features hash
    features_hash = hashlib.sha256(features.tobytes()).hexdigest()
    cache_key = f"pred:{features_hash}"

    # Kiểm tra cache trước
    cached = cache.get(cache_key)
    if cached is not None:
        cached["cached"] = True
        return cached

    model = _get_model()
    preds = model.predict(pd.DataFrame(features))
    raw = np.asarray(preds).reshape(-1)

    # Decode integer class indices → class names using label encoder
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

    # Lưu vào Redis cache (1 giờ)
    cache.set(cache_key, result, timeout=3600)
    return result
