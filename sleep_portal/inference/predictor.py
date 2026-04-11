"""
Inference app — Load model từ MLflow Registry, chạy prediction với Redis cache.
"""

import os
import pickle
import hashlib
import numpy as np
from django.core.cache import cache
from django.conf import settings
from loguru import logger
import mlflow.pyfunc


_model = None
_label_encoder = None


def _get_model():
    """Singleton: load model từ MLflow Registry (chỉ load 1 lần)."""
    global _model, _label_encoder
    if _model is None:
        model_uri = (
            f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"
        )
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.info(f"Loading model from MLflow: {model_uri}")
        _model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
    return _model


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
    import pandas as pd
    preds = model.predict(pd.DataFrame(features))

    result = {
        "predicted_class": str(preds[0]),
        "cached": False,
    }

    # Lưu vào Redis cache (1 giờ)
    cache.set(cache_key, result, timeout=3600)
    return result
