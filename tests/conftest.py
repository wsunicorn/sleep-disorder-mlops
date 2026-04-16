"""
conftest.py — Shared pytest fixtures for the Sleep Portal test suite.
"""
import os

import django
import numpy as np
import pytest

# Configure Django BEFORE any Django imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sleep_portal.settings.development")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key-do-not-use-in-prod")
os.environ.setdefault("DATABASE_URL", "sqlite:///test_db.sqlite3")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "mlruns")
os.environ.setdefault("MLFLOW_MODEL_NAME", "sleep-disorder-classifier")
os.environ.setdefault("MLFLOW_MODEL_STAGE", "None")

django.setup()

# Override static files storage so tests don't need a collectstatic manifest
from django.conf import settings as _settings  # noqa: E402
_settings.STATICFILES_STORAGE = "django.contrib.staticfiles.storage.StaticFilesStorage"


# ── Primitive fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_epoch():
    """Synthetic EEG epoch: 4 channels × 7 680 samples (256 Hz × 30 s)."""
    np.random.seed(42)
    return np.random.randn(4, 7680).astype(np.float32)


@pytest.fixture
def sample_features():
    """Single epoch feature vector (shape 1×43)."""
    np.random.seed(42)
    return np.random.randn(1, 24).astype(np.float32)


@pytest.fixture
def batch_features():
    """Batch of 5 epoch feature vectors (shape 5×43)."""
    np.random.seed(0)
    return np.random.randn(5, 24).astype(np.float32)


@pytest.fixture
def django_client():
    """Django test client for HTTP requests."""
    from django.test import Client
    return Client()
