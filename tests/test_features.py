"""
Tests — Unit tests cho feature extraction và API.
"""

import pytest
import numpy as np
import pandas as pd
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sleep_portal.settings.development")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///test_db.sqlite3")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

import django
django.setup()


@pytest.fixture
def sample_epoch():
    """Tạo một epoch EEG giả (4 channels, 100 samples)."""
    np.random.seed(42)
    return np.random.randn(4, 7680).astype(np.float32)  # 256 Hz × 30s


@pytest.fixture
def sample_features():
    """Feature vector giả."""
    np.random.seed(42)
    return np.random.randn(1, 50).astype(np.float32)


# ─── Feature Extraction Tests ──────────────────────────────────────────────

class TestFeatureExtraction:
    def test_bandpower_returns_positive(self, sample_epoch):
        from feature_engineering.extract_features import bandpower
        from scipy import signal
        freqs, psd = signal.welch(sample_epoch[0], fs=256, nperseg=256)
        bp = bandpower(psd, freqs, 0.5, 4.0)
        assert bp >= 0

    def test_spectral_entropy_bounded(self, sample_epoch):
        from feature_engineering.extract_features import spectral_entropy
        from scipy import signal
        freqs, psd = signal.welch(sample_epoch[0], fs=256, nperseg=256)
        se = spectral_entropy(psd)
        assert se >= 0

    def test_extract_epoch_features_returns_dict(self, sample_epoch):
        from feature_engineering.extract_features import extract_epoch_features
        features = extract_epoch_features(sample_epoch, sfreq=256.0)
        assert isinstance(features, dict)
        assert "delta_power_mean" in features
        assert "theta_power_mean" in features
        assert "alpha_power_mean" in features
        assert "beta_power_mean" in features
        assert "spectral_entropy_mean" in features
        assert "delta_beta_ratio" in features

    def test_label_from_filename(self):
        from feature_engineering.extract_features import get_label_from_filename
        assert get_label_from_filename("n1") == "healthy"
        assert get_label_from_filename("nfle5") == "nfle"
        assert get_label_from_filename("ins3") == "insomnia"
        assert get_label_from_filename("rbd10") == "rbd"
        assert get_label_from_filename("narco2") == "narcolepsy"


# ─── Annotation Parser Tests ───────────────────────────────────────────────

class TestAnnotationParser:
    def test_stage_map_coverage(self):
        from feature_engineering.annotation_parser import STAGE_MAP
        assert "SLEEP-S0" in STAGE_MAP
        assert "REM" in STAGE_MAP
        assert STAGE_MAP["REM"] == 5

    def test_get_epoch_labels_with_empty_df(self):
        from feature_engineering.annotation_parser import get_epoch_labels
        empty_df = pd.DataFrame(columns=["time_sec", "sleep_stage", "event", "duration_sec"])
        labels = get_epoch_labels(empty_df)
        assert isinstance(labels, dict)
        assert len(labels) == 0


# ─── API Tests ─────────────────────────────────────────────────────────────

class TestHealthCheckAPI:
    def test_health_endpoint(self, client):
        from django.test import Client
        c = Client()
        response = c.get("/api/v1/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_endpoint_bad_input(self):
        from django.test import Client
        c = Client()
        # Sai format
        response = c.post(
            "/api/v1/predict/",
            data='{"features": "not-a-list"}',
            content_type="application/json",
        )
        assert response.status_code == 400
