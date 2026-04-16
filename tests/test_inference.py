"""
test_inference.py — Unit tests for the inference predictor module.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── get_model_status ──────────────────────────────────────────────────────────

class TestGetModelStatus:
    def test_returns_dict(self):
        from inference.predictor import get_model_status
        with patch("inference.predictor._get_model", side_effect=Exception("no model")):
            status = get_model_status()
        assert isinstance(status, dict)

    def test_contains_required_keys(self):
        from inference.predictor import get_model_status
        with patch("inference.predictor._get_model", side_effect=Exception("no model")):
            status = get_model_status()
        for key in ("model_name", "model_stage", "tracking_uri", "feature_count"):
            assert key in status, f"Missing key: {key}"

    def test_feature_count_is_24(self):
        from inference.predictor import get_model_status
        with patch("inference.predictor._get_model", side_effect=Exception("no model")):
            status = get_model_status()
        assert status["feature_count"] == 24

    def test_ready_false_when_model_unavailable(self):
        from inference.predictor import get_model_status
        with patch("inference.predictor._get_model", side_effect=Exception("unavailable")):
            status = get_model_status()
        assert status["ready"] is False

    def test_ready_true_when_model_loads(self):
        from inference.predictor import get_model_status
        mock_model = MagicMock()
        with patch("inference.predictor._get_model", return_value=mock_model):
            status = get_model_status()
        assert status["ready"] is True


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def _make_mock_model(self, predictions):
        """Return a mock MLflow model that returns `predictions` from predict()."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(predictions)
        return mock_model

    def test_single_epoch_returns_predicted_class(self, sample_features):
        from inference.predictor import predict

        mock_model = self._make_mock_model(["healthy"])
        with patch("inference.predictor._get_model", return_value=mock_model), \
             patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = None
            result = predict(sample_features)

        assert "predicted_class" in result
        assert result["predicted_class"] == "healthy"

    def test_single_epoch_returns_prediction_count_1(self, sample_features):
        from inference.predictor import predict

        mock_model = self._make_mock_model(["healthy"])
        with patch("inference.predictor._get_model", return_value=mock_model), \
             patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = None
            result = predict(sample_features)

        assert result["prediction_count"] == 1

    def test_batch_prediction_count_matches_input(self, batch_features):
        from inference.predictor import predict

        labels = ["healthy", "insomnia", "nfle", "rbd", "healthy"]
        mock_model = self._make_mock_model(labels)
        with patch("inference.predictor._get_model", return_value=mock_model), \
             patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = None
            result = predict(batch_features)

        assert result["prediction_count"] == 5
        assert len(result["predictions"]) == 5

    def test_class_counts_are_correct(self, batch_features):
        from inference.predictor import predict

        labels = ["healthy", "insomnia", "healthy", "healthy", "insomnia"]
        mock_model = self._make_mock_model(labels)
        with patch("inference.predictor._get_model", return_value=mock_model), \
             patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = None
            result = predict(batch_features)

        assert result["class_counts"]["healthy"] == 3
        assert result["class_counts"]["insomnia"] == 2

    def test_cache_hit_returns_cached_result(self, sample_features):
        from inference.predictor import predict

        cached_result = {
            "predicted_class": "rbd",
            "predictions": ["rbd"],
            "prediction_count": 1,
            "class_counts": {"rbd": 1},
            "cached": False,
        }

        with patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = cached_result
            result = predict(sample_features)

        assert result["cached"] is True
        assert result["predicted_class"] == "rbd"

    def test_cached_false_on_first_call(self, sample_features):
        from inference.predictor import predict

        mock_model = self._make_mock_model(["healthy"])
        with patch("inference.predictor._get_model", return_value=mock_model), \
             patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = None
            result = predict(sample_features)

        assert result["cached"] is False

    def test_predictions_list_has_string_values(self, sample_features):
        from inference.predictor import predict

        mock_model = self._make_mock_model([2])  # numeric label
        with patch("inference.predictor._get_model", return_value=mock_model), \
             patch("inference.predictor.cache") as mock_cache:
            mock_cache.get.return_value = None
            result = predict(sample_features)

        # All predictions must be strings
        assert all(isinstance(p, str) for p in result["predictions"])
