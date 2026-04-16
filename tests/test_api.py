"""
test_api.py — Integration tests for the REST API endpoints.
"""
import json
from unittest.mock import patch

import pytest


PREDICT_URL = "/api/v1/predict/"
PREDICT_EDF_URL = "/api/v1/predict-edf/"
HEALTH_URL = "/api/v1/health/"
MODEL_INFO_URL = "/api/v1/model-info/"

MOCK_SINGLE_RESULT = {
    "predicted_class": "healthy",
    "predictions": ["healthy"],
    "prediction_count": 1,
    "class_counts": {"healthy": 1},
    "cached": False,
}

MOCK_BATCH_RESULT = {
    "predicted_class": "healthy",
    "predictions": ["healthy", "insomnia", "healthy"],
    "prediction_count": 3,
    "class_counts": {"healthy": 2, "insomnia": 1},
    "cached": False,
}


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, django_client):
        response = django_client.get(HEALTH_URL)
        assert response.status_code == 200

    def test_response_has_status_ok(self, django_client):
        response = django_client.get(HEALTH_URL)
        assert response.json()["status"] == "ok"

    def test_health_not_requiring_auth(self, django_client):
        """Health endpoint must be open — no auth header supplied."""
        response = django_client.get(HEALTH_URL)
        assert response.status_code != 401
        assert response.status_code != 403


# ── Model info ────────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_returns_200(self, django_client):
        with patch("inference.predictor._get_model", side_effect=Exception("no model")):
            response = django_client.get(MODEL_INFO_URL)
        assert response.status_code == 200

    def test_response_structure(self, django_client):
        with patch("inference.predictor._get_model", side_effect=Exception("no model")):
            data = django_client.get(MODEL_INFO_URL).json()
        assert "model_name" in data
        assert "model_stage" in data
        assert "tracking_uri" in data

    def test_loaded_false_when_model_unavailable(self, django_client):
        with patch("inference.predictor._get_model", side_effect=Exception("no model")):
            data = django_client.get(MODEL_INFO_URL).json()
        assert data.get("loaded") is False or data.get("ready") is False


# ── Predict endpoint ─────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_single_epoch_returns_200(self, django_client):
        payload = {"features": [[float(i) for i in range(24)]]}
        with patch("api.views.predict", return_value=MOCK_SINGLE_RESULT):
            response = django_client.post(
                PREDICT_URL,
                data=json.dumps(payload),
                content_type="application/json",
            )
        assert response.status_code == 200

    def test_single_epoch_returns_predicted_class(self, django_client):
        payload = {"features": [[float(i) for i in range(24)]]}
        with patch("api.views.predict", return_value=MOCK_SINGLE_RESULT):
            data = django_client.post(
                PREDICT_URL,
                data=json.dumps(payload),
                content_type="application/json",
            ).json()
        assert "predicted_class" in data

    def test_batch_predictions_returned(self, django_client):
        payload = {
            "features": [
                [float(i) for i in range(24)],
                [float(i + 10) for i in range(24)],
                [float(i + 20) for i in range(24)],
            ]
        }
        with patch("api.views.predict", return_value=MOCK_BATCH_RESULT):
            data = django_client.post(
                PREDICT_URL,
                data=json.dumps(payload),
                content_type="application/json",
            ).json()
        assert data["prediction_count"] == 3
        assert len(data["predictions"]) == 3

    def test_wrong_feature_count_returns_400(self, django_client):
        """Sending 10 features instead of 24 should return HTTP 400."""
        payload = {"features": [[0.1] * 10]}
        response = django_client.post(
            PREDICT_URL,
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_missing_features_key_returns_400(self, django_client):
        payload = {"wrong_key": [[0.1] * 24]}
        response = django_client.post(
            PREDICT_URL,
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_string_features_returns_400(self, django_client):
        payload = {"features": "not-a-list"}
        response = django_client.post(
            PREDICT_URL,
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_empty_features_list_returns_400(self, django_client):
        payload = {"features": []}
        response = django_client.post(
            PREDICT_URL,
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400


# ── EDF upload endpoint ───────────────────────────────────────────────────────

class TestPredictEDFEndpoint:
    def test_no_file_returns_400(self, django_client):
        response = django_client.post(PREDICT_EDF_URL)
        assert response.status_code == 400

    def test_non_edf_file_returns_400(self, django_client, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b"not an edf file")
        with open(txt_file, "rb") as f:
            response = django_client.post(PREDICT_EDF_URL, {"file": f})
        assert response.status_code == 400

    def test_oversized_file_returns_413(self, django_client, tmp_path):
        """A file named .edf but exceeding 500 MB should be rejected."""
        big_edf = tmp_path / "big.edf"
        # Write a header-only stub renamed .edf but above the size limit check.
        # We patch uploaded.size so we don't actually allocate 500MB in tests.
        import io
        from unittest.mock import MagicMock, patch

        mock_file = MagicMock()
        mock_file.name = "big.edf"
        mock_file.size = 501 * 1024 * 1024  # 501 MB
        mock_file.chunks.return_value = iter([b""])

        with patch("django.test.client.encode_multipart"):
            from django.test import RequestFactory
            from api.views import PredictEDFView

            factory = RequestFactory()
            request = factory.post(PREDICT_EDF_URL)
            request.FILES["file"] = mock_file
            view = PredictEDFView.as_view()
            response = view(request)

        assert response.status_code == 413


# ── Dashboard pages ───────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestDashboardPages:
    def test_home_returns_200(self, django_client):
        response = django_client.get("/")
        assert response.status_code == 200

    def test_patients_returns_200(self, django_client):
        response = django_client.get("/patients/")
        assert response.status_code == 200

    def test_predict_page_returns_200(self, django_client):
        response = django_client.get("/predict/")
        assert response.status_code == 200

    def test_pipeline_page_returns_200(self, django_client):
        response = django_client.get("/pipeline/")
        assert response.status_code == 200
