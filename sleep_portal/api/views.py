"""REST endpoints for the Sleep Portal web app and inference service."""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from django.conf import settings
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import PredictRequestSerializer
from inference.predictor import get_model_status, predict
from loguru import logger


def _ensure_project_root_on_path() -> None:
    """Make the shared feature package importable in local and Docker layouts."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "feature_engineering" / "cap_features.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()


class PredictView(APIView):
    """
    POST /api/v1/predict/
    Body: { "features": [[f1, f2, ..., f24]] }
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        features = np.array(serializer.validated_data["features"], dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            result = predict(features)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return Response(
                {"error": "Prediction failed. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class PredictEDFView(APIView):
    """
    POST /api/v1/predict-edf/
    Multipart form with field 'file' containing an EDF recording.
    Runs the full pipeline: bandpass filter -> 2-second windows -> extract 24 features -> predict.
    Features match the CAP Sleep dataset format: single EEG channel, 512 Hz, 1024 samples/window.
    """
    permission_classes = [AllowAny]
    authentication_classes = []
    parser_classes = [MultiPartParser]

    def post(self, request):
        uploaded = request.FILES.get("file")
        if not uploaded:
            return Response({"error": "No file uploaded. Use field name 'file'."}, status=status.HTTP_400_BAD_REQUEST)

        if not uploaded.name.lower().endswith(".edf"):
            return Response({"error": "Only .edf files are supported."}, status=status.HTTP_400_BAD_REQUEST)

        # Size guard: 500 MB max.
        if uploaded.size > 500 * 1024 * 1024:
            return Response({"error": "File too large (max 500 MB)."}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        try:
            import mne
            from feature_engineering.cap_features import (
                WINDOW_SEC,
                extract_feature_matrix,
            )

            # Write to temp file so MNE can read it
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                for chunk in uploaded.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Load EDF and pick the first EEG channel.
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            Path(tmp_path).unlink(missing_ok=True)

            sfreq = raw.info["sfreq"]
            duration = raw.times[-1]

            # Bandpass filter 0.5-40 Hz.
            raw.filter(l_freq=0.5, h_freq=40.0, method="fir", verbose=False)

            # Pick single EEG channel (first available)
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            ch_idx = eeg_picks[0]
            signal, _ = raw[[ch_idx], :]   # shape (1, n_samples)
            signal = signal[0]             # flatten to 1D

            # Epoch into notebook-standard 2-second windows.
            window_sec = WINDOW_SEC
            window_samples = int(window_sec * sfreq)
            if window_samples < 16:
                return Response({"error": "Sampling rate too low for 2-second window."}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

            n_epochs = int(len(signal) // window_samples)
            if n_epochs == 0:
                return Response({"error": "Recording too short for 2-second epochs."}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

            features = extract_feature_matrix(
                signal,
                sfreq=sfreq,
                window_samples=window_samples,
            )
            result = predict(features)
            result["n_epochs"] = n_epochs
            result["sfreq"] = sfreq
            result["duration_sec"] = duration
            result["channel_used"] = raw.ch_names[ch_idx]
            result["window_sec"] = window_sec
            return Response(result, status=status.HTTP_200_OK)

        except ImportError:
            return Response(
                {"error": "MNE-Python is not installed. Install with: pip install mne"},
                status=status.HTTP_501_NOT_IMPLEMENTED,
            )
        except Exception as e:
            logger.error(f"EDF prediction error: {e}")
            return Response(
                {"error": f"EDF processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HealthCheckView(APIView):
    """GET /api/v1/health/ - Service heartbeat."""
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class ModelInfoView(APIView):
    """GET /api/v1/model-info/ - Live model metadata."""
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response(get_model_status(), status=status.HTTP_200_OK)


class IngestView(APIView):
    """
    POST /api/v1/ingest/
    Receive IoT predictions and store Patient + EpochPrediction records.

    Body:
    {
        "patient_id": "patient_001",
        "disorder": "insomnia",
        "age": 35,
        "gender": "M",
        "epochs": [
            {
                "epoch_index": 0,
                "predicted_class": "nfle",
                "confidence": 0.72,
                "timestamp": "2026-04-17T02:18:39Z"
            },
            ...
        ]
    }
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        from dashboard.models import Patient, EpochPrediction
        from feature_engineering.cap_features import FEATURE_NAMES
        from monitoring.feature_store import write_ingest_feature_batch

        patient_id = request.data.get("patient_id", "").strip()
        if not patient_id:
            return Response({"error": "patient_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        disorder = request.data.get("disorder", "unknown").strip()
        age = request.data.get("age")
        gender = request.data.get("gender", "").strip() or None
        epochs_data = request.data.get("epochs", [])

        if not isinstance(epochs_data, list):
            return Response({"error": "epochs must be a list."}, status=status.HTTP_400_BAD_REQUEST)

        # Upsert Patient
        patient, created = Patient.objects.update_or_create(
            patient_id=patient_id,
            defaults={
                "diagnosis": disorder,
                "age": age,
                "gender": gender,
            },
        )

        # Bulk upsert EpochPredictions
        saved = 0
        skipped = 0
        for ep in epochs_data:
            try:
                epoch_index = int(ep.get("epoch_index", 0))
                predicted_class = str(ep.get("predicted_class", "unknown"))
                confidence = ep.get("confidence")
                ts_raw = ep.get("timestamp")
                if ts_raw:
                    try:
                        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    except ValueError:
                        ts = datetime.now(tz=timezone.utc)
                else:
                    ts = datetime.now(tz=timezone.utc)

                EpochPrediction.objects.update_or_create(
                    patient=patient,
                    epoch_index=epoch_index,
                    defaults={
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "timestamp": ts,
                    },
                )
                saved += 1
            except Exception as exc:
                logger.warning(f"Ingest: skip epoch {ep}: {exc}")
                skipped += 1

        feature_store_result = {"rows": 0, "local_path": None, "s3_uri": None}
        try:
            feature_store_result = write_ingest_feature_batch(
                patient_id=patient_id,
                diagnosis=disorder,
                epochs=epochs_data,
                feature_names=FEATURE_NAMES,
                local_dir=settings.MLOPS_FEATURE_STORE_LOCAL_DIR,
                s3_uri=settings.MLOPS_FEATURE_STORE_S3_URI,
                aws_region=settings.AWS_DEFAULT_REGION,
            )
        except Exception as exc:
            logger.warning(f"Ingest feature store write failed: {exc}")

        logger.info(
            f"Ingest: patient={patient_id} ({'created' if created else 'updated'}), "
            f"epochs saved={saved}, skipped={skipped}, "
            f"feature rows={feature_store_result.get('rows', 0)}"
        )
        return Response(
            {
                "patient_id": patient_id,
                "patient_created": created,
                "diagnosis": disorder,
                "epochs_saved": saved,
                "epochs_skipped": skipped,
                "feature_rows_saved": feature_store_result.get("rows", 0),
                "feature_store": feature_store_result,
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )
