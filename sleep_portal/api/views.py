"""REST endpoints for the Sleep Portal web app and inference service."""

import io
import tempfile
from pathlib import Path

import numpy as np
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import PredictRequestSerializer
from inference.predictor import get_model_status, predict
from loguru import logger


class PredictView(APIView):
    """
    POST /api/v1/predict/
    Body: { "features": [[f1, f2, ..., f43]] }
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
    Runs the full pipeline: bandpass filter → 2-second epoch → extract 24 features → predict.
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

        # Size guard — 500 MB max
        if uploaded.size > 500 * 1024 * 1024:
            return Response({"error": "File too large (max 500 MB)."}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        try:
            import mne
            from scipy import signal as scipy_signal
            from scipy.stats import entropy as scipy_entropy, skew, kurtosis

            # Write to temp file so MNE can read it
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                for chunk in uploaded.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Load EDF — pick first EEG channel
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            Path(tmp_path).unlink(missing_ok=True)

            sfreq = raw.info["sfreq"]
            duration = raw.times[-1]

            # Bandpass filter 0.5–40 Hz
            raw.filter(l_freq=0.5, h_freq=40.0, method="fir", verbose=False)

            # Pick single EEG channel (first available)
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            if len(eeg_picks) == 0:
                eeg_picks = [0]
            ch_idx = eeg_picks[0]
            signal, _ = raw[[ch_idx], :]   # shape (1, n_samples)
            signal = signal[0]             # flatten to 1D

            # Epoch into 2-second windows (1024 samples at 512 Hz)
            window_sec = 2.0
            window_samples = int(window_sec * sfreq)
            if window_samples < 16:
                return Response({"error": "Sampling rate too low for 2-second window."}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

            n_epochs = int(len(signal) // window_samples)
            if n_epochs == 0:
                return Response({"error": "Recording too short for 2-second epochs."}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

            FREQ_BANDS = {
                "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
            }

            def bandpower(psd, freqs, fmin, fmax):
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                return float(np.trapezoid(psd[idx], freqs[idx])) if idx.sum() > 0 else 0.0

            all_feature_rows = []
            for ep_idx in range(n_epochs):
                start = ep_idx * window_samples
                epoch = signal[start: start + window_samples]

                nperseg = min(256, len(epoch))
                freqs, psd = scipy_signal.welch(epoch, fs=sfreq, nperseg=nperseg)
                total_power = bandpower(psd, freqs, 0.5, 40.0) + 1e-12

                feats = []
                band_powers = {}
                for band_name, (fmin, fmax) in FREQ_BANDS.items():
                    bp = bandpower(psd, freqs, fmin, fmax)
                    band_powers[band_name] = bp
                    feats.append(bp)            # absolute power
                    feats.append(bp / total_power)  # relative power

                psd_norm = psd / (psd.sum() + 1e-12)
                feats.append(float(scipy_entropy(psd_norm + 1e-12)))   # spectral_entropy
                feats.append(float(freqs[np.argmax(psd)]))              # peak_frequency
                feats.append(float(np.sum(freqs * psd) / (psd.sum() + 1e-12)))  # mean_frequency

                feats.append(float(np.mean(np.abs(epoch))))             # amplitude_mean
                feats.append(float(np.std(epoch)))                      # amplitude_std
                feats.append(float(np.sqrt(np.mean(epoch ** 2))))       # rms

                feats.append(band_powers["delta"] / (band_powers["beta"] + 1e-12))   # delta_beta_ratio
                feats.append(band_powers["theta"] / (band_powers["alpha"] + 1e-12))  # theta_alpha_ratio

                feats.append(float(skew(epoch)))          # skewness
                feats.append(float(kurtosis(epoch)))      # kurtosis

                # zero_crossing_rate
                zcr = float(np.sum(np.abs(np.diff(np.sign(epoch)))) / (2 * len(epoch)))
                feats.append(zcr)

                # Hjorth parameters
                diff1 = np.diff(epoch)
                diff2 = np.diff(diff1)
                activity = float(np.var(epoch))
                mobility = float(np.sqrt(np.var(diff1) / (activity + 1e-12)))
                complexity = float(
                    np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12)
                )
                feats.extend([activity, mobility, complexity])   # hjorth_activity, hjorth_mobility, hjorth_complexity

                all_feature_rows.append(feats)  # 24 features

            features = np.array(all_feature_rows, dtype=np.float32)
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
    """GET /api/v1/health/ — Service heartbeat."""
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class ModelInfoView(APIView):
    """GET /api/v1/model-info/ — Live model metadata."""
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        return Response(get_model_status(), status=status.HTTP_200_OK)
