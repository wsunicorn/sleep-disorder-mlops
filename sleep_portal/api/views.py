"""
API app — REST endpoints cho Sleep Portal.
"""

import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .serializers import PredictRequestSerializer
from inference.predictor import predict
from loguru import logger


class PredictView(APIView):
    """
    POST /api/v1/predict/
    Body: { "features": [[f1, f2, ..., fn]] }
    Response: { "predicted_class": "insomnia", "cached": false }
    """
    permission_classes = [AllowAny]

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


class HealthCheckView(APIView):
    """GET /api/v1/health/ — Kiểm tra service."""
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)
