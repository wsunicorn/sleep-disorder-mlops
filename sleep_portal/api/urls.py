from django.urls import path

from .views import HealthCheckView, ModelInfoView, PredictView, PredictEDFView

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("predict-edf/", PredictEDFView.as_view(), name="predict_edf"),
    path("health/", HealthCheckView.as_view(), name="health"),
    path("model-info/", ModelInfoView.as_view(), name="model_info"),
]
