from django.urls import path
from .views import PredictView, HealthCheckView

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("health/", HealthCheckView.as_view(), name="health"),
]
