from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard_home, name="dashboard_home"),
    path("patients/", views.patient_list, name="patient_list"),
    path("patients/<str:patient_id>/", views.patient_detail, name="patient_detail"),
    path("predict/", views.predict_page, name="predict_page"),
    path("pipeline/", views.pipeline_page, name="pipeline_page"),
]
