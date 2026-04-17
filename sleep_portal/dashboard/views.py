import json
from collections import Counter
from pathlib import Path

from django.conf import settings
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, render

from .models import Patient, EpochPrediction

# Vietnamese name mapping for disorder classes
_VI_NAMES = {
    "healthy":    "Bình thường",
    "insomnia":   "Mất ngủ",
    "narcolepsy": "Ngủ rũ",
    "nfle":       "Động kinh thùy trán về đêm",
    "plm":        "Cử động chân định kỳ",
    "rbd":        "Rối loạn hành vi REM",
    "sdb":        "Rối loạn hô hấp khi ngủ",
}
_NORMAL_CLASSES = {"healthy"}


def _vi_name(cls: str) -> str:
    return _VI_NAMES.get(str(cls).lower(), cls)


def dashboard_home(request):
    patients = Patient.objects.all().order_by("patient_id")
    prediction_qs = EpochPrediction.objects.select_related("patient")
    total_patients = patients.count()
    total_predictions = prediction_qs.count()
    monitored_patients = prediction_qs.values("patient_id").distinct().count()
    average_confidence = prediction_qs.exclude(confidence__isnull=True).aggregate(
        value=Avg("confidence")
    )["value"] or 0
    diagnosis_counts = list(
        patients.values("diagnosis")
        .annotate(count=Count("id"))
        .order_by("-count", "diagnosis")
    )
    diagnosis_breakdown = [
        {
            "name": item["diagnosis"],
            "vi_name": _vi_name(item["diagnosis"]),
            "count": item["count"],
            "percentage": ((item["count"] / total_patients) * 100) if total_patients else 0,
        }
        for item in diagnosis_counts
    ]
    normal_count = sum(
        item["count"] for item in diagnosis_counts
        if item["diagnosis"].lower() in _NORMAL_CLASSES
    )
    abnormal_count = total_patients - normal_count
    normal_pct = (normal_count / total_patients * 100) if total_patients else 0
    abnormal_pct = (abnormal_count / total_patients * 100) if total_patients else 0
    return render(
        request,
        "dashboard/home.html",
        {
            "recent_predictions": prediction_qs.order_by("-timestamp")[:5],
            "total_patients": total_patients,
            "total_predictions": total_predictions,
            "monitored_patients": monitored_patients,
            "average_confidence": average_confidence,
            "diagnosis_breakdown": diagnosis_breakdown,
            "normal_count": normal_count,
            "abnormal_count": abnormal_count,
            "normal_pct": normal_pct,
            "abnormal_pct": abnormal_pct,
            "model_name": settings.MLFLOW_MODEL_NAME,
            "model_stage": settings.MLFLOW_MODEL_STAGE,
        },
    )


def patient_list(request):
    patients = Patient.objects.all().order_by("patient_id")
    diagnosis_counts = list(
        patients.values("diagnosis")
        .annotate(count=Count("id"))
        .order_by("-count", "diagnosis")
    )
    return render(
        request,
        "dashboard/patient_list.html",
        {
            "patients": patients,
            "total_patients": patients.count(),
            "diagnosis_breakdown": diagnosis_counts,
        },
    )


def patient_detail(request, patient_id):
    patient = get_object_or_404(Patient, patient_id=patient_id)
    predictions = EpochPrediction.objects.filter(patient=patient).order_by("epoch_index")
    confidence_values = predictions.exclude(confidence__isnull=True)

    # Build chart data for JS visualizations
    pred_list = list(predictions.values("epoch_index", "predicted_class", "confidence"))
    chart_data = json.dumps([
        {"epoch_index": p["epoch_index"], "cls": p["predicted_class"]}
        for p in pred_list
    ])
    class_distribution = dict(
        Counter(p["predicted_class"] for p in pred_list)
    )
    class_distribution_json = json.dumps(class_distribution)

    return render(
        request,
        "dashboard/patient_detail.html",
        {
            "patient": patient,
            "predictions": predictions,
            "n_epochs": predictions.count(),
            "average_confidence": confidence_values.aggregate(value=Avg("confidence"))["value"]
            or 0,
            "latest_prediction": predictions.order_by("-epoch_index").first(),
            "chart_data": chart_data,
            "class_distribution": class_distribution,
            "class_distribution_json": class_distribution_json,
        },
    )


def predict_page(request):
    """Dedicated inference studio — single feature vector, batch CSV, and EDF upload."""
    try:
        from inference.predictor import get_feature_count
        feature_count = get_feature_count()
    except Exception:
        feature_count = 18
    return render(
        request,
        "dashboard/predict.html",
        {"expected_feature_count": feature_count},
    )


def pipeline_page(request):
    """Pipeline status — model registry, workflows, monitoring."""
    workflow_root = Path(settings.BASE_DIR) / ".github" / "workflows"
    return render(
        request,
        "dashboard/pipeline.html",
        {
            "model_name": settings.MLFLOW_MODEL_NAME,
            "model_stage": settings.MLFLOW_MODEL_STAGE,
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "monitoring_ready": (workflow_root / "monitoring.yml").exists(),
            "retrain_ready": (workflow_root / "retrain.yml").exists(),
            "ci_ready": (workflow_root / "ci.yml").exists(),
        },
    )
