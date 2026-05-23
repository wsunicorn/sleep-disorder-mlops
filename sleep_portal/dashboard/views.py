import json
from collections import Counter
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

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
_PRODUCTION_APP_URL = "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com"
_PRODUCTION_MODEL_STAGE = "Production"
_PRODUCTION_MODEL_ARTIFACT_S3_URI = "s3://sleep-mlops-651709/models"
_PRODUCTION_FEATURE_STORE_S3_URI = "s3://sleep-mlops-651709/monitoring/current"


def _vi_name(cls: str) -> str:
    return _VI_NAMES.get(str(cls).lower(), cls)


def _serving_status() -> dict:
    """Return live serving metadata without breaking page rendering."""
    try:
        from inference.predictor import get_model_status

        return get_model_status()
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
            "model_name": settings.MLFLOW_MODEL_NAME,
            "model_stage": settings.MLFLOW_MODEL_STAGE,
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "feature_count": 24,
        }


def _find_workflow_root() -> Path | None:
    """Find .github/workflows in local checkout; production image may not include it."""
    candidates = [
        Path(settings.BASE_DIR).parent / ".github" / "workflows",
        Path(settings.BASE_DIR) / ".github" / "workflows",
        Path.cwd() / ".github" / "workflows",
        Path.cwd().parent / ".github" / "workflows",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _workflow_ready(workflow_root: Path | None, filename: str) -> bool:
    """Production images do not copy .github, but the workflows exist in the repo."""
    if workflow_root is not None:
        return (workflow_root / filename).exists()
    return not settings.DEBUG


def _production_value(value: str | None, fallback: str) -> str:
    value = str(value or "").strip()
    if not value or value.lower() in {"none", "mlruns"}:
        return fallback
    return value


def _display_context(model_status: dict) -> dict:
    """Values shown on the dashboard should point to the production deployment."""
    public_app_url = _production_value(
        getattr(settings, "PUBLIC_APP_URL", ""),
        _PRODUCTION_APP_URL,
    )
    tracking_uri = _production_value(
        model_status.get("tracking_uri") or getattr(settings, "MLFLOW_TRACKING_URI", ""),
        f"{public_app_url}:5000",
    )
    if not tracking_uri.startswith(("http://", "https://")):
        tracking_uri = f"{public_app_url}:5000"

    mlflow_ui_url = _production_value(
        getattr(settings, "MLFLOW_UI_URL", ""),
        tracking_uri,
    )
    if not mlflow_ui_url.startswith(("http://", "https://")):
        mlflow_ui_url = tracking_uri

    model_stage = _production_value(
        model_status.get("model_stage") or getattr(settings, "MLFLOW_MODEL_STAGE", ""),
        _PRODUCTION_MODEL_STAGE,
    )

    return {
        "public_app_url": public_app_url,
        "display_model_stage": model_stage,
        "display_tracking_uri": tracking_uri,
        "mlflow_ui_url": mlflow_ui_url,
        "display_artifact_s3_uri": _production_value(
            getattr(settings, "MODEL_ARTIFACT_S3_URI", ""),
            _PRODUCTION_MODEL_ARTIFACT_S3_URI,
        ),
        "display_feature_store_s3_uri": _production_value(
            getattr(settings, "MLOPS_FEATURE_STORE_S3_URI", ""),
            _PRODUCTION_FEATURE_STORE_S3_URI,
        ),
    }


def dashboard_home(request):
    patients = Patient.objects.all().order_by("patient_id")
    prediction_qs = EpochPrediction.objects.select_related("patient")
    model_status = _serving_status()
    display = _display_context(model_status)
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
            "model_status": model_status,
            **display,
        },
    )


@ensure_csrf_cookie
def patient_list(request):
    patients = (
        Patient.objects.annotate(
            epoch_count=Count("predictions"),
            avg_confidence=Avg("predictions__confidence"),
        )
        .order_by("patient_id")
    )
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


@ensure_csrf_cookie
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


@require_POST
def patient_delete(request, patient_id):
    patient = get_object_or_404(Patient, patient_id=patient_id)
    display_code = patient.display_patient_code
    raw_id = patient.patient_id
    epoch_count = patient.predictions.count()
    patient.delete()
    messages.success(
        request,
        f"Đã xóa hồ sơ {display_code} ({raw_id}) cùng {epoch_count} epoch liên quan.",
    )
    return redirect("patient_list")


def predict_page(request):
    """Dedicated inference studio — single feature vector, batch CSV, and EDF upload."""
    model_status = _serving_status()
    display = _display_context(model_status)
    feature_count = model_status.get("feature_count") or 24
    return render(
        request,
        "dashboard/predict.html",
        {
            "expected_feature_count": feature_count,
            "edf_sync_max_epochs": getattr(settings, "EDF_SYNC_MAX_EPOCHS", 96),
            "model_status": model_status,
            **display,
        },
    )


def pipeline_page(request):
    """Pipeline status — model registry, workflows, monitoring."""
    workflow_root = _find_workflow_root()
    model_status = _serving_status()
    display = _display_context(model_status)
    return render(
        request,
        "dashboard/pipeline.html",
        {
            "model_name": settings.MLFLOW_MODEL_NAME,
            "model_stage": settings.MLFLOW_MODEL_STAGE,
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "artifact_s3_uri": getattr(settings, "MODEL_ARTIFACT_S3_URI", ""),
            "feature_store_s3_uri": getattr(settings, "MLOPS_FEATURE_STORE_S3_URI", ""),
            "feature_store_local_dir": getattr(settings, "MLOPS_FEATURE_STORE_LOCAL_DIR", ""),
            "model_status": model_status,
            **display,
            "monitoring_ready": _workflow_ready(workflow_root, "monitoring.yml"),
            "retrain_ready": _workflow_ready(workflow_root, "retrain.yml"),
            "mlflow_ready": _workflow_ready(workflow_root, "mlflow.yml"),
            "ci_ready": _workflow_ready(workflow_root, "ci.yml"),
        },
    )
