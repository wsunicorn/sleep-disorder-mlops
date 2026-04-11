from django.shortcuts import render, get_object_or_404
from .models import Patient, EpochPrediction


def dashboard_home(request):
    patients = Patient.objects.all().order_by("patient_id")
    diagnosis_counts = {}
    for p in patients:
        diagnosis_counts[p.diagnosis] = diagnosis_counts.get(p.diagnosis, 0) + 1
    return render(request, "dashboard/home.html", {
        "patients": patients,
        "diagnosis_counts": diagnosis_counts,
        "total_patients": patients.count(),
    })


def patient_list(request):
    patients = Patient.objects.all().order_by("patient_id")
    return render(request, "dashboard/patient_list.html", {"patients": patients})


def patient_detail(request, patient_id):
    patient = get_object_or_404(Patient, patient_id=patient_id)
    predictions = EpochPrediction.objects.filter(patient=patient).order_by("epoch_index")
    return render(request, "dashboard/patient_detail.html", {
        "patient": patient,
        "predictions": predictions,
        "n_epochs": predictions.count(),
    })
