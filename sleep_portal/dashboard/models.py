from django.db import models


_DIAGNOSIS_CODE = {
    "healthy": "HEA",
    "insomnia": "INS",
    "narcolepsy": "NAR",
    "nfle": "NFL",
    "plm": "PLM",
    "rbd": "RBD",
    "sdb": "SDB",
    "monitoring_case": "MON",
}


class Patient(models.Model):
    patient_id = models.CharField(max_length=50, unique=True)
    diagnosis = models.CharField(max_length=100)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def display_patient_code(self) -> str:
        """Stable presentation code; the original patient_id stays untouched."""
        raw = self.patient_id.strip()
        normalized = raw.lower()
        diag_code = _DIAGNOSIS_CODE.get(self.diagnosis.lower(), "UNK")

        if normalized.startswith("demo-rich-"):
            serial = raw.rsplit("-", 1)[-1].zfill(2)
            return f"DEMO-RICH-{diag_code}-{serial}"
        if normalized.startswith("demo-iot-"):
            serial = raw.rsplit("-", 1)[-1].zfill(3)
            return f"DEMO-IOT-{diag_code}-{serial}"
        if normalized.startswith("pt-"):
            return f"CAP-{raw.upper()}"
        return raw.upper()

    @property
    def source_label(self) -> str:
        normalized = self.patient_id.lower()
        if normalized.startswith("demo-rich-"):
            return "Demo IoT đa bệnh nhân"
        if normalized.startswith("demo-iot-"):
            return "Demo IoT nhanh"
        if normalized.startswith("pt-"):
            return "Dữ liệu mẫu CAP"
        return "Ingest/API"

    def __str__(self):
        return f"{self.patient_id} ({self.diagnosis})"


class EpochPrediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="predictions")
    epoch_index = models.IntegerField()
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("patient", "epoch_index")
        ordering = ["epoch_index"]

    def __str__(self):
        return f"{self.patient.patient_id} epoch {self.epoch_index}: {self.predicted_class}"
