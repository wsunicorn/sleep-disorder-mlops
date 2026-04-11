from django.db import models


class Patient(models.Model):
    patient_id = models.CharField(max_length=50, unique=True)
    diagnosis = models.CharField(max_length=100)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

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
