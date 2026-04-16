"""
Promotion rules cho model sau retraining.
"""

import mlflow


def evaluate_run_and_promote(
    run_id: str,
    tracking_uri: str,
    model_name: str,
    threshold: float,
    stage: str = "Staging",
) -> tuple[bool, str]:
    """
    Evaluate metric val_f1_weighted của run và promote model version mới nhất nếu đạt ngưỡng.

    Returns:
        (promoted, details)
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    f1 = run.data.metrics.get("val_f1_weighted", 0.0)
    if f1 < threshold:
        return False, (
            f"Model not promoted: val_f1_weighted={f1:.4f} below threshold={threshold:.4f}"
        )

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return False, f"No model versions found for model {model_name}"

    # Chọn version mới nhất theo version number.
    latest_version = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage=stage,
    )

    return True, (
        f"Promoted {model_name} version {latest_version.version} to {stage} "
        f"(val_f1_weighted={f1:.4f})"
    )
