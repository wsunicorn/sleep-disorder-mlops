from .base import *  # noqa

DEBUG = False

# Production: ALLOWED_HOSTS từ env var
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

# ALB terminates SSL — disable Django's own SSL redirect to avoid 301 on health checks
SECURE_SSL_REDIRECT = False
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# MLflow production config
PUBLIC_APP_URL = os.environ.get(
    "PUBLIC_APP_URL",
    "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com",
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"{PUBLIC_APP_URL}:5000")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "sleep-disorder-classifier")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", MLFLOW_TRACKING_URI)
MODEL_ARTIFACT_S3_URI = os.environ.get("MODEL_ARTIFACT_S3_URI", "s3://sleep-mlops-651709/models")
MLOPS_FEATURE_STORE_S3_URI = os.environ.get(
    "MLOPS_FEATURE_STORE_S3_URI",
    "s3://sleep-mlops-651709/monitoring/current",
)
