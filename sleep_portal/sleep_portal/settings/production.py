from .base import *  # noqa

DEBUG = False

# Production: ALLOWED_HOSTS từ env var
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

# ALB terminates SSL — disable Django's own SSL redirect to avoid 301 on health checks
SECURE_SSL_REDIRECT = False
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# MLflow production config
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "sleep-disorder-classifier")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "None")
