from .base import *  # noqa

DEBUG = True
ALLOWED_HOSTS = ["*"]

# SQLite cho local dev (không cần RDS)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Cache local (memory) — không cần Redis
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

# Tắt SSL redirect khi dev
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# MLflow local
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_MODEL_NAME = "sleep-disorder-classifier"
MLFLOW_MODEL_STAGE = "None"  # local chưa promote lên Staging/Production
