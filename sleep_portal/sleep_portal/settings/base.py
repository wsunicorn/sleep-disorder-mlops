"""
Django settings — Base configuration (dùng cho cả dev và production).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "insecure-dev-key-change-in-prod")

DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "dashboard",
    "api",
    "inference",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "sleep_portal.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "sleep_portal.wsgi.application"

# Database — có thể override trong development.py / production.py
import dj_database_url
DATABASES = {
    "default": dj_database_url.config(
        default=os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR / 'db.sqlite3'}"),
        conn_max_age=600,
        ssl_require=False,
    )
}

# Cache — có thể override trong development.py
REDIS_URL = os.environ.get("REDIS_URL", "")
if REDIS_URL:
    CACHES = {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": REDIS_URL,
            "OPTIONS": {"CLIENT_CLASS": "django_redis.client.DefaultClient"},
            "TIMEOUT": 3600,
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Django REST Framework
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": os.environ.get("DRF_ANON_THROTTLE_RATE", "5000/day"),
        "user": os.environ.get("DRF_USER_THROTTLE_RATE", "10000/day"),
    },
}

# CORS
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

# MLflow
PUBLIC_APP_URL = os.environ.get(
    "PUBLIC_APP_URL",
    "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com",
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "sleep-disorder-classifier")
MLFLOW_MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_UI_URL = os.environ.get(
    "MLFLOW_UI_URL",
    MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI.startswith(("http://", "https://")) else f"{PUBLIC_APP_URL}:5000",
)
_DEFAULT_MODEL_ARTIFACT_DIR = (
    BASE_DIR / "models"
    if (BASE_DIR / "models").exists()
    else BASE_DIR.parent / "models"
)
MODEL_ARTIFACT_S3_URI = os.environ.get("MODEL_ARTIFACT_S3_URI", "")
MODEL_ARTIFACT_LOCAL_DIR = os.environ.get(
    "MODEL_ARTIFACT_LOCAL_DIR",
    str(_DEFAULT_MODEL_ARTIFACT_DIR),
)

# AWS
AWS_S3_BUCKET = os.environ.get("S3_BUCKET", "sleep-mlops-data")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1")

# MLOps feature store for ingested data batches.
MLOPS_FEATURE_STORE_LOCAL_DIR = os.environ.get(
    "MLOPS_FEATURE_STORE_LOCAL_DIR",
    str(BASE_DIR.parent / "data" / "monitoring" / "current")
    if os.access(BASE_DIR.parent, os.W_OK)
    else "/tmp/sleep-portal/monitoring/current",
)
MLOPS_FEATURE_STORE_S3_URI = os.environ.get("MLOPS_FEATURE_STORE_S3_URI", "")

# Synchronous EDF uploads run behind an ALB, so keep the interactive request
# bounded. Full-night processing should use the IoT/batch pipeline.
EDF_SYNC_MAX_EPOCHS = max(1, int(os.environ.get("EDF_SYNC_MAX_EPOCHS", "96")))

# Security headers (production)
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
