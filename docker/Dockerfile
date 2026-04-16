# Multi-stage build để giảm image size
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages từ builder vào /opt/app-packages (tránh vấn đề permission với /root)
COPY --from=builder /root/.local /opt/app-packages
ENV PATH=/opt/app-packages/bin:$PATH
ENV PYTHONPATH=/opt/app-packages/lib/python3.11/site-packages

COPY sleep_portal/ .
COPY mlruns/ ./mlruns/
COPY models/ ./models/

ENV DJANGO_SETTINGS_MODULE=sleep_portal.settings.production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tạo non-root user để tăng bảo mật
RUN useradd --no-create-home --shell /bin/false appuser
RUN chown -R appuser:appuser /app /opt/app-packages

# collectstatic chạy trước khi switch user (dùng dummy secret key cho build time)
RUN DATABASE_URL=sqlite:////tmp/build.db python manage.py collectstatic --noinput

USER appuser

EXPOSE 8000

CMD ["gunicorn", "sleep_portal.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
