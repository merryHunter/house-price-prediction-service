# Use Python 3.11 slim as base image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./app/

RUN pip install --no-cache-dir -r app/requirements.txt

COPY app/service.py ./app/app/service.py
COPY app/dev.env ./app/app/.env
COPY ml/models/price_model_2025-07-02-12-50-34.joblib ./ml/models/price_model_2025-07-02-12-50-34.joblib
COPY ml/features.py ./ml/features.py
COPY ml/train.py ./ml/train.py

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.app.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 