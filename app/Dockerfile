# Dockerfile for the Movie Recommender serving API
# Optimized for Cloud Run — no data baked in, loads from GCS at startup
#
# Build:  docker build -t movierecommender-serve -f app/Dockerfile .
# Run locally (with local files):
#   docker run -p 8080:8080 -e PORT=8080 movierecommender-serve
# Run on Cloud Run (loads from GCS automatically via service account):
#   gcloud run deploy movierecommender-serve --image ...

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    GCS_BUCKET=mlops489-dvc-123456

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy serving app only — no data, no model
# Data and model are loaded from GCS at startup
COPY app/ app/

EXPOSE ${PORT}

# Cloud Run sets $PORT; uvicorn listens on it
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
