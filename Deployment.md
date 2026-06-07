# Deployment Guide — TeamArtemisSE489 Movie Recommender

This guide covers the full GCP deployment pipeline for the movie recommender
system, from initial setup to a live Cloud Run endpoint.

---

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Docker Desktop installed and running
- Access to GCP project `mlops-recommenderproject`
- Python 3.11+ with the project dependencies installed

```bash
gcloud auth login
gcloud config set project mlops-recommenderproject
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## 1. GCP Project Setup (one-time)

```bash
# Enable required APIs
gcloud services enable \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    run.googleapis.com \
    compute.googleapis.com

# Verify
gcloud services list --enabled | grep -E "artifactregistry|cloudbuild|aiplatform|run"
```

---

## 2. Artifact Registry (one-time)

```bash
# Create the Docker repository
gcloud artifacts repositories create mlops489-docker \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker images for MLOps movie recommender"

# Verify
gcloud artifacts repositories list --location=us-central1
```

---

## 3. Training Data Setup

Training data is versioned with DVC and stored in GCS. Upload to a plain readable path for Vertex AI:

```bash
# Pull data locally first
dvc pull

# Upload to plain GCS path (readable by Vertex AI /gcs mount)
gsutil cp data/processed/ready_to_train_1M.parquet \
    gs://mlops489-dvc-123456/data/processed/ready_to_train_1M.parquet

gsutil cp data/raw/movies.parquet \
    gs://mlops489-dvc-123456/data/raw/movies.parquet
```

---

## 4. Build & Push Training Image

```bash
# Build and push via Cloud Build (recommended)
gcloud builds submit . --config=cloudbuild.yaml --substitutions=_TAG=v1

# Or manually
docker build -f dockerfiles/Dockerfile -t \
    us-central1-docker.pkg.dev/mlops-recommenderproject/mlops489-docker/teamartemisse489-train:v1 .
docker push \
    us-central1-docker.pkg.dev/mlops-recommenderproject/mlops489-docker/teamartemisse489-train:v1
```

---

## 5. Run Training on Vertex AI

```bash
# Submit the custom job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=mlops489-train \
    --config=config_cpu.yaml

# Stream logs (blocks until job ends)
gcloud ai custom-jobs stream-logs <job-id> --region=us-central1

# Verify model artifact saved
gsutil ls gs://mlops489-dvc-123456/models/
# gs://mlops489-dvc-123456/models/svd.joblib
```

**Job spec** (`config_cpu.yaml`):
```yaml
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-4
    replicaCount: 1
    containerSpec:
      imageUri: us-central1-docker.pkg.dev/mlops-recommenderproject/mlops489-docker/teamartemisse489-train:v1
      args:
        - data.data_path=/gcs/mlops489-dvc-123456/data/processed/ready_to_train_1M.parquet
        - paths.model_dir=/gcs/mlops489-dvc-123456/models
```

**Clean up** — finished jobs are free; only running jobs bill:
```bash
gcloud ai custom-jobs list --region=us-central1 \
    --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING"
```

---

## 6. Build & Push Serving Image

The serving image is built for `linux/amd64` (required by Cloud Run)

```bash
docker buildx build \
    --platform linux/amd64 \
    -f app/Dockerfile \
    -t us-central1-docker.pkg.dev/mlops-recommenderproject/mlops489-docker/movierecommender-serve:latest \
    --push \
    .
```

---

## 7. Deploy to Cloud Run

```bash
gcloud run deploy movierecommender-serve \
    --image us-central1-docker.pkg.dev/mlops-recommenderproject/mlops489-docker/movierecommender-serve:latest \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars GCS_BUCKET=mlops489-dvc-123456

# Verify
gcloud run services list
gcloud run services describe movierecommender-serve --region us-central1
```

**Service configuration:**

| Field | Value |
|---|---|
| Service name | `movierecommender-serve` |
| Region | `us-central1` |
| URL | `https://movierecommender-serve-682507623900.us-central1.run.app` |
| Memory | 4 GiB |
| CPU | 2 vCPU |
| Timeout | 300 seconds |
| Concurrency | 80 requests/instance |
| Max instances | 3 |
| Scaling | Auto (min 0, max 3) |
| Billing | Request-based |

---

## 8. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GCS_BUCKET` | `mlops489-dvc-123456` | GCS bucket containing model and data |
| `PORT` | `8080` | Port Cloud Run listens on (set automatically) |
| `LOCAL_MODEL_PATH` | `models/svd.joblib` | Local model path fallback (for local dev) |
| `LOCAL_MOVIES_PATH` | `data/raw/movies.parquet` | Local movies path fallback |
| `LOCAL_RATINGS_PATH` | `data/processed/ready_to_train_1M.parquet` | Local ratings path fallback |

To update environment variables without rebuilding:
```bash
gcloud run services update movierecommender-serve \
    --region us-central1 \
    --set-env-vars GCS_BUCKET=new-bucket-name
```

---

## 9. Continuous Deployment

Every push to `main` automatically rebuilds both images and redeploys via the `mlops-trigger` Cloud Build trigger. The full pipeline in `cloudbuild.yaml`:

```
git push origin main
    → Cloud Build trigger fires
    → Build teamartemisse489-train:v1
    → Push to Artifact Registry
    → Build movierecommender-serve:latest
    → Push to Artifact Registry
    → gcloud run deploy (new revision live)
```

To manually trigger a build without a push:
```bash
gcloud builds submit . --config=cloudbuild.yaml --substitutions=_TAG=v1
```

---

## 10. Rollback

Cloud Run keeps all revisions. To roll back to a previous revision:

```bash
# List revisions
gcloud run revisions list --service movierecommender-serve --region us-central1

# Roll back to a specific revision
gcloud run services update-traffic movierecommender-serve \
    --region us-central1 \
    --to-revisions movierecommender-serve-00001-8qr=100
```

---

## 11. Resource Cleanup

**Important: always clean up to avoid unexpected billing.**

```bash
# Delete Cloud Run service
gcloud run services delete movierecommender-serve --region us-central1 --quiet

# Cancel any running Vertex AI jobs
gcloud ai custom-jobs list --region=us-central1 \
    --filter="state:JOB_STATE_RUNNING"
gcloud ai custom-jobs cancel <job-id> --region=us-central1

# Delete VMs if any
gcloud compute instances list
gcloud compute instances delete <instance-name> --zone=us-central1-a --quiet

# GCS bucket (keep if you want to preserve the model)
# gsutil rm -r gs://mlops489-dvc-123456/models/
```

---

## 12. Troubleshooting

**Container failed to start:**
- Check logs: `gcloud logging read 'resource.type="cloud_run_revision"' --limit=50`
- Common cause: wrong port — Cloud Run expects `$PORT` (8080 by default)
- Common cause: image built for ARM64 — rebuild with `--platform linux/amd64`

**Out of memory:**
- Increase memory: `gcloud run services update movierecommender-serve --memory 8Gi --region us-central1`

**Slow cold start:**
The first request may experience additional latency because the application downloads model artifacts from GCS during startup.

**Model not found in GCS:**
```bash
gsutil ls gs://mlops489-dvc-123456/models/
# If empty, re-run training job
```

**Authentication errors:**
```bash
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
```
