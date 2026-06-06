# PHASE 3: Continuous Machine Learning (CML) & Deployment

## Overview
Phase 3 implements continuous integration/continuous deployment (CI/CD) pipelines and productionizes TeamArtemisSE489 on cloud infrastructure. This phase covers automated testing, containerized workflows, CML integration, and multi-platform deployment options including GCP, Cloud Run, and serverless functions.

---

## 1. Continuous Integration & Testing

- [ ] **Unit Tests**: Write pytest test scripts for data processing and model components
- [ ] **Integration Tests**: Create integration tests for full training pipeline
- [ ] **Test Coverage**: Aim for >80% code coverage with pytest-cov
- [ ] **GitHub Actions - Tests**: Create workflow for running tests on every push
  - [ ] Trigger on: push to main/develop branches and PRs
  - [ ] Test across multiple Python versions if applicable
  - [ ] Report coverage metrics
- [ ] **GitHub Actions - Code Quality**: Create workflow for:
  - [ ] Running ruff linter
  - [ ] Type checking with mypy
  - [ ] Formatting checks
- [ ] **GitHub Actions - Docker Build**: Create workflow for building Docker image
  - [ ] Build on PR and main branch push
  - [ ] Test built image
- [ ] **Pre-commit Hooks**: Set up pre-commit hooks for:
  - [ ] Formatting (black/ruff)
  - [ ] Linting
  - [ ] Type checking
  - [ ] Trailing whitespace
- [ ] **Test Documentation**: Document how to run tests locally and in CI

---

## 2. Continuous Docker Building & CML

- [x] **Automated Docker Builds**: `.github/workflows/docker-publish.yml`
  builds the existing `dockerfiles/Dockerfile` on pull requests, pushes to
  Docker Hub on commits to `main`, and also supports manual runs through
  `workflow_dispatch`. The workflow uses Docker Buildx plus the official Docker
  login, metadata, and build-push actions so every published image receives a
  commit SHA tag and the default branch receives `latest`.
  - Required GitHub Actions secrets: `DOCKER_HUB_USERNAME`,
    `DOCKER_HUB_TOKEN`, and `DOCKER_HUB_REPOSITORY`.
  - Evidence still needed after pushing: screenshot of the green workflow run
    and screenshot of the pushed image in Docker Hub.
- [x] **Docker Push**: The build-push step publishes only on `push` to `main`
  or manual workflow dispatch, while PRs build the image without publishing.
  This prevents feature branches from overwriting registry tags while still
  proving that the Dockerfile compiles before merge.
- [x] **CML Workflow**: `.github/workflows/cml.yml` runs on pushes, pull
  requests, and manual dispatch. It installs Python 3.11 with `uv`, sets up CML
  with `iterative/setup-cml@v2`, runs `scripts/run_cml_report.py`, and posts the
  generated Markdown report back to the PR with
  `cml comment create --publish report.md`.
- [x] **CML Metrics Output**: `scripts/run_cml_report.py` creates a
  deterministic CI-safe ratings dataset, calls the repository's SVD training
  function, and writes `reports/cml/metrics.md` plus
  `reports/cml/metrics.json`. The reported metrics are RMSE, MAE,
  precision@k, recall@k, and training time.
- [x] **CML Plots**: The same CML script writes `reports/cml/metrics.png`, a
  compact bar chart that CML publishes and embeds in the PR comment.
  - Evidence still needed after opening a PR: screenshot of the
    `github-actions[bot]` CML comment showing the metrics table and plot.

---

## 3. Deployment on GCP

- [x] **GCP Project Setup**: GCP project `mlops-recommenderproject` (ID: `682507623900`)
  was created and all required APIs were enabled including `artifactregistry.googleapis.com`,
  `cloudbuild.googleapis.com`, and `aiplatform.googleapis.com`. The project uses
  `us-central1` as the default region throughout to keep all resources co-located
  and minimize egress costs.
- [x] **Service Account**: Cloud Build uses the default Cloud Build service account
  (`682507623900@cloudbuild.gserviceaccount.com`) which has Artifact Registry Writer
  permissions. Vertex AI custom jobs run under the default Agent Platform service agent,
  which has access to the GCS bucket via `--scopes=cloud-platform`.
  - [x] Artifact Registry — Cloud Build service account has `roles/artifactregistry.writer`
  - [x] Vertex AI — Agent Platform service agent has access to GCS bucket
  - [ ] Cloud Run — permissions to be configured in section 3.3
  - [ ] Cloud Functions — not used (Cloud Run chosen as deployment option)
  - [x] Compute Engine — `--scopes=cloud-platform` used when VM was created
- [x] **Artifact Registry**: Docker repository `mlops489-docker` created in
  `us-central1`. The training image `teamartemisse489-train` is built automatically
  by Cloud Build on every push to `main` via the `mlops-trigger` trigger, and can
  also be built manually with `gcloud builds submit`. See `cloudbuild.yaml` at the
  repo root for the full build spec.
  - [x] **Create repository in Artifact Registry**: `mlops489-docker` repository
    created at `us-central1-docker.pkg.dev/mlops-recommenderproject/mlops489-docker`
  - [x] **Configure authentication from CI/CD**: Cloud Build authenticates to
    Artifact Registry automatically via the attached service account; local Docker
    auth configured with `gcloud auth configure-docker us-central1-docker.pkg.dev`
  - [x] **Push Docker images to registry**: Image
    `teamartemisse489-train:v1` (3.1 GB) successfully pushed. Latest image digest:
    `sha256:f8243a54...`. See Cloud Build history screenshot below.
  ![Cloud Build History](docs/screenshots/cloud-build-history.png)
  ![Artifact Registry](docs/screenshots/Artifact-Registry.png)
- [x] **Vertex AI Training (Option A)**: Custom training job submitted to Vertex AI
  using the `teamartemisse489-train:v1` container image. Training reads the
  1M-rating MovieLens dataset directly from GCS via the automatic `/gcs` mount
  (no `dvc pull` inside the container), trains an SVD collaborative filtering model,
  and writes the trained artifact back to GCS. Job spec is in `config_cpu.yaml`
  at the repo root.
  - [x] **Create training container image**: `dockerfiles/Dockerfile` builds a
    `python:3.11-slim-bookworm` image with all dependencies from `requirements.txt`,
    source code from `src/`, and Hydra configs from `configs/`.
  - [x] **Configure training job specification**: `config_cpu.yaml` specifies
    `n1-standard-4` (4 vCPUs / 15 GB RAM), 1 replica, and passes GCS paths as
    Hydra overrides via `containerSpec.args`:
  - [x] **Document how to submit training jobs**:
    ```bash
    gcloud ai custom-jobs create \
        --region=us-central1 \
        --display-name=mlops489-train \
        --config=config_cpu.yaml
    # Stream logs
    gcloud ai custom-jobs stream-logs <job-id> --region=us-central1
    ```
    Job `1276662175284330496` (`mlops489-train-v4`) completed with
    `JOB_STATE_SUCCEEDED` in 1 min 30 sec.
  ![Vertex AI Custom Jobs](docs/screenshots/vertex-AI-custom-jobs.png)
- [ ] **Compute Engine Training (Option B)**: Not used — Vertex AI (Option A) was
  chosen as the training platform. A Debian 12 VM (`mlops489-train`, `n1-standard-4`,
  `us-central1-a`) was created and Docker was installed during exploration, but
  training was ultimately run on Vertex AI. 
- [x] **Model Registry**: Trained model artifacts are stored in the GCS bucket
  `gs://mlops489-dvc-123456/models/`. The bucket already existed as the DVC
  remote store; a separate `models/` prefix is used for trained artifacts so
  DVC-managed data and model outputs stay cleanly separated.
  - [x] **Create GCS bucket for models**: `gs://mlops489-dvc-123456/models/`
    prefix used within the existing DVC bucket
      **GCS Bucket — `data/processed/` showing uploaded training data:**


    ![GCS Data](docs/screenshots/Bucket-data.png)
    
    **GCS Bucket — `models/` showing saved model artifact `svd.joblib`:**

    ![GCS Model Artifact](docs/screenshots/GCP-model.png)

  - [x] **Implement model upload from training**: The training script saves the
    fitted SVD model via `joblib.dump` to the path specified by `cfg.paths.model_dir`.
    When run on Vertex AI with `paths.model_dir=/gcs/mlops489-dvc-123456/models`,
    the file is written directly to GCS:
    ```bash
    gsutil ls gs://mlops489-dvc-123456/models/
    # gs://mlops489-dvc-123456/models/svd.joblib
    ```
  - [x] **Document model retrieval**:
    ```bash
    # Download model locally
    gsutil cp gs://mlops489-dvc-123456/models/svd.joblib models/svd.joblib
    # Or load directly in Python
    import joblib
    model = joblib.load("svd.joblib")
    ```

- [ ] **FastAPI Service**: Create FastAPI application for model serving
  - [ ] Define inference endpoint(s)
  - [ ] Implement request validation
  - [ ] Add health check endpoint
  - [ ] Document API specification
- [ ] **Cloud Functions Deployment (Option A)**: Deploy inference as Cloud Function
  - [ ] Package model and FastAPI app for Cloud Functions
  - [ ] Create Cloud Function with appropriate memory/timeout
  - [ ] Configure HTTP trigger
  - [ ] Document invocation and response format
- [ ] **Cloud Run Deployment (Option B)**: Deploy as containerized service on Cloud Run
  - [ ] Create Dockerfile optimized for Cloud Run
  - [ ] Test locally with Cloud Run emulator
  - [ ] Deploy to Cloud Run with auto-scaling
  - [ ] Document deployment process
- [ ] **Streamlit/Gradio Deployment (Option C)**: Deploy demo app on HuggingFace Spaces
  - [ ] Create Streamlit or Gradio interface for model
  - [ ] Push to GitHub repository
  - [ ] Deploy to HuggingFace Spaces
  - [ ] Document feature walkthrough
- [ ] **Load Testing**: Test deployment with load testing tool (locust, Apache JMeter)
  - [ ] Establish baseline performance metrics
  - [ ] Document scaling characteristics
- [ ] **Monitoring Setup**: Configure Cloud Monitoring and Cloud Logging
  - [ ] Set up log aggregation
  - [ ] Create monitoring dashboards
  - [ ] Set up alerts for anomalies

---

## 4. Documentation & Repository Updates

- [ ] **Comprehensive README**: Update README with:
  - [ ] Architecture diagram showing all components
  - [ ] CI/CD pipeline overview
  - [ ] Deployment instructions for each option (Cloud Run, Cloud Functions, HuggingFace)
  - [ ] GCP setup and configuration guide
  - [ ] How to invoke deployed models
  - [ ] Monitoring and troubleshooting guide
  - [ ] Cost estimation and optimization tips
- [ ] **Deployment Guide**: Create detailed DEPLOYMENT.md with:
  - [ ] Step-by-step GCP setup instructions
  - [ ] Cloud Run deployment procedure
  - [ ] Cloud Functions configuration
  - [ ] Environment variables and secrets management
  - [ ] Rollback procedures
- [ ] **API Documentation**: Document all endpoints with:
  - [ ] Request/response schemas
  - [ ] Example curl/Python requests
  - [ ] Error codes and messages
- [ ] **Architecture Documentation**: Include diagrams showing:
  - [ ] Data pipeline
  - [ ] Training pipeline
  - [ ] Inference/serving architecture
  - [ ] CI/CD workflow
- [ ] **Screenshots/Demos**: Add:
  - [ ] Cloud Run dashboard screenshot
  - [ ] Monitoring dashboard screenshot
  - [ ] Streamlit/Gradio app screenshot
  - [ ] API response example
  - [ ] CML workflow output sample
- [ ] **Troubleshooting Guide**: Document solutions for:
  - [ ] Common deployment errors
  - [ ] Authentication issues
  - [ ] Performance problems
  - [ ] Cost overruns
- [ ] **Resource Cleanup Reminder**: Create CLEANUP.md with instructions for:
  - [ ] Deleting GCP resources (VMs, databases, etc.)
  - [ ] Cleaning up Cloud Storage buckets
  - [ ] Disabling APIs to avoid charges
  - [ ] Cost monitoring recommendations
- [ ] **Contributing Guide Update**: Update CONTRIBUTING.md with:
  - [ ] CI/CD requirements
  - [ ] Testing requirements for PRs
  - [ ] Deployment process documentation
- [ ] **Changelog**: Maintain CHANGELOG.md documenting releases and deployments

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables.
