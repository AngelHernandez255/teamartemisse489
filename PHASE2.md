# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## Overview
Phase 2 focuses on scaling and operationalizing TeamArtemisSE489 by implementing containerization, advanced monitoring, profiling, experiment tracking, and comprehensive logging. This phase ensures your model can be reliably deployed, monitored in production, and continuously improved through systematic experimentation.

---

## 1. Containerization

### 1.1 Dockerfile

- [x] **Dockerfile Creation**: [`dockerfiles/Dockerfile`](dockerfiles/Dockerfile)
- [x] **Base Image**: `python:3.11-slim-bookworm` — the course-standard image, minimal footprint, no EOL risk
- [x] **Environment Variables**: `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `PIP_DISABLE_PIP_VERSION_CHECK=1`
- [x] **Build & Run documented in README**
- [x] **Docker Compose**: [`docker-compose.yaml`](docker-compose.yaml)
- [x] **Environment Consistency**: all Python dependencies pinned in `requirements.txt`

**How the image is structured:**

```
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir uv==0.5.30 && \
    uv pip install -r requirements.txt --system

COPY src/ scripts/ data/ models/ reports/ ./

RUN pip install --no-cache-dir . --no-deps

ENTRYPOINT ["python", "-u", "-m", "teamartemisse489.train_model"]
```

`uv pip install` is used instead of plain `pip` for ~10× faster dependency resolution during `docker build`. The `--system` flag installs into the container's system Python rather than a virtual environment. The entrypoint runs the package training module directly so Hydra config overrides can be passed as arguments after the image name.

### 1.2 Build & Run

**Prerequisites:** Install [Docker Desktop](https://docs.docker.com/get-docker/).

**Build the image:**

```bash
docker build -t teamartemisse489:latest .
# or via Make:
make docker_build
```

**Run training (models persist on the host via volume mount):**

```bash
# macOS / Linux
docker run -it --rm -v ${PWD}/models:/app/models teamartemisse489:latest

# Windows (cmd)
docker run -it --rm -v %cd%/models:/app/models teamartemisse489:latest

# or via Make:
make docker_run
```

**Pass Hydra config overrides:**

```bash
docker run -it --rm -v ${PWD}/models:/app/models teamartemisse489:latest \
    model.n_factors=200 training.n_epochs=30
```

**Docker Compose (mounts both data and models):**

```bash
docker compose up --build
```

**Keep monitoring logs on the host:**

```bash
docker run -it --rm \
    -v ${PWD}/models:/app/models \
    -v ${PWD}/logs:/app/logs \
    --entrypoint python teamartemisse489:latest \
    scripts/monitor_training.py --output logs/system_metrics.csv \
    -- python -m teamartemisse489.train_model
```

### 1.3 Environment Consistency

All runtime dependencies are pinned in [`requirements.txt`](requirements.txt) with exact version numbers (e.g., `numpy==1.26.4`, `pandas==2.2.2`, `scikit-surprise==1.1.4`). The same file is used both locally and inside the container, so `docker build` always resolves the identical dependency graph regardless of when or where it runs. The `pyproject.toml` declares the package metadata and entry points; `pip install . --no-deps` at the end of the build registers the `teamartemisse489` package without re-resolving anything already installed by `uv`.

---

## 2. Monitoring & Debugging

### 2.1 Monitoring

**Tool chosen: `psutil`-based CSV monitor** (`scripts/monitor_training.py` + `src/teamartemisse489/monitoring.py`)

We chose a lightweight `psutil` script over MLflow system metrics or Prometheus because it works identically locally and inside Docker without requiring a running server, and it produces a plain CSV that is easy to inspect in a spreadsheet or plot with pandas. W&B system metrics are already captured automatically during experiment tracking runs (see Section 4); the `psutil` monitor supplements that with a persistent local record tied to each training process.

**Key metrics captured (written to `logs/system_metrics.csv`):**

| Column | What it tells you |
|---|---|
| `system_cpu_percent` | Overall machine CPU load — spikes indicate compute-bound phases |
| `system_memory_percent` | System-wide RAM pressure |
| `system_memory_used_mb` | Absolute RAM consumption in MB |
| `process_memory_rss_mb` | Resident set size of the training process — watch for leaks |
| `process_cpu_percent` | CPU utilization of the training process alone |
| `process_threads` | Number of active threads (useful to verify parallelism) |
| `elapsed_seconds` | Wall-clock time since monitoring started |
| `phase` | `start` / `sample` / `end` — marks lifecycle events |

**Running the monitor:**

```bash
# Wrap the package entrypoint
python scripts/monitor_training.py --output logs/system_metrics.csv \
    -- python -m teamartemisse489.train_model

# Wrap the baseline train script directly
python scripts/monitor_training.py --output logs/system_metrics.csv \
    -- python models/train.py

# Adjust sample frequency (default: every 2 s)
python scripts/monitor_training.py --interval 5 --output logs/system_metrics.csv \
    -- python -m teamartemisse489.train_model
```

**Interpreting results** — load the CSV with pandas after a run:

```python
import pandas as pd
df = pd.read_csv("logs/system_metrics.csv")
df[["elapsed_seconds", "system_cpu_percent", "system_memory_used_mb",
    "process_memory_rss_mb"]].plot(x="elapsed_seconds")
```

High `system_cpu_percent` during SVD matrix factorization is expected. A steadily growing `process_memory_rss_mb` with no plateau would indicate a memory leak in the data loading pipeline.

**Implementation:** The `ResourceMonitor` class in [`src/teamartemisse489/monitoring.py`](src/teamartemisse489/monitoring.py) runs sampling in a daemon thread so it never blocks the training process. It opens the output CSV in append mode with a header guard, meaning multiple runs accumulate in the same file and can be compared post-hoc. The context manager interface (`with ResourceMonitor(...):`) guarantees a final `end` sample is written even if the training process raises an exception.

### 2.2 Debugging Practices

Debugging tools used: Python's built-in `pdb` (via `breakpoint()`), the `--debug` CLI flag in `models/train.py`, and explicit pre-training validation checks that fail fast with actionable error messages.

**Validation checks (in `models/train.py: validate_training_data`):**

Before the Surprise `Dataset` is constructed the trainer asserts:
1. The processed parquet file exists at the expected path.
2. The dataframe is non-empty.
3. All three required columns (`userId`, `movieId`, `rating`) are present.
4. None of the required columns contain null values.
5. The `rating` column is numeric.
6. All ratings are in the valid `[1, 5]` range.

Each check raises a specific `ValueError` or `TypeError` with a message that names the problem directly, so failures are self-diagnosing without needing a debugger.

**Using the `--debug` flag:**

```bash
python models/train.py --debug
```

This calls `breakpoint()` immediately after validation passes, dropping into `pdb` while `df` is in scope:

```
(Pdb) p df.shape
(10000, 3)
(Pdb) p df.dtypes
userId      int64
movieId     int64
rating    float64
dtype: object
(Pdb) p df["rating"].describe()
count    10000.000000
mean         3.541200
...
(Pdb) c   # continue to training
```

**Debugging inside Docker:**

```bash
docker run -it --rm \
    -v ${PWD}/models:/app/models \
    --entrypoint python teamartemisse489:latest \
    models/train.py --debug
```

The `-it` flags keep stdin open so `pdb` can accept keystrokes interactively.

**Debug Scenario 1 — missing required columns:**

*Symptom:* `ValueError: Training data is missing columns: ['rating']`

*Cause:* The preprocessing step wrote a parquet file with a column named `score` instead of `rating`.

*Resolution:* Inspect `df.columns` at the `pdb` prompt, then fix the rename in `src/teamartemisse489/data/make_dataset.py`. The validation check surfaces this immediately rather than letting it propagate to a cryptic Surprise internal error.

**Debug Scenario 2 — ratings outside the 1–5 range:**

*Symptom:* `ValueError: Training data ratings must be between 1 and 5. Example invalid values: [0.5, 6.0, ...]`

*Cause:* The raw Rotten Tomatoes dataset uses a 0–10 scale for some review sources; a normalization step was applied inconsistently.

*Resolution:* At the `pdb` prompt, run `p df.loc[~df["rating"].between(1, 5), "rating"].value_counts()` to see the distribution of out-of-range values, then update the normalization logic in the data pipeline. The `validate_training_data` function prints up to five example invalid values to guide the fix without requiring interactive debugging at all.

---

## 3. Profiling & Optimization

- [ ] **CPU Profiling**: Use cProfile to profile training and inference
- [ ] **Memory Profiling**: Profile memory usage with memory_profiler or similar
- [ ] **GPU Profiling (if applicable)**: Use PyTorch Profiler or similar for GPU workloads
- [ ] **Profiling Results**: Document baseline profiling results and bottlenecks identified
- [ ] **Optimization 1**: Implement and measure optimization (e.g., vectorization, caching)
- [ ] **Optimization 2**: Implement and measure additional optimization
- [ ] **Performance Benchmarks**: Document before/after performance metrics
- [ ] **Optimization Documentation**: Explain each optimization and its impact

---

## 4. Experiment Management & Tracking

- [ ] **MLflow Setup**: Initialize MLflow tracking server and client configuration
  - OR **Weights & Biases Setup**: Initialize W&B project and team workspace
- [ ] **Metric Logging**: Log training/validation metrics for each experiment
- [ ] **Parameter Logging**: Log all hyperparameters and configuration values
- [ ] **Model Artifact Logging**: Save model checkpoints and artifacts to tracking system
- [ ] **Experiment Comparison**: Create comparison of at least 3 different experiments
- [ ] **Visualization**: Generate performance comparison charts/plots
- [ ] **Best Model Selection**: Document criteria and process for selecting best model from experiments
- [ ] **Experiment Documentation**: Create table summarizing all experiments with results

---

## 5. Application & Experiment Logging

- [x] **Logger Setup**: Configure Python logger with appropriate handlers and formatters
  - OR **Rich Library Setup**: Use rich for enhanced console output and logging

used rich within the [logging config](src/teamartemisse489/logging_config.py) file 

- [x] **Log Levels**: Implement and use DEBUG, INFO, WARNING, ERROR appropriately

<img width="1059" height="153" alt="image" src="https://github.com/user-attachments/assets/ab670bf6-ecb3-499d-8c67-766f3a0becb9" />

Each of these files uses logs in some way, shape, or form.

 [monitor_training](scripts/monitor_training.py)
 [make_dataset](src/teamartemisse489/make_dataset.py)
 [predict_model](src/teamartemisse489/predict_model.py)
 [train_model](src/teamartemisse489/train_model.py)
 
- [x] **Log Messages**: Add informative log messages at key points in code

Most logging data is stored inside the logs directory in src/teamartemisse489/logs
Sometimes the log files get ignored, regardless of the gitkeep or removing .logs from gitignore. 
whether or not we are supposed to include them or not due to their potential size (10 mb each with up to 8 backups for previous versions)
 
- [x] **Training Log Example**: Document and include sample training log output
- [x] **Inference Log Example**: Document and include sample inference log output
- [x] **Error Logging**: Implement comprehensive error logging with context

Examples of error logging are found in  [monitor_training](scripts/monitor_training.py)
...
    if not command:
        logger.error(
            "No command provided. Example: "
            "python scripts/monitor_training.py -- python models/train.py"
        )
        return 2
...

- [x] **Performance Logging**: Log timing information for performance analysis

performance was logged in logs/system_metrics.csv rather than in the regular logs.

- [x] **Log Rotation**: Configure log rotation to prevent disk space issues

The log rotation is set at [logging config](src/teamartemisse489/logging_config.py) in the JSON formatter, declared at line 20. Each one has 10 mb of max space, and up to 8 are backed up.

---

## 6. Configuration Management

- [x] **Hydra Setup**: Install and configure Hydra for config management
- [x] **Config Files**: Create YAML config files for train/eval/inference configurations
- [x] **Config Structure**: Organize configs with appropriate hierarchy (base, model, data, etc.)
- [ ] **Config Example 1**: Create and document sample training config
- [x] **Config Example 2**: Create and document alternative config (different hyperparameters)
- [x] **Config Validation**: Implement config validation and schema checking
- [x] **Override Documentation**: Document how to override config values from command line
- [x] **Config Version Control**: Version all configs alongside code

---

## 7. Documentation & Repository Updates

- [x] **README Update**: Update README to include:
  - [ ] Containerization section with Docker usage
  - [ ] Debugging and profiling guide
  - [ ] Experiment tracking setup instructions
  - [x] Configuration management guide
  - [ ] Logging usage examples
- [ ] **Architecture Documentation**: Document system architecture with diagrams
- [ ] **Setup Guide**: Update setup guide to include all Phase 2 tools
- [ ] **Examples**: Add examples of running with different configurations
- [ ] **Tool Integration**: Document how all tools work together
- [ ] **Troubleshooting**: Add troubleshooting section for common issues
- [ ] **Performance Guide**: Document how to profile and optimize
- [ ] **Version Compatibility**: Document version requirements for all tools

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables.
