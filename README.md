# Artemis Movie Recommender

## Team Information

### TeamArtemisSE489
- Anjan Kumar Basavaraj Gurudatt (ABASAVAR@depaul.edu)
- Joshua Nevin Chandrasekar (JCHANDR2@depaul.edu)
- Sakshi Gorkhali (SGORKHAL@depaul.edu)
- Angel Hernandez (AHERN255@depaul.edu)
  
- SE 456

## Project Overview

We are TeamArtemisSE489, a team for the class Global Software Development, and we are working on a machine learning project that uses machine learning to recommend movies. The data set consists of 56 million user reviews of 10500 Movies. The project will use this dataset to provide recommendations to users on what movie to see next. 

"PROBLEM STATEMENT"

**Key Objectives:**
- [x] Objective 1
Build a scalable movie recommendation system using collaborative filtering techniques to generate personalized movie suggestions based on user ratings and interaction patterns.

- [x] Objective 2
Develop a reproducible MLOps pipeline that automates data preprocessing, feature engineering, model training, evaluation, and model persistence using industry-standard engineering practices.

- [ ] Objective 3
Establish a maintainable and collaborative machine learning workflow with version control, code quality checks, documentation, and modular project structure to support future deployment and continuous improvement.

## Architecture Diagram
![Project Architecture](achitecturediagramMlops.png)

## Phase Deliverables

### Phase 1: Project Design & Model Development
- See [PHASE1.md](PHASE1.md) for detailed checklist

### Phase 2: Containerization & Monitoring
- See [PHASE2.md](PHASE2.md) for detailed checklist

#### Containerization
This phase adds a project Docker image based on `python:3.11-slim-bookworm`. The image installs pinned Python dependencies with `uv`, installs the local `teamartemisse489` package, and runs the training entrypoint by default.

Install Docker Desktop from the official Docker documentation before running these commands: https://docs.docker.com/get-docker/

Build the image:
```bash
docker build -t teamartemisse489:latest .
```

Run training with the host `models/` folder mounted into the container so trained artifacts persist after the container exits:
```bash
docker run -it --rm -v ${PWD}/models:/app/models teamartemisse489:latest
```

Pass training options after the image name when needed:
```bash
docker run -it --rm -v ${PWD}/models:/app/models teamartemisse489:latest --epochs 5 --batch-size 64 --learning-rate 0.001
```

Optional Docker Compose run:
```bash
docker compose up --build
```

The Dockerfile was adapted from the Week 4 Docker exercise files with the package paths updated for this repository.

#### Monitoring
For Phase 2 monitoring, this project uses a lightweight `psutil` CSV monitor. It records CPU usage, RAM usage, process memory, process CPU, thread count, timestamps, and elapsed time while the training command runs. We chose this option because it works locally and inside Docker without requiring a separate dashboard service.

Run the Surprise SVD trainer with monitoring:
```bash
python scripts/monitor_training.py --output logs/system_metrics.csv -- python models/train.py
```

Run the package training entrypoint with monitoring:
```bash
python scripts/monitor_training.py --output logs/system_metrics.csv -- python -m teamartemisse489.train_model
```

Run monitoring inside Docker while keeping logs and models on the host:
```bash
docker run -it --rm -v ${PWD}/models:/app/models -v ${PWD}/logs:/app/logs --entrypoint python teamartemisse489:latest scripts/monitor_training.py --output logs/system_metrics.csv -- python models/train.py
```

The monitoring output is written to `logs/system_metrics.csv`. The most useful columns are `system_cpu_percent` for machine load, `system_memory_percent` and `system_memory_used_mb` for RAM pressure, `process_memory_rss_mb` for the training process memory footprint, and `elapsed_seconds` for runtime. For model health, the recommender trainer prints RMSE at the end of the run; lower RMSE indicates better recommendation accuracy on the held-out test split.

#### Debugging Practices
The baseline recommender trainer includes explicit validation checks before model training so common ML data bugs fail early with clear messages. It checks that the processed parquet file exists, the dataframe is not empty, required columns are present, required values are not null, and ratings are numeric values between 1 and 5.

Run the trainer normally:
```bash
python models/train.py
```

Use Python's built-in debugger when you need line-by-line inspection:
```bash
python -m pdb models/train.py
```

Use the project debug hook to pause after data loading and validation:
```bash
python models/train.py --debug
```

At the debugger prompt, useful checks include:
```python
p df.shape
p df.dtypes
p df[["userId", "movieId", "rating"]].head()
p df["rating"].describe()
```

Debugging scenario 1: if training fails with missing `userId`, `movieId`, or `rating` columns, inspect `df.columns` and fix the preprocessing step that writes `data/processed/ready_to_train_data.parquet`.

Debugging scenario 2: if training fails because ratings are null, nonnumeric, or outside the expected 1-5 range, inspect `df["rating"].describe()` and the invalid rows before creating the Surprise dataset.

### Phase 3: CI/CD & Deployment
- See [PHASE3.md](PHASE3.md) for detailed checklist

## Setup Instructions

### Prerequisites
- Python 3.11+ installed
- Git installed
- (Optional) Docker and Docker Compose for the Phase 2 container workflow

### Installation

**Option 1: Using uv (recommended - faster)**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Option 2: Using pip**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Running the Pipeline

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict

# See all available commands
make help
```

## Contribution Summary

TODO:
- [x] Team members have been assigned

## References

- [Project Documentation](docs/index.md)
- [Phase 1 — Project Design & Model Development](PHASE1.md)
- [Phase 2 — Containerization & Monitoring](PHASE2.md)
- [Phase 3 — CI/CD & Deployment](PHASE3.md)
- We have used a dataset from kaggle link: https://www.kaggle.com/datasets/bwandowando/rotten-tomatoes-9800-movie-critic-and-user-reviews. For this project, we are using only 1 million user reviews to recommend movies to users.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
