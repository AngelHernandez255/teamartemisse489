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


#### Experiment Tracking

This project uses Weights & Biases for experiment tracking.

- Project name: `Team-Artemisse489-Recommender`
- Entity: `sakshigorkhaliprojects`
- Training entrypoint: `python -m teamartemisse489.train_model`
- Sweep file: `src/teamartemisse489/sweep.yaml`

The training code logs hyperparameters, evaluation metrics, and the trained SVD model artifact to W&B. For the sweep, we optimize `precision_at_10` because the recommender's main goal is to return relevant movies in the top-10 list. `rmse` is still logged as a secondary metric for reference.

To run the sweep:

```bash
cd src/teamartemisse489
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

Run names follow the hyperparameters, for example `svd_nf50_ep30_lr0.005_reg0.02`, so it is easy to compare runs in the W&B UI.

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
