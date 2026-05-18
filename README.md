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

### Phase 3: CI/CD & Deployment
- See [PHASE3.md](PHASE3.md) for detailed checklist

## Setup Instructions

### Prerequisites
- Python 3.11+ installed
- Git installed
- (Optional) Docker and Docker Compose

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
