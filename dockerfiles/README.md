# Dockerfiles Directory

Store Docker configurations and container setup files here.

## Contents

- **`Dockerfile`** - Alternate path for the main project image definition
- **`train.dockerfile`** - Week 4 training exercise reference used as the pattern for the project Dockerfile
- **`predict.dockerfile`** - Week 4 prediction exercise reference used as the pattern for project runtime conventions
- **`docker-compose.yaml`** - Multi-container orchestration in the project root

## Usage

```bash
# Build image
docker build -t teamartemisse489:latest .

# Run training with persistent model artifacts
docker run -it --rm -v ${PWD}/models:/app/models teamartemisse489:latest

# Optional Compose workflow
docker compose up --build
```

## Phase

Phase 2 deliverable - Docker infrastructure setup.
