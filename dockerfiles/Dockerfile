# -----------------------------------------------------------------------------
# TeamArtemisSE489 training image.
#
# Adapted from the Week 4 Docker exercise, updated for this repository's
# package layout.
#
# Build: docker build -t teamartemisse489:latest .
# Run:   docker run -it --rm -v ${PWD}/models:/app/models teamartemisse489:latest
# Run:   docker run -it --rm -v %cd%/models:/app/models teamartemisse489:latest # (Windows)
# -----------------------------------------------------------------------------

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

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

RUN pip install --no-cache-dir . --no-deps

ENTRYPOINT ["python", "-u", "-m", "teamartemisse489.train_model"]
