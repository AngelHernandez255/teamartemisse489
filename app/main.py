"""FastAPI serving app for the TeamArtemisSE489 Movie Recommender.

Endpoints:
    GET  /              root info
    GET  /health        health check
    POST /predict       top-N movie recommendations for a user

Run locally:
   # Download model and data first
    gsutil cp gs://mlops489-dvc-123456/models/svd.joblib models/svd.joblib

    gsutil cp gs://mlops489-dvc-123456/data/processed/ready_to_train_1M.parquet \
        data/processed/ready_to_train_1M.parquet

    gsutil cp gs://mlops489-dvc-123456/data/raw/movies.parquet data/raw/movies.parquet

    uvicorn app.main:app --reload --port 8080

Environment variables:
    GCS_BUCKET   GCS bucket name (default: mlops489-dvc-123456)
    PORT         port to listen on (default: 8080, set by Cloud Run)
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET", "mlops489-dvc-123456")

# For local development, fall back to local files if they exist
LOCAL_MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", "models/svd.joblib"))
LOCAL_MOVIES_PATH = Path(os.getenv("LOCAL_MOVIES_PATH", "data/raw/movies.parquet"))
LOCAL_RATINGS_PATH = Path(
    os.getenv(
        "LOCAL_RATINGS_PATH",
        "data/processed/ready_to_train_1M.parquet"
    )
)
# GCS paths
GCS_MODEL_PATH = "models/svd.joblib"
GCS_MOVIES_PATH = "data/raw/movies.parquet"
GCS_RATINGS_PATH = "data/processed/ready_to_train_1M.parquet"

# ---------------------------------------------------------------------------
# GCS loader
# ---------------------------------------------------------------------------
def load_from_gcs_or_local(local_path: Path, gcs_blob_path: str):
    """Load a file from local path if it exists, otherwise download from GCS."""
    if local_path.exists():
        print(f"Loading from local: {local_path}")
        return str(local_path)

    print(f"Downloading from GCS: gs://{GCS_BUCKET}/{gcs_blob_path}")
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_blob_path)

        # Download to a temp file
        suffix = Path(gcs_blob_path).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        blob.download_to_filename(tmp.name)
        print(f"Downloaded to {tmp.name}")
        return tmp.name
    except Exception as e:
        raise RuntimeError(
             f"Could not load {gcs_blob_path} from GCS bucket {GCS_BUCKET}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Global state loaded once at startup
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data once at startup; release on shutdown."""

    # Load model
    print("Loading SVD model...")
    model_path = load_from_gcs_or_local(LOCAL_MODEL_PATH, GCS_MODEL_PATH)
    _state["model"] = joblib.load(model_path)
    print("Model loaded ✅")

    # Load movies metadata
    print("Loading movies metadata...")
    movies_path = load_from_gcs_or_local(LOCAL_MOVIES_PATH, GCS_MOVIES_PATH)
    movies_df = pd.read_parquet(movies_path)
    keep_cols = ["movieId", "movieTitle", "movieYear", "rating",
                 "critic_score", "audience_score", "original_language", "runtime"]
    keep_cols = [c for c in keep_cols if c in movies_df.columns]
    _state["movies"] = movies_df[keep_cols].set_index("movieId")
    print(f"Movies loaded: {len(_state['movies'])} titles ✅")

    # Load ratings to build user-seen lookup
    print("Loading ratings data...")
    ratings_path = load_from_gcs_or_local(LOCAL_RATINGS_PATH, GCS_RATINGS_PATH)
    ratings_df = pd.read_parquet(ratings_path, columns=["userId", "movieId"])
    _state["user_seen"] = (
        ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
    )
    _state["all_movies"] = set(ratings_df["movieId"].unique())
    print(
        f"Ratings loaded: {len(_state['all_movies'])} movies, "
        f"{len(_state['user_seen'])} users ✅"
    )

    print("\nAPI Ready ✅\n")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Movie Recommender API",
    description=(
        "SVD-based collaborative filtering recommendations. "
        "POST a userId to /predict and receive the top-N movies "
        "ranked by predicted rating."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    user_id: int = Field(
        ...,
        description="Integer userId from the MovieLens dataset",
        example=782587125
    )
    top_n: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of recommendations to return (1-100)"
    )


class MovieRecommendation(BaseModel):
    movie_id: str
    title: str
    year: int | None
    predicted_rating: float
    critic_score: float | None
    audience_score: float | None
    mpaa_rating: str | None
    language: str | None
    runtime: str | None


class PredictResponse(BaseModel):
    user_id: int
    top_n: int
    recommendations: list[MovieRecommendation]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Root endpoint — confirms the API is live."""
    return {
        "message": "Movie Recommender API",
        "status": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    """Health check — used by Cloud Run readiness probe."""
    model_loaded = "model" in _state
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "catalogue_size": len(_state.get("all_movies", [])),
        "users_in_index": len(_state.get("user_seen", {})),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Return top-N movie recommendations for a given user.

    - Filters out movies the user has already rated.
    - Scores all unseen movies using the SVD model.
    - Returns the top-N ranked by predicted rating.
    """
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_id_str = str(request.user_id)
    seen = _state["user_seen"].get(request.user_id, set())
    unseen = _state["all_movies"] - seen

    if not unseen:
        raise HTTPException(
            status_code=404,
            detail=f"No unseen movies found for user {request.user_id}. "
                   "The user may not exist in the training set.",
        )

    # Score all unseen movies and take top N
    scored = [
        (movie_id, model.predict(uid=user_id_str, iid=movie_id).est)
        for movie_id in unseen
    ]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[: request.top_n]

    movies_meta = _state["movies"]
    recommendations = []
    for movie_id, pred_rating in top:
        if movie_id in movies_meta.index:
            row = movies_meta.loc[movie_id]
            rec = MovieRecommendation(
                movie_id=movie_id,
                title=str(row.get("movieTitle", "Unknown")),
                year=int(row["movieYear"]) if pd.notna(row.get("movieYear")) else None,
                predicted_rating=round(float(pred_rating), 4),
                critic_score=(
                    float(row["critic_score"])
                    if pd.notna(row.get("critic_score"))
                    else None
                ),
                audience_score=(
                    float(row["audience_score"])
                    if pd.notna(row.get("audience_score"))
                    else None
                ),
                mpaa_rating=(
                    str(row["rating"])
                    if pd.notna(row.get("rating"))
                    else None
                ),
                language=(
                    str(row["original_language"])
                    if pd.notna(row.get("original_language"))
                    else None
                ),
                runtime=str(row["runtime"]) if pd.notna(row.get("runtime")) else None,
            )
        else:
            rec = MovieRecommendation(
                movie_id=movie_id,
                title="Unknown",
                year=None,
                predicted_rating=round(float(pred_rating), 4),
                critic_score=None,
                audience_score=None,
                mpaa_rating=None,
                language=None,
                runtime=None,
            )
        recommendations.append(rec)

    return PredictResponse(
        user_id=request.user_id,
        top_n=request.top_n,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
