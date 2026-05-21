"""Model training entrypoint with Hydra configuration."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import joblib
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SVD

from teamartemisse489.logging_config import get_logger, setup_logging
from teamartemisse489.utils.seed import set_seed

_WANDB_RUNS = Path.home() / ".wandb-runs"
_WANDB_RUNS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(_WANDB_RUNS))


logger = get_logger(__name__)


def validate_config(cfg) -> None:
    """Validate important Hydra config values."""
    if cfg.training.epochs <= 0:
        raise ValueError("training.epochs must be greater than 0")

    if cfg.training.batch_size <= 0:
        raise ValueError("training.batch_size must be greater than 0")

    if cfg.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be greater than 0")

    if cfg.inference.top_k <= 0:
        raise ValueError("inference.top_k must be greater than 0")

    """Train the model and persist the fitted artifact to ``model_dir``."""


def eval_topk(predictions: list[Any], k: int = 10, threshold: float = 3.0):
    user_data = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_data[str(uid)].append((est, true_r))
    tp = fp = fn = 0
    precision_list, recall_list = [], []

    for _, items in user_data.items():
        items.sort(
            key=lambda x: x[0], reverse=True
        )  # Sort by estimated rating (highest first)
        topk = items[:k]
        n_rel = sum(1 for _, r in items if r >= threshold)  # Number of relevant items
        n_tp = sum(
            1 for e, r in topk if r >= threshold
        )  # Number of recommended items in top K
        tp += n_tp
        fp += max(0, len(topk) - n_tp)
        fn += max(0, n_rel - n_tp)
        precision_list.append(n_tp / k if k else 0.0)
        recall_list.append(n_tp / n_rel if n_rel else 0.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_list": precision_list,
        "recall_list": recall_list,
        "precision": sum(precision_list) / len(precision_list)
        if precision_list
        else 0.0,
        "recall": sum(recall_list) / len(recall_list) if recall_list else 0.0,
    }


# Train
def train(data_path: Path, model_dir: Path, cfg) -> tuple[Path, dict[str, float]]:

    logger.info("Loading training data from %s", data_path)

    df = pd.read_parquet(data_path)
    rating_col = "target_rating" if "target_rating" in df.columns else "rating"
    num_users = int(df["userId"].nunique())
    num_movies = int(df["movieId"].nunique())
    num_ratings = int(len(df))

    wandb.log(
        {
            "num_users": num_users,
            "num_movies": num_movies,
            "num_ratings": num_ratings,
        }
    )

    # Surprise dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["userId", "movieId", rating_col]].rename(columns={rating_col: "rating"}),
        reader,
    )

    trainset, testset = train_test_split(
        data,
        test_size=cfg.data.test_size,
        random_state=cfg.training.seed,
    )
    # Model
    model = SVD(
        n_factors=cfg.model.n_factors,
        n_epochs=cfg.model.n_epochs,
        lr_all=cfg.model.lr_all,
        reg_all=cfg.model.reg_all,
        random_state=cfg.training.seed,
    )

    # Train
    start = time.time()
    model.fit(trainset)
    training_time = time.time() - start

    # Predict
    predictions = model.test(testset)

    # metrics
    rmse_val = accuracy.rmse(predictions, verbose=False)
    mae_val = accuracy.mae(predictions, verbose=False)
    eval_res = eval_topk(predictions, k=cfg.eval.k, threshold=cfg.eval.threshold)
    metrics = {
        "rmse": rmse_val,
        "mae": mae_val,
        "precision_at_10": eval_res["precision"],
        "recall_at_10": eval_res["recall"],
        "training_time": training_time,
    }

    wandb.log(metrics)

    wandb.run.summary.update(
        {
            "best_rmse": float(rmse_val),
            "best_mae": float(mae_val),
            "best_precision_at_10": float(eval_res["precision"]),
            "best_recall_at_10": float(eval_res["recall"]),
            "num_users": num_users,
            "num_movies": num_movies,
            "num_ratings": num_ratings,
            "rating_column": rating_col,
            "rating_scale_min": 1,
            "rating_scale_max": 5,
            "test_size": float(cfg.data.test_size),
            "random_state": int(cfg.training.seed),
            "preprocessing_target_rating_used": bool("target_rating" in df.columns),
        }
    )

    wandb.log(
        {
            "precision_distribution": wandb.Histogram(eval_res["precision_list"]),
            "recall_distribution": wandb.Histogram(eval_res["recall_list"]),
            "tp": eval_res["tp"],
            "fp": eval_res["fp"],
            "fn": eval_res["fn"],
        }
    )

    # save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "svd.joblib"

    joblib.dump(model, model_path)

    artifact = wandb.Artifact("svd-model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    logger.info("Saved model to %s", model_path)
    return model_path, metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    validate_config(cfg)
    set_seed(cfg.training.seed)

    logger.info("Loaded Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logging.wandb_mode,
    )

    train(
        data_path=Path(cfg.data.data_path),
        model_dir=Path(cfg.paths.model_dir),
        cfg=cfg,
    )

    wandb.finish()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
