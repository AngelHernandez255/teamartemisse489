"""Model training entrypoint with Hydra configuration."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import os 

_WANDB_RUNS = Path.home() / ".wandb-runs"
_WANDB_RUNS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(_WANDB_RUNS))

import wandb
from surprise.accuracy import rmse
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SVD

import hydra
from omegaconf import DictConfig, OmegaConf

from teamartemisse489.logging_config import get_logger, setup_logging
from teamartemisse489.utils.io import save_json
from teamartemisse489.utils.seed import set_seed

logger = get_logger(__name__)


def eval_topk(predictions: list[Any], k: int = 10, threshold: float = 3.0):
    user_data = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_data[str(uid)].append((est, true_r))
    tp = fp = fn = 0
    precision_list, recall_list = [], []

    for _, items in user_data.items():
        items.sort(key=lambda x: x[0], reverse=True) # Sort by estimated rating (highest first)
        topk = items[:k]
        n_rel = sum(1 for _, r in items if r >= threshold) # Number of relevant items
        n_tp = sum(1 for e, r in topk if r >= threshold) # Number of recommended items in top K
        tp += n_tp
        fp += max(0, len(topk) - n_tp)
        fn += max(0, n_rel - n_tp)
        precision_list.append(n_tp / k if k else 0.0)
        recall_list.append(n_tp / n_rel if n_rel else 0.0)

    return {"tp": tp,"fp": fp, "fn": fn, "precision_list": precision_list, "recall_list": recall_list, "precision": sum(precision_list) / len(precision_list) if precision_list else 0.0,"recall": sum(recall_list) / len(recall_list) if recall_list else 0.0,}

# Train
def train(data_path: Path, model_dir: Path, cfg) -> tuple[Path, dict[str, float]]:

    logger.info("Loading training data from %s", data_path)

    df = pd.read_parquet(data_path)
    rating_col = "target_rating" if "target_rating" in df.columns else "rating"
    num_users = int(df["userId"].nunique())
    num_movies = int(df["movieId"].nunique())
    num_ratings = int(len(df))

    wandb.log({
        "num_users": num_users,
        "num_movies": num_movies,
        "num_ratings": num_ratings,
    })

    # Surprise dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[["userId", "movieId", rating_col]].rename(columns={rating_col: "rating"}),
        reader)

    trainset, testset = train_test_split(data, test_size=cfg.test_size,random_state=cfg.random_state)

    # Model
    model = SVD(
        n_factors=cfg.n_factors,
        n_epochs=cfg.n_epochs,
        lr_all=cfg.lr_all,
        reg_all=cfg.reg_all,
        random_state=cfg.random_state
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
    eval_res = eval_topk(predictions, k=cfg.k, threshold=cfg.threshold)
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
            "test_size": float(cfg.test_size),
            "random_state": int(cfg.random_state),
            "preprocessing_target_rating_used": bool("target_rating" in df.columns),
        }
    )

    wandb.log({
        "precision_distribution": wandb.Histogram(eval_res["precision_list"]),
        "recall_distribution": wandb.Histogram(eval_res["recall_list"]),
        "tp": eval_res["tp"],
        "fp": eval_res["fp"],
        "fn": eval_res["fn"],
    })

    # save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "svd.joblib"

    joblib.dump(model, model_path)

    artifact = wandb.Artifact("svd-model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    logger.info("Saved model to %s", model_path)
    return model_path, metrics


def main():

    setup_logging()
    wandb.init(
        project="Team-Artemisse489-Recommender",
        entity="sakshigorkhaliprojects",
        name="svd-training",
        config={
            "n_factors": 50,
            "n_epochs": 30,
            "lr_all": 0.005,
            "reg_all": 0.02,
            "random_state": 42,
            "test_size": 0.2,
            "k": 10,
            "threshold": 3.0,
        },
    )

    cfg = wandb.config
    wandb.run.name = (
        f"svd_nf{cfg.n_factors}_ep{cfg.n_epochs}_lr{cfg.lr_all}_reg{cfg.reg_all}"
    )

    train(
        data_path=PROCESSED_DATA_DIR / "ready_to_train_1M.parquet",
        model_dir=MODELS_DIR,
        cfg=cfg,
    )

    wandb.finish()
    # parser = argparse.ArgumentParser(description="Train the model") 
    # parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR) 
    # parser.add_argument("--model-dir", type=Path, default=MODELS_DIR) 
    # parser.add_argument("--epochs", type=int, default=cfg.epochs)
    # parser.add_argument("--batch-size", type=int, default=cfg.batch_size) 
    # parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate) 
    # parser.add_argument("--seed", type=int, default=cfg.seed) # args = parser.parse_args() # setup_logging() # set_seed(args.seed) # train(args.data_path, args.model_dir, args.epochs, args.batch_size, args.learning_rate) # logger.info("Training complete")

if __name__ == "__main__":
    main()