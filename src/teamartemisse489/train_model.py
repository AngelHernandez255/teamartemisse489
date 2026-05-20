"""Model training entrypoint."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import wandb
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import (AlgoBase, CoClustering,KNNBasic,KNNWithMeans,NMF,SVD,)

from teamartemisse489.config import DEFAULT_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR
from teamartemisse489.logging_config import get_logger, setup_logging
from teamartemisse489.utils.io import save_json
from teamartemisse489.utils.seed import set_seed

logger = get_logger(__name__)


def precision_recall_at_k(
    predictions: list[Any], k: int = 10, threshold: float = 3.0
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-user Precision@K and Recall@K from Surprise predictions."""
    user_est_true: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for uid, _iid, true_r, est, _details in predictions:
        user_est_true[str(uid)].append((float(est), float(true_r)))

    precisions: dict[str, float] = {}
    recalls: dict[str, float] = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda t: t[0], reverse=True)

        n_rel = sum(true_r >= threshold for _est, true_r in user_ratings)
        n_rec_k = sum(est >= threshold for est, _true_r in user_ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold) and (est >= threshold)
            for est, true_r in user_ratings[:k]
        )

        precisions[uid] = (n_rel_and_rec_k / n_rec_k) if n_rec_k else 0.0
        recalls[uid] = (n_rel_and_rec_k / n_rel) if n_rel else 0.0

    return precisions, recalls


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_algorithm(args: argparse.Namespace) -> AlgoBase:
    """Construct a Surprise algorithm instance from CLI args."""
    if args.algorithm == "svd":
        return SVD(
            n_factors=args.n_factors,
            n_epochs=args.epochs,
            lr_all=args.learning_rate,
            reg_all=args.reg_all,
            random_state=args.seed,
        )

    if args.algorithm == "nmf":
        return NMF(
            n_factors=args.n_factors,
            n_epochs=args.epochs,
            reg_pu=args.reg_pu,
            reg_qi=args.reg_qi,
            random_state=args.seed,
        )

    if args.algorithm == "coclustering":
        return CoClustering(
            n_cltr_u=args.n_cltr_u,
            n_cltr_i=args.n_cltr_i,
            n_epochs=args.epochs,
            random_state=args.seed,
        )

    if args.algorithm == "knn_basic":
        sim_options = {
            "name": args.similarity,
            "user_based": bool(args.user_based),
            "min_support": args.min_support,
        }
        return KNNBasic(k=args.k_neighbors, sim_options=sim_options)

    if args.algorithm == "knn_means":
        sim_options = {
            "name": args.similarity,
            "user_based": bool(args.user_based),
            "min_support": args.min_support,
        }
        return KNNWithMeans(k=args.k_neighbors, sim_options=sim_options)

    raise ValueError(f"Unknown algorithm: {args.algorithm}")


def train(
    data_path: Path,
    model_dir: Path,
    args: argparse.Namespace,
    run: wandb.sdk.wandb_run.Run | None,
) -> tuple[Path, dict[str, float]]:
    """Train a Surprise recommender and persist artifacts."""
    logger.info("Loading training data from %s", data_path)
    df = pd.read_parquet(data_path)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows))
    required_cols = {"userId", "movieId", "target_rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

    ratings = df[["userId", "movieId", "target_rating"]].rename(
        columns={"target_rating": "rating"}
    )
    reader = Reader(rating_scale=(args.rating_min, args.rating_max))
    data = Dataset.load_from_df(ratings, reader)

    trainset, testset = train_test_split(
        data, test_size=args.test_size, random_state=args.seed
    )

    algo = build_algorithm(args)

    start = time.time()
    algo.fit(trainset)
    train_time_s = time.time() - start

    predictions = algo.test(testset)
    rmse = float(accuracy.rmse(predictions, verbose=False))
    mae = float(accuracy.mae(predictions, verbose=False))

    precisions, recalls = precision_recall_at_k(
        predictions, k=args.k_eval, threshold=args.threshold
    )
    avg_precision = _mean(list(precisions.values()))
    avg_recall = _mean(list(recalls.values()))

    metrics: dict[str, float] = {
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "train_time_seconds": float(train_time_s),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.algorithm}.joblib"
    metrics_path = model_dir / f"{args.algorithm}_metrics.json"

    joblib.dump(algo, model_path)
    save_json(metrics, metrics_path)

    if run is not None:
        # W&B logging (metrics)
        run.log(metrics)

        artifact = wandb.Artifact(
            name=f"{args.algorithm}-model",
            type="model",
            metadata={
                "data_path": str(data_path),
                "algorithm": args.algorithm,
                **{k: float(v) for k, v in metrics.items()},
            },
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(metrics_path))
        # W&B logging (artifacts)
        run.log_artifact(artifact)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)
    return model_path, metrics


def main() -> None:
    """CLI entrypoint for model training."""
    cfg = DEFAULT_CONFIG.training

    parser = argparse.ArgumentParser(description="Train a recommender and log to W&B")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "ready_to_train_1M.parquet",
        help="Parquet file with columns userId, movieId, target_rating",
    )
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="svd",
        choices=["svd", "knn_basic", "knn_means", "nmf", "coclustering"],
    )

    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows (fast smoke tests)",
    )

    parser.add_argument("--rating-min", type=float, default=0.5)
    parser.add_argument("--rating-max", type=float, default=5.0)
    parser.add_argument("--k-eval", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=3.0)

    parser.add_argument("--n-factors", type=int, default=100)
    parser.add_argument("--reg-all", type=float, default=0.02)
    parser.add_argument("--reg-pu", type=float, default=0.06)
    parser.add_argument("--reg-qi", type=float, default=0.06)
    parser.add_argument("--n-cltr-u", type=int, default=3)
    parser.add_argument("--n-cltr-i", type=int, default=5)

    parser.add_argument("--k-neighbors", type=int, default=40)
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "msd", "pearson", "pearson_baseline"],
    )
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--user-based", action="store_true")

    parser.add_argument("--wandb-project", type=str, default="teamartemisse489")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    run: wandb.sdk.wandb_run.Run | None = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "data_path": str(args.data_path),
                "model_dir": str(args.model_dir),
                "algorithm": args.algorithm,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "test_size": args.test_size,
                "rating_min": args.rating_min,
                "rating_max": args.rating_max,
                "k_eval": args.k_eval,
                "threshold": args.threshold,
                "n_factors": args.n_factors,
                "reg_all": args.reg_all,
                "reg_pu": args.reg_pu,
                "reg_qi": args.reg_qi,
                "n_cltr_u": args.n_cltr_u,
                "n_cltr_i": args.n_cltr_i,
                "k_neighbors": args.k_neighbors,
                "similarity": args.similarity,
                "min_support": args.min_support,
                "user_based": args.user_based,
            },
        )

    try:
        _model_path, metrics = train(args.data_path, args.model_dir, args, run)
        logger.info("Training complete; rmse=%.4f", metrics["rmse"])
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
