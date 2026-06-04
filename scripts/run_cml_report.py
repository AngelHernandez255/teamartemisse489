"""Generate model metrics and plots for the CML GitHub Actions report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from omegaconf import OmegaConf

from teamartemisse489.train_model import train


ROOT = Path(__file__).resolve().parents[1]
CML_DIR = ROOT / "reports" / "cml"
DATA_PATH = ROOT / "data" / "processed" / "cml_sample_ratings.parquet"
MODEL_DIR = ROOT / "models" / "cml"


def build_sample_ratings() -> pd.DataFrame:
    """Create a small deterministic ratings matrix for CI-safe training."""
    rows: list[dict[str, float | int]] = []
    for user_id in range(1, 13):
        for movie_id in range(1, 19):
            rating = ((user_id * 3 + movie_id * 2) % 5) + 1
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "target_rating": float(rating),
                }
            )
    return pd.DataFrame(rows)


def write_metrics_table(metrics: dict[str, float]) -> None:
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        *[
            f"| {name.replace('_', ' ')} | {value:.4f} |"
            for name, value in sorted(metrics.items())
        ],
    ]
    (CML_DIR / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (CML_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_metrics_plot(metrics: dict[str, float]) -> None:
    plotted_metrics = {
        name: value for name, value in metrics.items() if name != "training_time"
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [name.replace("_", "\n") for name in plotted_metrics],
        list(plotted_metrics.values()),
        color=["#2f6f6d", "#d97706", "#4f46e5", "#7c3aed"][: len(plotted_metrics)],
    )
    ax.set_title("CML model evaluation metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CML_DIR / "metrics.png", dpi=140)
    plt.close(fig)


def main() -> None:
    CML_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    build_sample_ratings().to_parquet(DATA_PATH, index=False)

    cfg = OmegaConf.create(
        {
            "data": {"test_size": 0.25},
            "training": {"seed": 42},
            "model": {
                "n_factors": 20,
                "n_epochs": 5,
                "lr_all": 0.005,
                "reg_all": 0.02,
            },
            "eval": {"k": 5, "threshold": 3.0},
        }
    )

    wandb.init(
        project="Team-Artemisse489-Recommender",
        name="cml-evaluation",
        mode="offline",
    )
    _, metrics = train(data_path=DATA_PATH, model_dir=MODEL_DIR, cfg=cfg)
    wandb.finish()

    write_metrics_table(metrics)
    write_metrics_plot(metrics)


if __name__ == "__main__":
    main()
