"""Model training entrypoint with Hydra configuration."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from teamartemisse489.logging_config import get_logger, setup_logging
from teamartemisse489.utils.seed import set_seed

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

def train(
    data_path: Path, model_dir: Path, epochs: int, batch_size: int, lr: float
) -> None:
    """Train the model and persist the fitted artifact to ``model_dir``."""
    logger.info(
        "Training with data=%s epochs=%d bs=%d lr=%g",
        data_path,
        epochs,
        batch_size,
        lr,
    )
    model_dir.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for model training."""
    setup_logging()
    validate_config(cfg)
    set_seed(cfg.training.seed)

    logger.info("Loaded Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    train(
        data_path=Path(cfg.data.data_path),
        model_dir=Path(cfg.paths.model_dir),
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.learning_rate,
    )

    logger.info("Training complete")


if __name__ == "__main__":
    main()