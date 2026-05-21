"""Hydra configuration schema validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42


@dataclass
class DataConfig:
    data_path: str = "data/processed"


@dataclass
class PathsConfig:
    model_dir: str = "models"


@dataclass
class ModelConfig:
    name: str = "svd"
    type: str = "collaborative_filtering"
    artifact_name: str = "model.joblib"


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["rmse", "mae"])
    evaluation_output_dir: str = "reports/evaluation"
    save_results: bool = True


@dataclass
class InferenceConfig:
    batch_size: int = 32
    top_k: int = 10
    prediction_output_dir: str = "reports/predictions"


@dataclass
class ProjectConfig:
    name: str = "teamartemisse489"
    phase: str = "phase_2"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)