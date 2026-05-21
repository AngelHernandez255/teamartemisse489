"""Tests for training-time validation and debugging safeguards."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "models" / "train.py"
SPEC = importlib.util.spec_from_file_location("baseline_train", TRAIN_SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
baseline_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline_train)


def test_validate_training_data_accepts_expected_schema() -> None:
    df = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
            "rating": [4.0, 5.0],
        }
    )

    baseline_train.validate_training_data(df)


def test_validate_training_data_rejects_missing_columns() -> None:
    df = pd.DataFrame({"userId": [1], "rating": [4.0]})

    with pytest.raises(ValueError, match="missing columns"):
        baseline_train.validate_training_data(df)


def test_validate_training_data_rejects_invalid_rating_range() -> None:
    df = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
            "rating": [4.0, 6.0],
        }
    )

    with pytest.raises(ValueError, match="between 1 and 5"):
        baseline_train.validate_training_data(df)
