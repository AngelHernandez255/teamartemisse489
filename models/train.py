import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "ready_to_train_data.parquet"
MODEL_DIR = BASE_DIR / "models"
REQUIRED_COLUMNS = {"userId", "movieId", "rating"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline recommender model")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pause after data validation so dataframe state can be inspected.",
    )
    return parser.parse_args()


def validate_training_data(df: pd.DataFrame) -> None:
    """Fail fast on common recommender training data issues."""
    if df.empty:
        raise ValueError("Training data is empty")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise ValueError(
            "Training data contains nulls in required columns: "
            f"{columns_with_nulls.to_dict()}"
        )

    if not pd.api.types.is_numeric_dtype(df["rating"]):
        raise TypeError("Training data rating column must be numeric")

    invalid_ratings = ~df["rating"].between(1, 5)
    if invalid_ratings.any():
        examples = df.loc[invalid_ratings, "rating"].head(5).tolist()
        raise ValueError(
            "Training data ratings must be between 1 and 5. "
            f"Example invalid values: {examples}"
        )


def main() -> None:
    args = parse_args()
    logger.info("Loading processed data from %s", args.data_path)

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {args.data_path}. "
            "Run the data preparation step before training."
        )

    df = pd.read_parquet(args.data_path)
    validate_training_data(df)
    logger.info("Validated training data with %d rows", len(df))

    if args.debug:
        logger.info("Debug mode enabled. Inspect `df`, `df.dtypes`, and `df.head()`.")
        breakpoint()

    from surprise import SVD, Dataset, Reader
    from surprise.accuracy import rmse
    from surprise.model_selection import train_test_split

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    logger.info("Training model with Surprise SVD")

    model = SVD(random_state=42)

    model.fit(trainset)

    predictions = model.test(testset)

    score = rmse(predictions)

    args.model_dir.mkdir(exist_ok=True)

    model_path = args.model_dir / "baseline_model.pkl"

    joblib.dump(model, model_path)

    logger.info("Model saved to: %s", model_path)
    logger.info("RMSE: %.4f", score)


if __name__ == "__main__":
    main()
