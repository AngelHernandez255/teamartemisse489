from pathlib import Path

import joblib
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "ready_to_train_data.parquet"
MODEL_DIR = BASE_DIR / "models"


def main() -> None:
    print("Loading processed data...")

    df = pd.read_parquet(PROCESSED_DATA)

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("Training model...")

    model = SVD(random_state=42)

    model.fit(trainset)

    predictions = model.test(testset)

    score = rmse(predictions)

    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "baseline_model.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    print(f"RMSE: {score}")


if __name__ == "__main__":
    main()
