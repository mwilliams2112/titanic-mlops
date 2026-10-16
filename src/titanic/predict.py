from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from titanic.config import FEATURES, MODEL_PATH


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict(model: Pipeline, records: pd.DataFrame) -> pd.DataFrame:
    X = records[FEATURES]
    return pd.DataFrame({
        "prediction": model.predict(X),
        "probability": model.predict_proba(X)[:, 1],
    })