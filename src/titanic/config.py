import os
from pathlib import Path

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

CATEGORICAL_FEATURES = ["Sex"]
NUMERIC_FEATURES = ["Pclass", "SibSp", "Parch"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
}