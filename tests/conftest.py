import pandas as pd
import pytest

from titanic.model import build_pipeline


@pytest.fixture
def training_data():
    X = pd.DataFrame({
        "Pclass": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3] * 2,
        "Sex":    ["female", "female", "male", "female", "male",
                   "male", "female", "male", "male", "male"] * 2,
        "SibSp":  [0, 1, 0, 0, 1, 0, 2, 0, 0, 1] * 2,
        "Parch":  [0, 0, 1, 0, 0, 1, 1, 0, 0, 0] * 2,
    })
    y = pd.Series([1, 1, 0, 1, 0, 0, 1, 0, 0, 0] * 2)
    return X, y


@pytest.fixture
def model(training_data):
    X, y = training_data
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline