import pandas as pd
import pytest

from titanic.model import build_pipeline
from titanic.predict import predict

def test_predict_returns_valid_output(model):
    sample = pd.DataFrame([
        {"Pclass": 3, "Sex": "male", "SibSp": 0, "Parch": 0},
        {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0},
    ])

    result = predict(model, sample)

    assert len(result) == 2
    assert set(result.columns) == {"prediction", "probability"}
    assert result["probability"].between(0, 1).all()
    assert set(result["prediction"]).issubset({0, 1})