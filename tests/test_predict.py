import joblib
import pandas as pd
import pytest

from titanic.model import build_pipeline
from titanic.predict import load_model, predict


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


def test_predict_handles_unseen_category(model):
    sample = pd.DataFrame([
        {"Pclass": 1, "Sex": "unknown", "SibSp": 0, "Parch": 0},
    ])

    result = predict(model, sample)

    assert len(result) == 1
    assert 0 <= result["probability"].iloc[0] <= 1


def test_predict_raises_on_missing_column(model):
    sample = pd.DataFrame([
        {"Pclass": 1, "Sex": "female", "SibSp": 0},  # Parch missing
    ])

    with pytest.raises(KeyError):
        predict(model, sample)


def test_save_load_round_trip(model, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = load_model(path)

    sample = pd.DataFrame([
        {"Pclass": 3, "Sex": "male", "SibSp": 0, "Parch": 0},
        {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0},
    ])

    pd.testing.assert_frame_equal(predict(model, sample), predict(loaded, sample))