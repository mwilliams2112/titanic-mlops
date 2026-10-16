import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from titanic.evaluate import evaluate

EXPECTED_KEYS = {"accuracy", "roc_auc", "recall_class_0", "recall_class_1"}


class PerfectModel:
    """Stub that always predicts the true labels, so the expected metrics are known."""

    def __init__(self, y):
        self.y = np.asarray(y)

    def predict(self, X):
        return self.y

    def predict_proba(self, X):
        return np.column_stack([1 - self.y, self.y])


def test_evaluate_returns_expected_keys_and_valid_range(model, training_data):
    X, y = training_data

    results = evaluate(model, X, y)

    assert set(results) == EXPECTED_KEYS
    assert all(0 <= value <= 1 for value in results.values())


def test_evaluate_returns_plain_floats(model, training_data):
    X, y = training_data

    results = evaluate(model, X, y)

    assert all(type(value) is float for value in results.values())


def test_evaluate_perfect_model(training_data):
    X, y = training_data

    results = evaluate(PerfectModel(y), X, y)

    assert results["accuracy"] == 1.0
    assert results["roc_auc"] == 1.0
    assert results["recall_class_0"] == 1.0
    assert results["recall_class_1"] == 1.0


def test_evaluate_baseline_matches_hand_calculation(training_data):
    X, y = training_data
    baseline = DummyClassifier(strategy="most_frequent").fit(X, y)

    results = evaluate(baseline, X, y)

    # 12 of the 20 labels are 0, so always predicting 0 is right 60% of the time
    assert results["accuracy"] == pytest.approx(0.6)
    assert results["roc_auc"] == pytest.approx(0.5)
    assert results["recall_class_0"] == 1.0
    assert results["recall_class_1"] == 0.0