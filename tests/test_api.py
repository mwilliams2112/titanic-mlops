import pytest
from fastapi.testclient import TestClient

from titanic import api

VALID = {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0}


@pytest.fixture
def client(model, monkeypatch):
    # Swap the real model loader for the small fixture model,
    # so the tests need no files on disk.
    monkeypatch.setattr(api, "load_model", lambda: model)
    with TestClient(api.app) as client:  # the with-block triggers lifespan startup
        yield client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1


def test_predict_ranks_passengers_sensibly(client):
    woman = client.post("/predict", json=VALID).json()
    man = client.post(
        "/predict", json={"Pclass": 3, "Sex": "male", "SibSp": 0, "Parch": 0}
    ).json()

    assert woman["probability"] > man["probability"]


@pytest.mark.parametrize(
    "bad_field",
    [
        {"Sex": "unknown"},
        {"Pclass": 4},
        {"SibSp": -1},
        {"Parch": 11},
    ],
)
def test_predict_rejects_invalid_input(client, bad_field):
    response = client.post("/predict", json={**VALID, **bad_field})

    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID.items() if k != "Parch"}

    response = client.post("/predict", json=incomplete)

    assert response.status_code == 422