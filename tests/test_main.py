import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["models_loaded"] is True

def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()

def test_predict(client):
    response = client.post(
        "/predict",
        json={"text": "I was charged twice"},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["category"] in [
        "Billing", "Account", "Delivery", "Technical Support"
    ]
    assert result["priority"] in ["Low", "Medium", "High"]
    assert 0 <= result["confidence"] <= 1

def test_empty_text(client):
    response = client.post(
        "/predict",
        json={"text": ""},
    )

    assert response.status_code == 422

def test_missing_text(client):
    response = client.post("/predict", json={})

    assert response.status_code == 422

def test_predict_batch(client):
    response = client.post(
        "/predict-batch",
        json={
            "tickets": [
                {"text": "I was charged twice"},
                {"text": "I cannot log in to my account"},
            ]
        },
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
