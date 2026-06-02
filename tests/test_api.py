import pytest
from fastapi.testclient import TestClient

import bootstrap
from backend.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health(client):
    response = client.get("/api/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_analytics_summary(client):
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200


def test_training_algorithms(client):
    response = client.get("/api/training/algorithms")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_system_info(client):
    response = client.get("/api/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "algorithms" in data


def test_models_endpoint(client):
    response = client.get("/api/models/")
    assert response.status_code == 200
    assert "model_dir" in response.json()


def test_training_status(client):
    response = client.get("/api/training/status")
    assert response.status_code == 200
    assert "status" in response.json()
