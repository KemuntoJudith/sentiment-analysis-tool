import pytest
from api_server import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_api_prediction(client):
    response = client.post("/predict-sentiment", json={
        "text": "Great banking service"
    })

    assert response.status_code == 200

    