from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ask_endpoint():
    response = client.post(
        "/ask",
        json={"question": "What skills are required for this role?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
