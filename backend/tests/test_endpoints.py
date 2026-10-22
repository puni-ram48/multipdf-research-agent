from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Ensure health endpoint returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": "MultiPDF Research Agent"}

def test_chat_unindexed_session_endpoint():
    """Ensure chat endpoint handles missing index gracefully."""
    payload = {
        "session_id": "test_unindexed_session_999",
        "question": "What is the primary contribution?"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    assert "No document index found" in response.json()["answer"]