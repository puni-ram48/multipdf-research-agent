import os
import requests
from typing import List, Dict, Any, Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_V1_STR = "/api/v1"


class APIClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> bool:
        """Verifies backend service connectivity."""
        try:
            res = requests.get(f"{self.base_url}/health", timeout=5)
            return res.status_code == 200
        except Exception:
            return False

    def ingest_pdfs(self, files: List[Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Uploads PDF files to the FastAPI /ingest endpoint.
        """
        url = f"{self.base_url}{API_V1_STR}/ingest"
        multipart_files = [
            ("files", (f.name, f.getvalue(), "application/pdf"))
            for f in files
        ]
        data = {}
        if session_id:
            data["session_id"] = session_id

        response = requests.post(url, files=multipart_files, data=data, timeout=None)
        if response.status_code == 201:
            return response.json()
        else:
            raise RuntimeError(f"Ingestion failed ({response.status_code}): {response.text}")

    def send_chat_question(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        Sends a query to the FastAPI /chat endpoint.
        """
        url = f"{self.base_url}{API_V1_STR}/chat"
        payload = {"session_id": session_id, "question": question}
        response = requests.post(url, json=payload, timeout=None)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Chat request failed ({response.status_code}): {response.text}")

    def clear_memory(self, session_id: str) -> bool:
        """
        Clears conversation memory for a session.
        """
        url = f"{self.base_url}{API_V1_STR}/chat/memory/{session_id}"
        response = requests.delete(url, timeout=10)
        return response.status_code == 200


api_client = APIClient()