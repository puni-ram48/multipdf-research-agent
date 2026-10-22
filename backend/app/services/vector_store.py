import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.services.embedding import embedding_service


class VectorStoreService:
    def __init__(self, base_storage_path: str = settings.INDEX_STORAGE_PATH):
        self.base_storage_path = base_storage_path
        os.makedirs(self.base_storage_path, exist_ok=True)

    def _get_session_index_path(self, session_id: str) -> str:
        """Returns the local storage directory for a given session FAISS index."""
        return os.path.join(self.base_storage_path, f"faiss_{session_id}")

    def build_index(self, session_id: str, chunks: List[Document]) -> FAISS:
        """
        Builds a new FAISS vector store index from document chunks and saves it locally.
        Clears any existing index for the given session ID.
        """
        index_path = self._get_session_index_path(session_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        embeddings = embedding_service.get_embeddings()
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        vector_store.save_local(index_path)
        return vector_store

    def load_index(self, session_id: str) -> Optional[FAISS]:
        """Loads an existing session FAISS vector store index if available."""
        index_path = self._get_session_index_path(session_id)
        if not os.path.exists(os.path.join(index_path, "index.faiss")):
            return None

        embeddings = embedding_service.get_embeddings()
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def delete_index(self, session_id: str) -> bool:
        """Deletes a session FAISS index folder."""
        index_path = self._get_session_index_path(session_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            return True
        return False


# Global singleton instance
vector_store_service = VectorStoreService()