import os
import shutil
from typing import List, Optional, Tuple
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
        Builds a new FAISS vector store or appends new document chunks 
        to an existing session index.
        """
        index_path = self._get_session_index_path(session_id)
        embeddings = embedding_service.get_embeddings()

        # If an index already exists for this session, append new chunks
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            existing_vector_store = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            existing_vector_store.add_documents(chunks)
            existing_vector_store.save_local(index_path)
            return existing_vector_store
        else:
            # Create a brand new index if none exists
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

    def similarity_search_with_boost(
        self, 
        session_id: str, 
        query: str, 
        k: int = settings.SEARCH_TOP_K
    ) -> List[Document]:
        """
        Performs similarity search with section priority boosting:
        Reranks retrieved chunks so Abstract, Introduction, and Conclusion sections
        are prioritized when answering broad overview or summary questions.
        """
        db = self.load_index(session_id)
        if db is None:
            return []

        # 1. Perform L2 similarity search with distance scores
        results_with_scores: List[Tuple[Document, float]] = db.similarity_search_with_score(
            query, k=k
        )

        boosted_docs: List[Tuple[Document, float]] = []

        # 2. Apply section priority multipliers to distance scores (lower distance = higher similarity)
        priority_sections = {"abstract", "introduction", "conclusion", "methodology", "summary"}
        boost_weight = getattr(settings, "SECTION_BOOST_WEIGHT", 1.5)

        for doc, score in results_with_scores:
            section = str(doc.metadata.get("section", "")).lower()
            doc_type = str(doc.metadata.get("type", "")).lower()

            adjusted_score = score
            if any(p in section for p in priority_sections) or doc_type == "abstract":
                adjusted_score = score / boost_weight  # Lower distance score increases rank

            boosted_docs.append((doc, adjusted_score))

        # 3. Sort by adjusted distance score ascending
        boosted_docs.sort(key=lambda x: x[1])

        return [doc for doc, _ in boosted_docs]

    def delete_index(self, session_id: str) -> bool:
        """Deletes a session FAISS index folder."""
        index_path = self._get_session_index_path(session_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            return True
        return False


# Global singleton instance
vector_store_service = VectorStoreService()