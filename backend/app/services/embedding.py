from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings


class EmbeddingService:
    def __init__(self):
        self._model: Optional[HuggingFaceEmbeddings] = None
        self.active_model_name: str = ""

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns cached HuggingFaceEmbeddings instance.
        Tries PRIMARY_EMBED_MODEL (bge-large-en-v1.5) first,
        falling back to FALLBACK_EMBED_MODEL (all-MiniLM-L6-v2) on error.
        """
        if self._model is not None:
            return self._model

        models_to_try = [
            settings.PRIMARY_EMBED_MODEL,
            settings.FALLBACK_EMBED_MODEL,
        ]

        for model_name in models_to_try:
            try:
                print(f"Attempting to load embedding model: {model_name}")
                self._model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.active_model_name = model_name
                print(f"Successfully loaded embedding model: {model_name}")
                return self._model
            except Exception as e:
                print(f"Failed to load {model_name}: {e}. Trying fallback...")

        raise RuntimeError("Could not initialize primary or fallback embedding models.")


# Global singleton instance
embedding_service = EmbeddingService()