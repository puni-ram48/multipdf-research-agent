import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "MultiPDF Research Agent"
    API_V1_STR: str = "/api/v1"
    
    # Model Configurations
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    PRIMARY_EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    FALLBACK_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector DB / FAISS Config
    INDEX_STORAGE_PATH: str = "./faiss_indices"
    
    # Retrieval Hyperparameters
    RETRIEVAL_K: int = 7
    MAX_MEMORY_TURNS: int = 6
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()