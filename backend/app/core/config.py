import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "MultiPDF Research Agent"
    API_V1_STR: str = "/api/v1"

    # API Keys
    GROQ_API_KEY: str
    GEMINI_API_KEY: str = "dummy_key_for_pydantic"

    # Embedding Models
    PRIMARY_EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    FALLBACK_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ingestion & Performance Settings
    INDEX_STORAGE_PATH: str = "./faiss_indices"
    NUM_CPU_THREADS: int = 8
    ENABLE_ADAPTIVE_OCR: bool = True

    # Retrieval & RAG Hyperparameters
    RETRIEVAL_K: int = 7    # Default K for standard specific questions
    SEARCH_TOP_K: int = 12  # Expanded K for high-level summary / overview questions
    QUERY_EXPANSION_ENABLED: bool = True     # Expands broad user queries into targeted sub-queries
    SECTION_BOOST_WEIGHT: float = 1.5        # Priority weight multiplier for Abstract/Intro/Conclusion chunks
    MAX_MEMORY_TURNS: int = 6         # Number of Q&A turns retained in conversation memory

    # PostgreSQL Database Configuration
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "mysecretpassword"
    POSTGRES_DB: str = "multipdf_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()