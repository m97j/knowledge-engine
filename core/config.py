# core/config.py

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    This is a class that manages application global settings.  
    It reads values ​​from .env files or system environment variables and strictly validates types using Pydantic.
    """
    
    # 1. Project Info
    PROJECT_NAME: str = Field(default="Knowledge Engine", description="Project name")
    VERSION: str = Field(default="1.0.0", description="API version")
    ENVIRONMENT: str = Field(default="development", description="Execution environment (development, staging, production)")
    LOG_LEVEL: str = Field(default="INFO", description="Global logging level")
    DATA_DIR: str = Field(default="./data", description="Data storage directory path")
    REPO_ID: str = Field(default="m97j/ke-store", description="Hugging Face repository ID")

    # 2. Storage Settings (Vector DB & RDBMS)
    QDRANT_PATH: str = Field(default="./data/qdrant", description="Qdrant local storage path")
    QDRANT_COLLECTION: str = Field(default="knowledge_base", description="Qdrant collection name")
    SQLITE_PATH: str = Field(default="./data/corpus/corpus.sqlite", description="SQLite DB file path")

    # 3. Model Settings (Embedder & Reranker)
    EMBEDDER_NAME: str = Field(default="BAAI/bge-m3", description="FlagEmbedding model name")
    RERANKER_NAME: str = Field(default="BAAI/bge-reranker-v2-m3", description="Cross-Encoder model name")
    USE_FP16: bool = Field(default=True, description="Whether to use FP16 precision in GPU environment")

    # 4. Search Hyperparameters
    DEFAULT_TOP_K: int = Field(default=5, description="Final number of documents to return")
    QDRANT_FETCH_LIMIT: int = Field(default=50, description="Number of candidates to fetch from Vector DB before reranking")

    # Pydantic v2 settings
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True, # case-sensitive environment variables
        extra="ignore"       # ignore unexpected fields in .env or environment variables
    )

@lru_cache()
def get_settings() -> Settings:
    """
    It caches and returns the Settings object as a Singleton.  
    It offers performance advantages as it does not read or parse the file every time.
    """
    return Settings()

# Instantiate as a global variable so that it can be easily imported from other modules
settings = get_settings()