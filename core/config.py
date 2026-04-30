# core/config.py

import os
from functools import lru_cache

from pydantic import Field, computed_field
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
    QDRANT_COLLECTION: str = Field(default="knowledge_base", description="Qdrant collection name")
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant server URL (if using client-server mode)")

    @computed_field
    @property
    def SQLITE_PATH(self) -> str:
        """
        Computed property to ensure that the SQLite path is always correctly resolved based on the DATA_DIR.
        This allows dynamic changes to DATA_DIR without breaking the SQLITE_PATH reference.
        """
        return os.path.join(self.DATA_DIR, "knowledge_base/corpus.sqlite")
    
    @computed_field
    @property
    def QDRANT_PATH(self) -> str:
        """
        Computed property to ensure that the Qdrant path is always correctly resolved based on the DATA_DIR.
        This allows dynamic changes to DATA_DIR without breaking the QDRANT_PATH reference.
        """
        return os.path.join(self.DATA_DIR, "vector_store/qdrant")

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