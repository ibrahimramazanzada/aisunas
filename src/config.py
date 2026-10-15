"""Configuration settings for az-legal-rag.

Provides centralized management of directory paths, model identifiers,
retrieval thresholds, and legal disclaimer constants across the pipeline.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    DATA_TRAINING_DIR: Path = BASE_DIR / "data" / "training_data"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"
    MODELS_DIR: Path = BASE_DIR / "src" / "dialogue" / "models"

    # Processed data files
    ARTICLES_JSON: Path = DATA_PROCESSED_DIR / "articles.json"
    ARTICLES_JSONL: Path = DATA_PROCESSED_DIR / "articles.jsonl"
    BM25_INDEX_PATH: Path = DATA_PROCESSED_DIR / "bm25_index.pkl"

    # Model identifiers
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    LLM_MODEL_NAME: str = "gemini-flash-lite-latest"
    GEMINI_API_KEY: Optional[str] = None

    # Retrieval and Guardrail parameters
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_MIN_SCORE: float = 0.35
    CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.60

    # Azerbaijani legal disclaimer and fallback constants
    LEGAL_DISCLAIMER_AZ: str = (
        "XƏBƏRDARLIQ: Bu sistem yalnız məlumatlandırma xarakteri daşıyır "
        "və peşəkar hüquqi məsləhət hesab edilmir."
    )

    LEGAL_DISCLAIMER_EN: str = (
        "DISCLAIMER: This system is for informational purposes only and "
        "does not constitute professional legal counsel."
    )

    NO_ARTICLE_FALLBACK_AZ: str = (
        "Bu mövzu haqqında Əmək Məcəlləsində dəqiq maddə tapılmadı. "
        "Zəhmət olmasa sualınızı və ya vəziyyətinizi daha ətraflı qeyd edin."
    )


settings = Settings()
