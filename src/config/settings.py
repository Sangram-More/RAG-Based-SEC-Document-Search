from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Configure to auto-load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra env vars not defined here
    )
    
    # API Keys
    gemini_api_key: str
    langsmith_api_key: str | None = None
    
    # Data Paths
    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    chromadb_dir: Path = Path("./data/chroma_db")
    
    # Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemini-1.5-flash"
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

# Cache the settings instance
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
