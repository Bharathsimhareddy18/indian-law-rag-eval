from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
from pydantic import Field

class Settings(BaseSettings):
    # --- Project Metadata ---
    PROJECT_NAME: str = "Indian Law RAG Assistant"
    VERSION: str = "1.0.0"
    
    # --- Server Settings ---
    HOST: str = "0.0.0.0"
    PORT: int = 7860  # Default for Hugging Face Spaces
    
    # --- API Keys & Database ---
    OPENAI_API_KEY: str
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # --- Model Configuration ---
    # We can easily swap these later without touching the logic
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHAT_MODEL_NAME: str = "gpt-4.1-mini"
    EVAL_MODEL_NAME: str = "gpt-4.1-mini"
    DIRECT_DATABASE_URL: Optional[str] = Field(None, env="DIRECT_DATABASE_URL_supabase")
    
    # --- File Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    FAISS_INDEX_PATH: Path = BASE_DIR / "cache" / "faiss.index"
    BM25_INDEX_PATH: Path = BASE_DIR / "cache" / "bm25.pkl"
    DATA_PATH: Path = BASE_DIR / "cache" / "data_cache.pkl"

    # This tells Pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global instance to be imported elsewhere
settings = Settings()