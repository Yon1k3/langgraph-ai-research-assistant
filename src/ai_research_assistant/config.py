from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    ollama_model: str = Field(default="llama3.1:8b", min_length=1)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        pattern=r"^https?://",
    )
    web_search_api_key: SecretStr | None = None
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite3")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
