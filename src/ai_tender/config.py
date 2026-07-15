from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AI_TENDER_", extra="ignore"
    )

    model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    top_k: int = 5
    max_requirements: int = 80
    output_dir: Path = Path("data/reports")


def get_settings() -> Settings:
    return Settings()
