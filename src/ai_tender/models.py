from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Evidence(BaseModel):
    file: str
    location: str
    quote: str
    score: float | None = None
    page: int | None = None


class Status(StrEnum):
    found = "found"
    partial = "partial"
    not_found = "not_found"
    uncertain = "uncertain"


STATUS_LABELS = {
    Status.found.value: "Применимо",
    Status.partial.value: "Частично применимо",
    Status.not_found.value: "Не применимо",
    Status.uncertain.value: "Недостаточно данных",
}

STATUS_PRIORITY = {
    Status.found: 0,
    Status.partial: 1,
    Status.uncertain: 2,
    Status.not_found: 3,
}

DEFAULT_USER_INSTRUCTION = (
    "Эталон — подробное техническое описание изготавливаемой продукции. "
    "Тендер задаёт обобщённые требования к закупке (без привязки к поставщику). "
    "Для каждого требования из тендера оцени, подтверждают ли цитаты эталона "
    "применимость продукции к этому требованию по смыслу, а не только дословно. "
    "Указывай конкретные ТС/модели из эталона, если они явно названы в цитатах."
)


class Finding(BaseModel):
    query_text: str
    tender: Evidence
    asset_hits: list[Evidence] = Field(default_factory=list)
    status: Status = Status.uncertain
    explanation: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)


class AnalysisReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tender_path: str
    assets_path: str
    embedding_model: str
    llm_model: str
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    indexed_files: list[str] = Field(default_factory=list)
    index_reused: bool = False
    elapsed_seconds: float | None = None
    report_dir: Path | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AI_TENDER_", extra="ignore"
    )

    # Embeddings (локально) + LLM (API)
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    llm_provider: str = "deepseek"  # deepseek | openai
    llm_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_base_url: str = "https://api.openai.com/v1"

    # Retrieval: запросы из тендера → поиск по эталонам
    top_k: int = 3
    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_tender_queries: int = 15
    max_findings: int = 12
    min_retrieval_score: float = 0.0

    # LLM prompt prefix (задача оценки)
    user_instruction: str = DEFAULT_USER_INSTRUCTION

    # OCR для PDF-сканов
    ocr_enabled: bool = True
    ocr_languages: str = "rus+eng"

    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/reports")


def get_settings() -> Settings:
    return Settings()
