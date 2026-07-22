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
    line_start: int | None = None
    line_end: int | None = None


class ExtractedRequirement(BaseModel):
    """Требование, извлечённое LLM из тендерной документации."""

    text: str
    quote: str
    # К какому пункту "предмета закупки" относится требование.
    scope_item: str | None = None
    file: str
    location: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    kind: str = "other"  # product | specs | other
    priority: int = 2
    confidence: float = Field(default=0.7, ge=0, le=1)


class Status(StrEnum):
    found = "found"
    partial = "partial"
    not_found = "not_found"
    uncertain = "uncertain"


class PositionMatchStatus(StrEnum):
    matched = "matched"
    partial = "partial"
    none = "none"


STATUS_LABELS = {
    Status.found.value: "Применимо",
    Status.partial.value: "Частично применимо",
    Status.not_found.value: "Не применимо",
    Status.uncertain.value: "Недостаточно данных",
}

POSITION_STATUS_LABELS = {
    PositionMatchStatus.matched.value: "Есть вариант",
    PositionMatchStatus.partial.value: "Частично",
    PositionMatchStatus.none.value: "Нет варианта",
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
    "Для каждой позиции перечня оцени, есть ли в цитатах эталона подходящий "
    "конкретный вариант (модель/серия/обозначение), и подтверждают ли цитаты "
    "применимость к требованиям позиции по смыслу."
)


class Finding(BaseModel):
    query_text: str
    tender: Evidence
    asset_hits: list[Evidence] = Field(default_factory=list)
    status: Status = Status.uncertain
    explanation: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    kind: str = "other"  # product | specs | other


class ScopePositionMatch(BaseModel):
    """Позиция перечня: требования тендера + подобранный вариант из эталона."""

    scope_name: str
    qty: float | int | None = None
    unit: str = ""
    requirements: list[ExtractedRequirement] = Field(default_factory=list)
    status: PositionMatchStatus = PositionMatchStatus.none
    product_name: str = ""
    explanation: str = ""
    asset_hits: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class AnalysisReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tender_path: str
    assets_path: str
    embedding_model: str
    llm_model: str
    summary: str = ""
    verdict: str = ""
    findings: list[Finding] = Field(default_factory=list)
    position_matches: list[ScopePositionMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    indexed_files: list[str] = Field(default_factory=list)
    index_reused: bool = False
    elapsed_seconds: float | None = None
    query_selection: dict = Field(default_factory=dict)
    extracted_requirements: list[ExtractedRequirement] = Field(default_factory=list)
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

    # Retrieval / оценка
    top_k: int = 3
    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_tender_queries: int = 15
    max_findings: int = 12
    max_reqs_per_scope_item: int = 10

    # LangGraph: выбор файлов и extract
    max_extract_chars_per_doc: int = 120_000
    max_tender_files_initial: int = 3
    max_tender_files_total: int = 6

    user_instruction: str = DEFAULT_USER_INSTRUCTION

    ocr_enabled: bool = True
    ocr_languages: str = "rus+eng"

    cache_dir: Path = Path("data/cache")


def get_settings() -> Settings:
    return Settings()
