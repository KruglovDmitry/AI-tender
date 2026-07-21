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
    kind: str = "other"  # product | specs | other
    match_mode: str = ""  # product_first | specs_fallback | specs_only


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

    # Retrieval: запросы из тендера → поиск по эталонам
    top_k: int = 3
    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_tender_queries: int = 15
    max_findings: int = 12
    min_retrieval_score: float = 0.0
    # Стратегия сопоставления: hybrid | product | specs
    match_strategy: str = "hybrid"
    # Product-first: при найденном артикуле не разбирать все ТТХ
    product_match_confidence: float = 0.55
    max_specs_when_product_matched: int = 3
    # Извлечение требований: документ целиком → RAG по эталону → оценка
    llm_extract_requirements: bool = True
    max_extract_chars_per_doc: int = 120_000
    # Выбор тендерных файлов перед extract
    llm_select_tender_files: bool = True
    max_tender_files_initial: int = 3
    max_tender_files_total: int = 6
    extract_early_stop: bool = True
    extract_early_stop_min_specs: int = 2
    extract_early_stop_min_confidence: float = 0.55
    # Минимум файлов до остановки extract. Нужен, чтобы не пропускать "главное описание"
    # когда оно лежит в другом документе (первый слой vs детализация).
    extract_early_stop_min_files: int = 2
    # устаревшие алиасы
    max_extract_candidates: int = 40
    extract_batch_size: int = 8
    llm_select_queries: bool = True
    max_classify_candidates: int = 40
    classify_batch_size: int = 12

    # LLM prompt prefix (задача оценки)
    user_instruction: str = DEFAULT_USER_INSTRUCTION

    # OCR для PDF-сканов
    ocr_enabled: bool = True
    ocr_languages: str = "rus+eng"

    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/reports")


def get_settings() -> Settings:
    return Settings()
