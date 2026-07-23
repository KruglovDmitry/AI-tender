from enum import StrEnum
from pathlib import Path
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from llama_index.core import Document
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


class PositionMatchStatus(StrEnum):
    matched = "matched"
    partial = "partial"
    none = "none"


POSITION_STATUS_LABELS = {
    PositionMatchStatus.matched.value: "Есть вариант",
    PositionMatchStatus.partial.value: "Частично",
    PositionMatchStatus.none.value: "Нет варианта",
}

DEFAULT_USER_INSTRUCTION = (
    "Эталон — подробное техническое описание изготавливаемой продукции. "
    "Тендер задаёт обобщённые требования к закупке (без привязки к поставщику). "
    "Для каждой позиции перечня оцени, есть ли в цитатах эталона подходящий "
    "конкретный вариант (модель/серия/обозначение) основного изделия. "
    "Если позиция — комплект, а в эталоне подтверждено только основное изделие "
    "без части комплектующих — это частичное покрытие, не отказ."
)


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
    position_matches: list[ScopePositionMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    indexed_files: list[str] = Field(default_factory=list)
    index_reused: bool = False
    elapsed_seconds: float | None = None
    query_selection: dict = Field(default_factory=dict)


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
    max_reqs_per_scope_item: int = 10

    # LangGraph: выбор файлов и extract
    max_extract_chars_per_doc: int = 120_000
    max_tender_files_initial: int = 3
    max_tender_files_total: int = 6
    # Сколько файлов-кандидатов пробовать на позицию (после дедупа docx/pdf).
    max_requirement_files: int = 3
    # Параллельные LLM-запросы требований по позициям внутри одного файла.
    requirements_parallelism: int = 6
    # Параллельный подбор эталона по позициям (retrieval + match LLM).
    match_parallelism: int = 4

    # Трассировка LLM/retrieval в data/llm_traces/run_*
    llm_trace_enabled: bool = True
    llm_trace_dir: Path = Path("data/llm_traces")

    user_instruction: str = DEFAULT_USER_INSTRUCTION

    ocr_enabled: bool = True
    ocr_languages: str = "rus+eng"

    cache_dir: Path = Path("data/cache")


def get_settings() -> Settings:
    return Settings()


ProgressCallback = Callable[[str, float], None]


class PipelineState(TypedDict, total=False):
    """Состояние LangGraph-пайплайна."""

    tender_path: str
    assets_path: str
    llm: Any
    settings: Any
    progress: Any
    cleanup_box: Any

    inventory: Any
    catalog_entries: list[Any]
    ranked_paths: list[str]
    doc_selection: dict[str, Any]

    loaded_labels: list[str]
    documents: Annotated[list[Document], operator.add]
    scope_queue: list[str]
    scope_files_used: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]

    scope_items: list[dict[str, Any]]
    scope_meta: dict[str, Any]

    requirements_by_item: list[list[ExtractedRequirement]]
    requirements_stats: dict[str, Any]
    requirement_queue: list[str]
    requirement_files_tried: list[str]
    current_requirement_file: str

    assets_index: Any
    indexed_files: list[str]
    index_reused: bool

    position_matches: list[ScopePositionMatch]
    verdict: str
    query_selection: dict[str, Any]
