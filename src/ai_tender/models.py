from enum import StrEnum
from pathlib import Path
import operator
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from llama_index.core import Document
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _openai_compatible_base_url(url: str) -> str:
    """Нормализует base URL для OpenAI SDK (…/v1)."""
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


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


class DocumentKind(StrEnum):
    """Тип эталонного документа."""

    catalog = "catalog"
    product = "product"
    other = "other"
    asset = "asset"  # постраничная VL-индексация без классификации


DOCUMENT_KIND_LABELS: dict[DocumentKind, str] = {
    DocumentKind.catalog: "каталог",
    DocumentKind.product: "описание/паспорт продукта",
    DocumentKind.other: "прочее",
    DocumentKind.asset: "эталон",
}


class IndexingStatus(StrEnum):
    """Статус обработки файла индексатором."""

    skipped = "skipped"  # не индексируем (прочее)
    pending = "pending"  # тип определён, спец. логика ещё не реализована
    indexed = "indexed"  # успешно проиндексирован своим индексатором
    failed = "failed"


@dataclass
class IndexingResult:
    """Результат индексации одного файла (для UI / оркестратора)."""

    relative_path: str
    doc_kind: DocumentKind
    status: IndexingStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingContext:
    """Контекст для индексатора (VL, embeddings и т.п.)."""

    assets_path: Path
    cache_dir: Path | None = None
    llm: Any = None
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    ocr_enabled: bool = True
    ocr_languages: str = "rus+eng"
    vl_base_url: str = "http://127.0.0.1:8000/v1"
    vl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vl_api_key: str = "EMPTY"
    vl_max_pages: int = 40
    vl_image_scale: float = 1.5
    vl_timeout_sec: float = 120.0
    vl_max_output_tokens: int = 2000
    extra: dict[str, Any] = field(default_factory=dict)


class ProductSource(BaseModel):
    catalog_id: str = ""
    version: str = ""
    page: int | None = None
    bbox: list[float] | None = None


class Product(BaseModel):
    """Продукт, извлечённый из каталога или паспорта."""

    id: str = ""
    model: str = ""
    manufacturer: str = ""
    category: str = ""
    canonical_desc: str = ""
    raw_chunk: str = ""
    source: ProductSource = Field(default_factory=ProductSource)
    characteristics: list[str] = Field(default_factory=list)
    standards: list[str] = Field(default_factory=list)


class ProductDocumentIndex(BaseModel):
    """JSON-индекс одного исходного файла (каталог или паспорт)."""

    source_file: str
    doc_kind: DocumentKind
    catalog_name: str = ""
    products: list[Product] = Field(default_factory=list)
    # Страницы, на которых VL нашёл описание продукции (после скана).
    product_pages: list[int] = Field(default_factory=list)
    embedding_model: str = ""
    warnings: list[str] = Field(default_factory=list)


DEFAULT_USER_INSTRUCTION = (
    "Эталон — подробное техническое описание изготавливаемой нами продукции. "
    "Тендер задаёт требования к закупке (часто чужой тип «или аналог»). "
    "Для каждой позиции подбери конкретную модель/артикул из цитат эталона "
    "(строка таблицы каталога, не только имя серии), в том числе как аналог, "
    "если характеристики подходят. Не отказывай только потому, что артикул "
    "заказчика не встречается в эталоне. "
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
    # Конкретное обозначение из тендера (позиция/требования), если заказчик его задал.
    required_product: str = ""
    # Подобранный вариант из цитат эталона (assets).
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

    # Vision-LLM для постраничной индексации эталонов (OpenAI-compatible, напр. vLLM).
    # Можно задать AI_TENDER_VL_* или LOCAL_LLM_* (см. get_settings).
    vl_base_url: str = "http://127.0.0.1:8000/v1"
    vl_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vl_api_key: str = "EMPTY"
    vl_max_pages: int = 40
    vl_image_scale: float = 1.5
    vl_timeout_sec: float = 120.0
    vl_max_output_tokens: int = 2000

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
    """Загружает Settings; LOCAL_LLM_* подставляются в VL, если AI_TENDER_VL_* не заданы."""
    settings = Settings()
    updates: dict[str, Any] = {}

    if "AI_TENDER_VL_BASE_URL" not in os.environ:
        local_url = os.getenv("LOCAL_LLM_BASE_URL")
        if local_url:
            updates["vl_base_url"] = _openai_compatible_base_url(local_url)
    else:
        updates["vl_base_url"] = _openai_compatible_base_url(settings.vl_base_url)

    if "AI_TENDER_VL_MODEL" not in os.environ:
        local_model = os.getenv("LOCAL_LLM_DEFAULT_MODEL")
        if local_model:
            updates["vl_model"] = local_model.strip()

    if "AI_TENDER_VL_TIMEOUT_SEC" not in os.environ:
        local_timeout = os.getenv("LOCAL_LLM_TIMEOUT_SEC")
        if local_timeout:
            try:
                updates["vl_timeout_sec"] = float(local_timeout)
            except ValueError:
                pass

    if "AI_TENDER_VL_MAX_OUTPUT_TOKENS" not in os.environ:
        local_max = os.getenv("LOCAL_LLM_MAX_OUTPUT_TOKENS")
        if local_max:
            try:
                updates["vl_max_output_tokens"] = int(local_max)
            except ValueError:
                pass

    return settings.model_copy(update=updates) if updates else settings


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
