from enum import StrEnum
from pathlib import Path
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
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


class DocumentKind(StrEnum):
    """Тип эталонного документа."""

    catalog = "catalog"
    product = "product"
    other = "other"
    asset = "asset"  # Qwen whole-file индексация эталона


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
    """Контекст индексации эталона (Qwen extract + embeddings)."""

    assets_path: Path
    cache_dir: Path | None = None
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
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
    product_pages: list[int] = Field(default_factory=list)
    embedding_model: str = ""
    warnings: list[str] = Field(default_factory=list)


DEFAULT_USER_INSTRUCTION = (
    "Эталон — каталог продукции, извлечённый Qwen whole-file (модель, характеристики). "
    "Тендер задаёт требования к закупке (часто чужой тип «или аналог»). "
    "Для каждой позиции выбери конкретную модель/артикул из asset_hits (топ кандидатов каталога), "
    "в том числе как аналог, если характеристики подходят. "
    "Не отказывай только потому, что артикул заказчика не встречается в каталоге. "
    "Если позиция — комплект, а в каталоге подтверждено только основное изделие "
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
    llm_provider: str = "qwen"  # qwen | deepseek | openai
    llm_model: str = "qwen-plus"
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_base_url: str = "https://api.openai.com/v1"

    # Retrieval / оценка
    top_k: int = 3
    max_reqs_per_scope_item: int = 10

    # LangGraph: выбор файлов
    max_tender_files_initial: int = 3
    max_tender_files_total: int = 6
    # Параллельный подбор эталона по позициям (retrieval + match LLM).
    match_parallelism: int = 4

    # Трассировка LLM/retrieval в data/llm_traces/run_*
    llm_trace_enabled: bool = True
    llm_trace_dir: Path = Path("data/llm_traces")

    user_instruction: str = DEFAULT_USER_INSTRUCTION

    # VL для сканов (нет текстового слоя) или принудительно для всех файлов.
    vl_enabled: bool = True

    cache_dir: Path = Path("data/cache")

    # Whole-file извлечение Qwen DashScope (отдельно от match LLM).
    extract_backend: str = "qwen"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    # Whole-file extract: intl (Singapore) — qwen-plus + fileid://; Beijing — qwen-doc-turbo / qwen-long
    qwen_doc_model: str = "qwen-plus"
    qwen_long_model: str = "qwen-long"
    qwen_vl_model: str = "qwen-vl-plus"
    qwen_vl_pages_per_call: int = 2
    qwen_vl_max_pages: int = 80
    qwen_extract_schema_version: str = "3"
    qwen_max_file_mb: int = 150


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
    qwen_extracted_files: Annotated[list[str], operator.add]

    requirements_by_item: list[list[ExtractedRequirement]]
    requirements_stats: dict[str, Any]

    assets_index: Any
    product_catalog: Any
    indexed_files: list[str]
    index_reused: bool

    position_matches: list[ScopePositionMatch]
    verdict: str
    query_selection: dict[str, Any]
