from collections.abc import Callable
from pathlib import Path

from .index import (
    build_tender_index,
    load_or_build_assets_index,
    node_to_evidence,
    retrieve_candidates,
    select_asset_query_nodes,
)
from .models import AnalysisReport, Settings, get_settings
from .providers import assess_findings, build_llm

ProgressCallback = Callable[[str, float], None]


def analyze(
    tender_path: Path,
    assets_path: Path,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    settings = settings or get_settings()

    def update(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    update("Загрузка и подготовка индекса эталонов", 0.1)
    _, asset_nodes, asset_warnings, index_reused = load_or_build_assets_index(
        assets_path,
        settings.cache_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_device,
    )

    update("Индексация тендерной документации", 0.35)
    tender_index, _, tender_warnings = build_tender_index(
        tender_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    update("Поиск вхождений эталона в тендер (hybrid retrieval)", 0.6)
    query_nodes = select_asset_query_nodes(asset_nodes, settings.max_asset_queries)
    candidates = retrieve_candidates(
        query_nodes,
        tender_index,
        top_k=settings.top_k,
        min_score=settings.min_retrieval_score,
    )

    update("Оценка соответствия через LLM", 0.85)
    llm = build_llm(settings)
    summary, findings = assess_findings(llm, candidates, node_to_evidence)

    update("Сбор результата", 0.98)
    return AnalysisReport(
        tender_path=str(tender_path.resolve()),
        assets_path=str(assets_path.resolve()),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        summary=summary,
        findings=findings,
        warnings=asset_warnings + tender_warnings,
        index_reused=index_reused,
    )

