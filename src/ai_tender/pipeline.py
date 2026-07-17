from collections.abc import Callable
from pathlib import Path

from .index import (
    indexed_file_paths,
    load_or_build_assets_index,
    load_tender_nodes,
    node_to_evidence,
    retrieve_candidates,
    select_query_nodes,
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

    update(
        "Индекс эталонов: чтение PDF и эмбеддинги (после смены файлов — полная пересборка, "
        "это долго; дальше будет из кэша)",
        0.08,
    )
    assets_index, asset_nodes, asset_warnings, index_reused = load_or_build_assets_index(
        assets_path,
        settings.cache_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_device,
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )
    indexed_files = indexed_file_paths(asset_nodes)
    update(
        (
            f"Индекс эталонов готов ({len(asset_nodes)} чанков, "
            f"{'из кэша' if index_reused else 'построен заново'})"
        ),
        0.3,
    )

    update("Чтение тендерной документации", 0.35)
    tender_nodes, tender_warnings = load_tender_nodes(
        tender_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )

    update("Поиск подтверждений в эталоне (hybrid retrieval)", 0.6)
    query_nodes = select_query_nodes(tender_nodes, settings.max_tender_queries)
    candidates = retrieve_candidates(
        query_nodes,
        assets_index,
        top_k=settings.top_k,
        min_score=settings.min_retrieval_score,
    )

    update("Оценка применимости через LLM", 0.85)
    llm = build_llm(settings)
    summary, findings = assess_findings(
        llm,
        candidates,
        node_to_evidence,
        user_instruction=settings.user_instruction,
        max_findings=settings.max_findings,
    )

    update("Сбор результата", 0.98)
    return AnalysisReport(
        tender_path=str(tender_path.resolve()),
        assets_path=str(assets_path.resolve()),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        summary=summary,
        findings=findings,
        warnings=asset_warnings + tender_warnings,
        indexed_files=indexed_files,
        index_reused=index_reused,
    )
