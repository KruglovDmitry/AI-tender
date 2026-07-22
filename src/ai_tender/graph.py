"""Анализ тендера через LangGraph (scope) + requirements/match.

Поток scope:
  START → load_next_scope_file → extract_scope → (needs_more?) load | finalize → END

analyze(): выбор файлов → scope → требования по позициям → индекс эталонов
         → match по позициям → итоговый вердикт.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, TypedDict

from llama_index.core import Document
from llama_index.core.llms import LLM
from langgraph.graph import END, START, StateGraph

from .doc_select import (
    TenderFileEntry,
    ranked_file_paths,
    select_tender_files,
)
from .extract import (
    extract_procurement_scope_from_documents,
    extract_requirements_per_scope_items,
    scope_has_detailed_list,
)
from .index import (
    indexed_file_paths,
    load_or_build_assets_index,
    node_to_evidence,
)
from .loaders import load_documents
from .models import (
    AnalysisReport,
    ExtractedRequirement,
    ScopePositionMatch,
    Settings,
    get_settings,
)
from .providers import build_llm, build_tender_verdict, match_scope_position
from .query_select import retrieve_hits_for_position


class ScopeGraphState(TypedDict, total=False):
    tender_path: str
    inventory: Any
    catalog_entries: list[TenderFileEntry]
    ranked_paths: list[str]
    llm: Any
    settings: Any

    loaded_labels: list[str]
    documents: Annotated[list[Document], operator.add]
    scope_queue: list[str]
    scope_files_used: Annotated[list[str], operator.add]
    scope_items: list[dict[str, Any]]
    scope_meta: dict[str, Any]
    warnings: Annotated[list[str], operator.add]
    query_selection: dict[str, Any]


def _next_unloaded(state: ScopeGraphState) -> str | None:
    loaded = set(state.get("loaded_labels") or [])
    for path in state.get("scope_queue") or []:
        if path not in loaded:
            return path
    for path in state.get("ranked_paths") or []:
        if path not in loaded:
            return path
    return None


def _load_labels(state: ScopeGraphState, labels: list[str]) -> tuple[list[Document], list[str]]:
    if not labels:
        return [], []
    settings: Settings = state["settings"]
    docs, warns = load_documents(
        Path(state["tender_path"]),
        corpus="tender",
        inventory=state.get("inventory"),
        only_labels=set(labels),
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )
    return docs, warns


def node_load_next_scope_file(state: ScopeGraphState) -> dict[str, Any]:
    label = _next_unloaded(state)
    if not label:
        return {}
    docs, warns = _load_labels(state, [label])
    loaded = list(state.get("loaded_labels") or [])
    if label not in loaded:
        loaded.append(label)
    return {
        "documents": docs,
        "loaded_labels": loaded,
        "scope_files_used": [label],
        "warnings": warns,
    }


def node_extract_scope(state: ScopeGraphState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    docs = list(state.get("documents") or [])
    scope_items, scope_meta = extract_procurement_scope_from_documents(
        docs,
        state["llm"],
        max_chars_per_doc=settings.max_extract_chars_per_doc,
    )
    return {"scope_items": scope_items, "scope_meta": scope_meta}


def route_after_scope(state: ScopeGraphState) -> Literal["load_next_scope_file", "finalize"]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    needs_more = bool(scope_meta.get("needs_more_docs", False)) or not scope_has_detailed_list(
        scope_items
    )
    if needs_more and _next_unloaded(state) is not None:
        return "load_next_scope_file"
    return "finalize"


def node_finalize(state: ScopeGraphState) -> dict[str, Any]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    loaded_labels = list(state.get("loaded_labels") or [])
    query_selection = {
        "scope_first": True,
        "graph": "langgraph",
        "scope": {
            "items": scope_items,
            "summary": scope_meta.get("scope_summary"),
            "overall_confidence": scope_meta.get("overall_confidence"),
            "needs_more_docs": scope_meta.get("needs_more_docs"),
            "missing_signals": scope_meta.get("missing_signals"),
            "files_used": list(state.get("scope_files_used") or []),
        },
        "loaded_labels": loaded_labels,
        "loaded_labels_count": len(loaded_labels),
        "tender_warnings": list(state.get("warnings") or []),
    }
    return {"query_selection": query_selection}


def build_graph():
    graph = StateGraph(ScopeGraphState)
    graph.add_node("load_next_scope_file", node_load_next_scope_file)
    graph.add_node("extract_scope", node_extract_scope)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "load_next_scope_file")
    graph.add_edge("load_next_scope_file", "extract_scope")
    graph.add_conditional_edges(
        "extract_scope",
        route_after_scope,
        {
            "load_next_scope_file": "load_next_scope_file",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def _run_scope_graph(
    *,
    tender_path: Any,
    tender_inventory: Any,
    catalog_entries: list[TenderFileEntry],
    ranked_paths: list[str],
    llm: LLM,
    settings: Settings,
) -> tuple[dict[str, Any], list[Document], list[str]]:
    """Запуск LangGraph: предмет закупки. Возвращает query_selection, docs, loaded."""
    initial_queue = ranked_paths[: max(1, settings.max_tender_files_initial)]
    graph = get_compiled_graph()
    result = graph.invoke(
        {
            "tender_path": str(tender_path),
            "inventory": tender_inventory,
            "catalog_entries": catalog_entries,
            "ranked_paths": ranked_paths,
            "llm": llm,
            "settings": settings,
            "loaded_labels": [],
            "documents": [],
            "scope_queue": initial_queue,
            "scope_files_used": [],
            "scope_items": [],
            "scope_meta": {},
            "warnings": [],
            "query_selection": {},
        }
    )
    query_selection = dict(result.get("query_selection") or {})
    documents = list(result.get("documents") or [])
    loaded_labels = list(result.get("loaded_labels") or [])
    return query_selection, documents, loaded_labels


ProgressCallback = Callable[[str, float], None]


def analyze(
    tender_path: Path,
    assets_path: Path,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    settings = settings or get_settings()
    started = perf_counter()

    def update(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    update("Каталог и выбор файлов тендера", 0.1)
    llm = build_llm(settings)

    tender_inventory = None
    query_selection: dict = {}
    tender_warnings: list[str] = []
    documents: list[Document] = []
    ranked_paths: list[str] = []
    scope_items: list[dict[str, Any]] = []
    reqs_by_item: list[list[ExtractedRequirement]] = []
    position_matches: list[ScopePositionMatch] = []
    verdict = ""
    asset_warnings: list[str] = []
    indexed_files: list[str] = []
    index_reused = False
    all_requirements: list[ExtractedRequirement] = []

    try:
        tender_inventory, catalog_entries, doc_selection = select_tender_files(
            tender_path,
            llm,
            use_llm=True,
            max_files=settings.max_tender_files_total,
        )
        ranked_paths = ranked_file_paths(doc_selection)
        update(
            (
                f"Выбрано {len(ranked_paths)} из {doc_selection.get('catalog_count', 0)} "
                f"файлов ({doc_selection.get('mode', 'llm')})"
            ),
            0.2,
        )
        if not ranked_paths:
            raise ValueError("Не выбрано ни одного файла тендера для анализа")

        update("LangGraph: предмет закупки (перечень позиций)", 0.3)
        query_selection, documents, loaded_labels = _run_scope_graph(
            tender_path=tender_path,
            tender_inventory=tender_inventory,
            catalog_entries=catalog_entries,
            ranked_paths=ranked_paths,
            llm=llm,
            settings=settings,
        )
        tender_warnings = query_selection.pop("tender_warnings", []) or []
        loaded = loaded_labels or query_selection.get("loaded_labels") or ranked_paths
        query_selection["doc_selection"] = {
            "mode": doc_selection.get("mode"),
            "catalog_count": doc_selection.get("catalog_count"),
            "selected": doc_selection.get("files"),
            "skipped": doc_selection.get("skip"),
            "loaded": loaded,
        }
        if doc_selection.get("error"):
            query_selection["doc_selection"]["error"] = doc_selection["error"]

        scope = query_selection.get("scope") or {}
        scope_items = list(scope.get("items") or [])

        # Догрузить оставшиеся ranked-файлы для extract требований.
        remaining = [path for path in ranked_paths if path not in set(loaded_labels)]
        if remaining:
            update(f"Догрузка файлов для требований ({len(remaining)})", 0.4)
            extra_docs, extra_warns = load_documents(
                tender_path,
                corpus="tender",
                inventory=tender_inventory,
                only_labels=set(remaining),
                ocr_enabled=settings.ocr_enabled,
                ocr_languages=settings.ocr_languages,
            )
            documents.extend(extra_docs)
            tender_warnings.extend(extra_warns)
            loaded_labels = list(loaded_labels) + remaining
            query_selection["loaded_labels"] = loaded_labels
            query_selection["loaded_labels_count"] = len(loaded_labels)
            query_selection["doc_selection"]["loaded"] = loaded_labels

        update(
            f"Извлечение требований по позициям (макс. {settings.max_reqs_per_scope_item})",
            0.5,
        )
        reqs_by_item, req_stats = extract_requirements_per_scope_items(
            documents,
            scope_items=scope_items,
            llm=llm,
            max_per_item=settings.max_reqs_per_scope_item,
            max_chars_per_doc=settings.max_extract_chars_per_doc,
            file_order=ranked_paths,
        )
        query_selection["requirements_stats"] = req_stats
        for err in req_stats.get("parse_errors") or []:
            tender_warnings.append(f"Не разобран JSON требований: {err}")
        for bucket in reqs_by_item:
            all_requirements.extend(bucket)
    finally:
        if tender_inventory is not None:
            tender_inventory.cleanup()

    update(
        "Индекс эталонов: чтение PDF и эмбеддинги (после смены файлов — полная пересборка)",
        0.6,
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
        0.7,
    )

    top_k = max(settings.top_k, 5)
    total_items = max(len(scope_items), 1)
    for index, scope_item in enumerate(scope_items):
        name = str(scope_item.get("name") or "").strip() or f"позиция {index + 1}"
        update(f"Подбор эталона: {index + 1}/{len(scope_items)} — {name[:60]}", 0.7 + 0.2 * (index / total_items))
        requirements = reqs_by_item[index] if index < len(reqs_by_item) else []
        hits = retrieve_hits_for_position(name, requirements, assets_index, top_k=top_k)
        asset_evidence = [node_to_evidence(hit.node, hit.score) for hit in hits]
        match = match_scope_position(
            llm,
            scope_item=scope_item,
            requirements=requirements,
            asset_hits=asset_evidence,
            user_instruction=settings.user_instruction,
        )
        position_matches.append(match)

    update("Итоговый вывод по тендеру", 0.92)
    scope_summary = str((query_selection.get("scope") or {}).get("summary") or "")
    verdict = build_tender_verdict(llm, position_matches, scope_summary=scope_summary)

    covered = sum(1 for item in position_matches if item.status.value != "none")
    summary_lines = [
        verdict,
        f"Позиций: {len(position_matches)}, с вариантом эталона: {covered}.",
        f"Требований извлечено: {len(all_requirements)}.",
    ]
    summary = "\n".join(summary_lines)

    update(
        f"Готово: позиций={len(position_matches)}, закрыто={covered}, требований={len(all_requirements)}",
        0.97,
    )

    elapsed_seconds = round(perf_counter() - started, 2)
    return AnalysisReport(
        tender_path=str(tender_path.resolve()),
        assets_path=str(assets_path.resolve()),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        summary=summary,
        verdict=verdict,
        findings=[],
        position_matches=position_matches,
        warnings=asset_warnings + tender_warnings,
        indexed_files=indexed_files,
        index_reused=index_reused,
        elapsed_seconds=elapsed_seconds,
        query_selection=query_selection,
        extracted_requirements=all_requirements,
    )
