from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, TypedDict

from llama_index.core import Document
from llama_index.core.llms import LLM
from langgraph.graph import END, START, StateGraph

from .anchors import refine_requirement_anchors
from .doc_select import (
    TenderFileEntry,
    ranked_file_paths,
    select_files_heuristic,
    select_tender_files,
)
from .extract import (
    extract_procurement_scope_from_documents,
    extract_requirements_for_scope_from_documents,
)
from .index import (
    indexed_file_paths,
    load_or_build_assets_index,
    node_to_evidence,
)
from .loaders import load_documents
from .models import AnalysisReport, ExtractedRequirement, Settings, get_settings
from .providers import (
    assess_findings,
    build_compact_summary,
    build_llm,
    select_important_findings,
)
from .query_select import retrieve_requirement_candidates


class ScopeGraphState(TypedDict, total=False):
    # runtime deps (передаём через invoke)
    tender_path: str
    inventory: Any
    catalog_entries: list[TenderFileEntry]
    ranked_paths: list[str]
    llm: Any
    settings: Any

    # working state
    loaded_labels: list[str]
    documents: Annotated[list[Document], operator.add]
    scope_queue: list[str]
    scope_files_used: Annotated[list[str], operator.add]
    scope_items: list[dict[str, Any]]
    scope_meta: dict[str, Any]
    requirements: list[ExtractedRequirement]
    requirements_stats: dict[str, Any]
    missing_scope_items: list[str]
    expand_done: bool
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


def route_after_scope(state: ScopeGraphState) -> Literal["load_next_scope_file", "load_remaining_for_reqs"]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    needs_more = bool(scope_meta.get("needs_more_docs", False)) or not scope_items
    if needs_more and _next_unloaded(state) is not None:
        return "load_next_scope_file"
    return "load_remaining_for_reqs"


def node_load_remaining_for_reqs(state: ScopeGraphState) -> dict[str, Any]:
    loaded = set(state.get("loaded_labels") or [])
    remaining = [p for p in (state.get("ranked_paths") or []) if p not in loaded]
    if not remaining:
        return {}
    docs, warns = _load_labels(state, remaining)
    loaded_list = list(state.get("loaded_labels") or [])
    for label in remaining:
        if label not in loaded_list:
            loaded_list.append(label)
    return {
        "documents": docs,
        "loaded_labels": loaded_list,
        "warnings": warns,
    }


def node_extract_requirements(state: ScopeGraphState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    scope_meta = dict(state.get("scope_meta") or {})
    if not scope_items:
        scope_items = [{"name": "предмет закупки", "confidence": 0.0, "quote": ""}]
        scope_meta["needs_more_docs"] = True

    requirements, req_stats = extract_requirements_for_scope_from_documents(
        list(state.get("documents") or []),
        scope_items=scope_items,
        llm=state["llm"],
        limit=settings.max_tender_queries,
        max_chars_per_doc=settings.max_extract_chars_per_doc,
        file_order=state.get("ranked_paths") or None,
    )
    return {
        "scope_items": scope_items,
        "scope_meta": scope_meta,
        "requirements": requirements,
        "requirements_stats": req_stats,
    }


def node_check_coverage(state: ScopeGraphState) -> dict[str, Any]:
    scope_items = state.get("scope_items") or []
    requirements = state.get("requirements") or []
    coverage = {str(item.get("name", "")): False for item in scope_items}
    for req in requirements:
        if req.scope_item and req.scope_item in coverage:
            coverage[req.scope_item] = True
    missing = [name for name, ok in coverage.items() if name and not ok]
    return {"missing_scope_items": missing}


def route_after_coverage(state: ScopeGraphState) -> Literal["expand_docs", "finalize"]:
    if state.get("expand_done"):
        return "finalize"
    missing = state.get("missing_scope_items") or []
    if not missing:
        return "finalize"
    loaded = set(state.get("loaded_labels") or [])
    catalog = state.get("catalog_entries") or []
    if any(entry.path not in loaded for entry in catalog):
        return "expand_docs"
    return "finalize"


def node_expand_docs(state: ScopeGraphState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    loaded = set(state.get("loaded_labels") or [])
    catalog = list(state.get("catalog_entries") or [])
    extra_cap = min(len(catalog), settings.max_tender_files_total + 4)
    extra_pick = select_files_heuristic(catalog, max_files=extra_cap)
    extra_paths = [
        str(item.get("path"))
        for item in extra_pick.get("files", [])
        if item.get("path") and str(item.get("path")) not in loaded
    ][:4]
    if not extra_paths:
        return {"expand_done": True}

    docs, warns = _load_labels(state, extra_paths)
    loaded_list = list(state.get("loaded_labels") or [])
    for label in extra_paths:
        if label not in loaded_list:
            loaded_list.append(label)
    return {
        "documents": docs,
        "loaded_labels": loaded_list,
        "warnings": warns,
        "expand_done": True,
    }


def node_finalize(state: ScopeGraphState) -> dict[str, Any]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    missing = state.get("missing_scope_items") or []
    covered = max(0, len(scope_items) - len(missing))
    loaded_labels = list(state.get("loaded_labels") or [])
    query_selection = {
        "scope_first": True,
        "graph": "langgraph",
        "scope": {
            "items": [item.get("name") for item in scope_items],
            "summary": scope_meta.get("scope_summary"),
            "overall_confidence": scope_meta.get("overall_confidence"),
            "needs_more_docs": scope_meta.get("needs_more_docs"),
            "missing_signals": scope_meta.get("missing_signals"),
            "files_used": list(state.get("scope_files_used") or []),
        },
        "requirements_stats": state.get("requirements_stats") or {},
        "scope_coverage": {
            "covered": covered,
            "total": len(scope_items),
            "missing": missing,
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
    graph.add_node("load_remaining_for_reqs", node_load_remaining_for_reqs)
    graph.add_node("extract_requirements", node_extract_requirements)
    graph.add_node("check_coverage", node_check_coverage)
    graph.add_node("expand_docs", node_expand_docs)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "load_next_scope_file")
    graph.add_edge("load_next_scope_file", "extract_scope")
    graph.add_conditional_edges(
        "extract_scope",
        route_after_scope,
        {
            "load_next_scope_file": "load_next_scope_file",
            "load_remaining_for_reqs": "load_remaining_for_reqs",
        },
    )
    graph.add_edge("load_remaining_for_reqs", "extract_requirements")
    graph.add_edge("extract_requirements", "check_coverage")
    graph.add_conditional_edges(
        "check_coverage",
        route_after_coverage,
        {
            "expand_docs": "expand_docs",
            "finalize": "finalize",
        },
    )
    graph.add_edge("expand_docs", "extract_requirements")
    graph.add_edge("finalize", END)
    return graph.compile()


_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def _run_extract_graph(
    *,
    tender_path: Any,
    tender_inventory: Any,
    catalog_entries: list[TenderFileEntry],
    ranked_paths: list[str],
    llm: LLM,
    settings: Settings,
) -> tuple[list[ExtractedRequirement], dict[str, Any], list[Document]]:
    """Запуск LangGraph: предмет закупки → требования."""
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
            "requirements": [],
            "requirements_stats": {},
            "missing_scope_items": [],
            "expand_done": False,
            "warnings": [],
            "query_selection": {},
        }
    )

    requirements = list(result.get("requirements") or [])
    query_selection = dict(result.get("query_selection") or {})
    documents = list(result.get("documents") or [])
    return requirements, query_selection, documents


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

    update("Каталог и выбор файлов тендера", 0.35)
    llm = build_llm(settings)

    tender_inventory = None
    requirements: list[ExtractedRequirement] = []
    query_selection: dict = {}
    tender_warnings: list[str] = []
    ranked_paths: list[str] = []

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
            0.38,
        )
        if not ranked_paths:
            raise ValueError("Не выбрано ни одного файла тендера для анализа")

        update("LangGraph: предмет закупки → требования", 0.4)
        requirements, query_selection, _documents = _run_extract_graph(
            tender_path=tender_path,
            tender_inventory=tender_inventory,
            catalog_entries=catalog_entries,
            ranked_paths=ranked_paths,
            llm=llm,
            settings=settings,
        )
        tender_warnings = query_selection.pop("tender_warnings", []) or []
        loaded = query_selection.get("loaded_labels") or ranked_paths
        query_selection["doc_selection"] = {
            "mode": doc_selection.get("mode"),
            "catalog_count": doc_selection.get("catalog_count"),
            "selected": doc_selection.get("files"),
            "skipped": doc_selection.get("skip"),
            "loaded": loaded,
        }
        if doc_selection.get("error"):
            query_selection["doc_selection"]["error"] = doc_selection["error"]
    finally:
        if tender_inventory is not None:
            tender_inventory.cleanup()

    requirements = refine_requirement_anchors(requirements, tender_path)
    query_selection["anchored"] = sum(
        1 for item in requirements if item.line_start is not None
    )
    query_selection["top_requirements"] = [
        {
            "priority": item.priority,
            "confidence": item.confidence,
            "kind": item.kind,
            "scope_item": item.scope_item,
            "text": item.text[:160],
            "location": item.location,
            "line_start": item.line_start,
            "line_end": item.line_end,
        }
        for item in requirements[:5]
    ]

    update(f"Поиск и оценка требований в эталоне ({len(requirements)})", 0.7)
    candidates = retrieve_requirement_candidates(
        requirements, assets_index, top_k=settings.top_k
    )
    _, all_findings = assess_findings(
        llm,
        candidates,
        node_to_evidence,
        user_instruction=settings.user_instruction,
        max_findings=settings.max_findings,
        select_important=False,
    )
    findings = select_important_findings(all_findings, max_findings=settings.max_findings)
    summary = build_compact_summary(all_findings, findings)
    query_selection["checked"] = len(candidates)

    req_stats = query_selection.get("requirements_stats") or {}
    truncated = req_stats.get("truncated_files") or []
    extra = f", обрезано файлов: {len(truncated)}" if truncated else ""
    update(
        (
            f"Готово: требований={len(requirements)}, проверено={len(candidates)}, "
            f"в отчёте={len(findings)}, якоря={query_selection.get('anchored', 0)}{extra}"
        ),
        0.95,
    )

    elapsed_seconds = round(perf_counter() - started, 2)
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
        elapsed_seconds=elapsed_seconds,
        query_selection=query_selection,
        extracted_requirements=requirements,
    )
