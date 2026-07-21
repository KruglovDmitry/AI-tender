"""Анализ тендера через LangGraph (пока: предмет закупки / перечень позиций).

Поток:
  START
    → load_next_scope_file
    → extract_scope
    → (needs_more?) load_next_scope_file | finalize
    → END

Снаружи: analyze() — выбор файлов → граф scope → отчёт.
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
    scope_has_detailed_list,
)
from .loaders import load_documents
from .models import AnalysisReport, Settings, get_settings
from .providers import build_llm


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
) -> dict[str, Any]:
    """Запуск LangGraph: только предмет закупки / перечень позиций."""
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
    return dict(result.get("query_selection") or {})


ProgressCallback = Callable[[str, float], None]


def analyze(
    tender_path: Path,
    assets_path: Path,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    """Пока извлекает только предмет закупки; assets_path сохранён для следующего этапа."""
    del assets_path  # следующий этап: сверка позиций с эталоном
    settings = settings or get_settings()
    started = perf_counter()

    def update(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    update("Каталог и выбор файлов тендера", 0.15)
    llm = build_llm(settings)

    tender_inventory = None
    query_selection: dict = {}
    tender_warnings: list[str] = []

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
            0.3,
        )
        if not ranked_paths:
            raise ValueError("Не выбрано ни одного файла тендера для анализа")

        update("LangGraph: предмет закупки (перечень позиций)", 0.45)
        query_selection = _run_scope_graph(
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

    scope = query_selection.get("scope") or {}
    items = scope.get("items") or []
    update(
        f"Готово: позиций перечня={len(items)}, файлов={query_selection.get('loaded_labels_count', 0)}",
        0.95,
    )

    summary = (scope.get("summary") or "").strip()
    if items:
        lines = []
        if summary:
            lines.append(summary)
        lines.append(f"Позиций перечня: {len(items)}.")
        summary = "\n".join(lines)

    elapsed_seconds = round(perf_counter() - started, 2)
    return AnalysisReport(
        tender_path=str(tender_path.resolve()),
        assets_path="",
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        summary=summary,
        findings=[],
        warnings=tender_warnings,
        indexed_files=[],
        index_reused=False,
        elapsed_seconds=elapsed_seconds,
        query_selection=query_selection,
        extracted_requirements=[],
    )
