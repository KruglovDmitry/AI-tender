from __future__ import annotations
import operator
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, TypedDict

from llama_index.core import Document
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
from .utils import export_graph_diagram


ProgressCallback = Callable[[str, float], None]


class PipelineState(TypedDict, total=False):
    # runtime deps
    tender_path: str
    assets_path: str
    llm: Any
    settings: Any
    progress: Any
    cleanup_box: Any  # {"inventory": TenderInventory | None} для finally в analyze

    # file selection
    inventory: Any
    catalog_entries: list[TenderFileEntry]
    ranked_paths: list[str]
    doc_selection: dict[str, Any]

    # documents / loading
    loaded_labels: list[str]
    documents: Annotated[list[Document], operator.add]
    scope_queue: list[str]
    scope_files_used: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]

    # scope
    scope_items: list[dict[str, Any]]
    scope_meta: dict[str, Any]

    # requirements
    requirements_by_item: list[list[ExtractedRequirement]]
    requirements_stats: dict[str, Any]

    # assets index (runtime object, не сериализуем)
    assets_index: Any
    indexed_files: list[str]
    index_reused: bool

    # match + verdict
    position_matches: list[ScopePositionMatch]
    verdict: str
    query_selection: dict[str, Any]


def _progress(state: PipelineState, message: str, value: float) -> None:
    callback = state.get("progress")
    if callable(callback):
        callback(message, value)


def _next_unloaded(state: PipelineState) -> str | None:
    loaded = set(state.get("loaded_labels") or [])
    for path in state.get("scope_queue") or []:
        if path not in loaded:
            return path
    for path in state.get("ranked_paths") or []:
        if path not in loaded:
            return path
    return None


def _load_labels(state: PipelineState, labels: list[str]) -> tuple[list[Document], list[str]]:
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


def node_select_files(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    _progress(state, "Каталог и выбор файлов тендера", 0.1)
    inventory, catalog_entries, doc_selection = select_tender_files(
        Path(state["tender_path"]),
        state["llm"],
        use_llm=True,
        max_files=settings.max_tender_files_total,
    )
    box = state.get("cleanup_box")
    if isinstance(box, dict):
        box["inventory"] = inventory

    ranked_paths = ranked_file_paths(doc_selection)
    if not ranked_paths:
        raise ValueError("Не выбрано ни одного файла тендера для анализа")

    initial_queue = ranked_paths[: max(1, settings.max_tender_files_initial)]
    _progress(
        state,
        (
            f"Выбрано {len(ranked_paths)} из {doc_selection.get('catalog_count', 0)} "
            f"файлов ({doc_selection.get('mode', 'llm')})"
        ),
        0.2,
    )
    return {
        "inventory": inventory,
        "catalog_entries": catalog_entries,
        "doc_selection": doc_selection,
        "ranked_paths": ranked_paths,
        "scope_queue": initial_queue,
        "loaded_labels": [],
        "documents": [],
        "scope_files_used": [],
        "scope_items": [],
        "scope_meta": {},
        "warnings": list(doc_selection.get("warnings") or []),
        "requirements_by_item": [],
        "requirements_stats": {},
        "position_matches": [],
        "verdict": "",
        "query_selection": {},
        "indexed_files": [],
        "index_reused": False,
    }


def node_load_next_scope_file(state: PipelineState) -> dict[str, Any]:
    label = _next_unloaded(state)
    if not label:
        return {}
    _progress(state, f"Загрузка файла для scope: {Path(label).name}", 0.28)
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


def node_extract_scope(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    _progress(state, "LangGraph: предмет закупки (перечень позиций)", 0.32)
    docs = list(state.get("documents") or [])
    scope_items, scope_meta = extract_procurement_scope_from_documents(
        docs,
        state["llm"],
        max_chars_per_doc=settings.max_extract_chars_per_doc,
    )
    return {"scope_items": scope_items, "scope_meta": scope_meta}


def route_after_scope(state: PipelineState) -> Literal["load_next_scope_file", "load_remaining"]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    needs_more = bool(scope_meta.get("needs_more_docs", False)) or not scope_has_detailed_list(
        scope_items
    )
    if needs_more and _next_unloaded(state) is not None:
        return "load_next_scope_file"
    return "load_remaining"


def node_load_remaining(state: PipelineState) -> dict[str, Any]:
    loaded = set(state.get("loaded_labels") or [])
    remaining = [path for path in (state.get("ranked_paths") or []) if path not in loaded]
    if not remaining:
        return {}
    _progress(state, f"Догрузка файлов для требований ({len(remaining)})", 0.4)
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


def node_extract_requirements(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    _progress(
        state,
        f"Извлечение требований по позициям (макс. {settings.max_reqs_per_scope_item})",
        0.5,
    )
    reqs_by_item, req_stats = extract_requirements_per_scope_items(
        list(state.get("documents") or []),
        scope_items=scope_items,
        llm=state["llm"],
        max_per_item=settings.max_reqs_per_scope_item,
        max_chars_per_doc=settings.max_extract_chars_per_doc,
        file_order=state.get("ranked_paths") or None,
    )
    warnings = [
        f"Не разобран JSON требований: {err}"
        for err in (req_stats.get("parse_errors") or [])
    ]
    return {
        "requirements_by_item": reqs_by_item,
        "requirements_stats": req_stats,
        "warnings": warnings,
    }


def node_build_assets_index(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    _progress(
        state,
        "Индекс эталонов: чтение PDF и эмбеддинги (после смены файлов — полная пересборка)",
        0.6,
    )
    assets_index, asset_nodes, asset_warnings, index_reused = load_or_build_assets_index(
        Path(state["assets_path"]),
        settings.cache_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_device,
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )
    indexed_files = indexed_file_paths(asset_nodes)
    _progress(
        state,
        (
            f"Индекс эталонов готов ({len(asset_nodes)} чанков, "
            f"{'из кэша' if index_reused else 'построен заново'})"
        ),
        0.7,
    )
    return {
        "assets_index": assets_index,
        "indexed_files": indexed_files,
        "index_reused": index_reused,
        "warnings": asset_warnings,
    }


def node_match_positions(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    reqs_by_item = list(state.get("requirements_by_item") or [])
    assets_index = state.get("assets_index")
    top_k = max(settings.top_k, 5)
    total = max(len(scope_items), 1)
    matches: list[ScopePositionMatch] = []

    for index, scope_item in enumerate(scope_items):
        name = str(scope_item.get("name") or "").strip() or f"позиция {index + 1}"
        _progress(
            state,
            f"Подбор эталона: {index + 1}/{len(scope_items)} — {name[:60]}",
            0.7 + 0.2 * (index / total),
        )
        requirements = reqs_by_item[index] if index < len(reqs_by_item) else []
        hits = retrieve_hits_for_position(name, requirements, assets_index, top_k=top_k)
        asset_evidence = [node_to_evidence(hit.node, hit.score) for hit in hits]
        match = match_scope_position(
            state["llm"],
            scope_item=scope_item,
            requirements=requirements,
            asset_hits=asset_evidence,
            user_instruction=settings.user_instruction,
        )
        matches.append(match)

    return {"position_matches": matches}


def node_build_verdict(state: PipelineState) -> dict[str, Any]:
    _progress(state, "Итоговый вывод по тендеру", 0.92)
    scope_meta = state.get("scope_meta") or {}
    verdict = build_tender_verdict(
        state["llm"],
        list(state.get("position_matches") or []),
        scope_summary=str(scope_meta.get("scope_summary") or ""),
    )
    return {"verdict": verdict}


def node_finalize(state: PipelineState) -> dict[str, Any]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    loaded_labels = list(state.get("loaded_labels") or [])
    doc_selection = dict(state.get("doc_selection") or {})
    position_matches = list(state.get("position_matches") or [])
    reqs_by_item = list(state.get("requirements_by_item") or [])
    all_requirements = [req for bucket in reqs_by_item for req in bucket]
    covered = sum(1 for item in position_matches if item.status.value != "none")
    verdict = str(state.get("verdict") or "")

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
        "requirements_stats": state.get("requirements_stats") or {},
        "loaded_labels": loaded_labels,
        "loaded_labels_count": len(loaded_labels),
        "doc_selection": {
            "mode": doc_selection.get("mode"),
            "catalog_count": doc_selection.get("catalog_count"),
            "selected": doc_selection.get("files"),
            "skipped": doc_selection.get("skip"),
            "loaded": loaded_labels,
            **(
                {"error": doc_selection["error"]}
                if doc_selection.get("error")
                else {}
            ),
        },
    }

    _progress(
        state,
        (
            f"Готово: позиций={len(position_matches)}, закрыто={covered}, "
            f"требований={len(all_requirements)}"
        ),
        0.97,
    )
    return {
        "query_selection": query_selection,
        "verdict": verdict,
    }


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("select_files", node_select_files)
    graph.add_node("load_next_scope_file", node_load_next_scope_file)
    graph.add_node("extract_scope", node_extract_scope)
    graph.add_node("load_remaining", node_load_remaining)
    graph.add_node("extract_requirements", node_extract_requirements)
    graph.add_node("build_assets_index", node_build_assets_index)
    graph.add_node("match_positions", node_match_positions)
    graph.add_node("build_verdict", node_build_verdict)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "select_files")
    graph.add_edge("select_files", "load_next_scope_file")
    graph.add_edge("load_next_scope_file", "extract_scope")
    graph.add_conditional_edges(
        "extract_scope",
        route_after_scope,
        {
            "load_next_scope_file": "load_next_scope_file",
            "load_remaining": "load_remaining",
        },
    )
    graph.add_edge("load_remaining", "extract_requirements")
    graph.add_edge("extract_requirements", "build_assets_index")
    graph.add_edge("build_assets_index", "match_positions")
    graph.add_edge("match_positions", "build_verdict")
    graph.add_edge("build_verdict", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


_COMPILED_GRAPH = None


def get_compiled_graph():
    """Ленивая компиляция; после warm_up_graph() структура уже в памяти."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def warm_up_graph(*, export_diagram: bool = True) -> Any:
    """Скомпилировать граф при старте и сохранить его схему в docs/."""
    compiled = get_compiled_graph()
    if export_diagram:
        export_graph_diagram(compiled)
    return compiled


def analyze(
    tender_path: Path,
    assets_path: Path,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    settings = settings or get_settings()
    started = perf_counter()
    llm = build_llm(settings)
    graph = get_compiled_graph()
    cleanup_box: dict[str, Any] = {"inventory": None}

    try:
        result = graph.invoke(
            {
                "tender_path": str(tender_path),
                "assets_path": str(assets_path),
                "llm": llm,
                "settings": settings,
                "progress": progress,
                "cleanup_box": cleanup_box,
                "loaded_labels": [],
                "documents": [],
                "scope_queue": [],
                "scope_files_used": [],
                "scope_items": [],
                "scope_meta": {},
                "warnings": [],
                "requirements_by_item": [],
                "requirements_stats": {},
                "position_matches": [],
                "verdict": "",
                "query_selection": {},
                "indexed_files": [],
                "index_reused": False,
            }
        )
    finally:
        inventory = cleanup_box.get("inventory")
        if inventory is not None and hasattr(inventory, "cleanup"):
            inventory.cleanup()

    position_matches = list(result.get("position_matches") or [])
    reqs_by_item = list(result.get("requirements_by_item") or [])
    all_requirements = [req for bucket in reqs_by_item for req in bucket]
    covered = sum(1 for item in position_matches if item.status.value != "none")
    verdict = str(result.get("verdict") or "")
    summary = "\n".join(
        [
            verdict,
            f"Позиций: {len(position_matches)}, с вариантом эталона: {covered}.",
            f"Требований извлечено: {len(all_requirements)}.",
        ]
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
        warnings=list(result.get("warnings") or []),
        indexed_files=list(result.get("indexed_files") or []),
        index_reused=bool(result.get("index_reused")),
        elapsed_seconds=elapsed_seconds,
        query_selection=dict(result.get("query_selection") or {}),
        extracted_requirements=all_requirements,
    )
