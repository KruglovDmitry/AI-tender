"""LangGraph: сборка пайплайна и точка входа analyze()."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from .services.logging_service import clear_trace, start_trace
from .models import AnalysisReport, PipelineState, ProgressCallback, Settings, get_settings
from .nodes.build_index import node_build_assets_index
from .nodes.finalize import node_finalize
from .nodes.match import node_match_positions
from .nodes.requirements import (
    node_extract_requirements,
    node_load_next_requirement_file,
    route_after_requirements,
)
from .nodes.scope import (
    node_extract_scope,
    node_load_next_scope_file,
    route_after_scope,
)
from .nodes.select_files import node_select_files
from .nodes.verdict import node_build_verdict
from .providers import build_llm

DEFAULT_GRAPH_DIAGRAM_DIR = Path(__file__).resolve().parents[2] / "docs"


def export_graph_diagram(
    compiled: Any,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Сохранить структуру графа: Mermaid (.mmd) и, по возможности, PNG."""
    out_dir = Path(out_dir or DEFAULT_GRAPH_DIAGRAM_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    drawable = compiled.get_graph()
    mmd_path = out_dir / "pipeline_graph.mmd"
    png_path = out_dir / "pipeline_graph.png"
    mmd_path.write_text(drawable.draw_mermaid(), encoding="utf-8")

    written: dict[str, Path] = {"mermaid": mmd_path}
    try:
        png_path.write_bytes(drawable.draw_mermaid_png())
        written["png"] = png_path
    except Exception:
        # PNG тянет mermaid.ink / локальный рендер — без сети не обязателен.
        pass
    return written


# Реэкспорт для тестов / внешних импортов
__all__ = [
    "PipelineState",
    "ProgressCallback",
    "analyze",
    "build_graph",
    "export_graph_diagram",
    "get_compiled_graph",
    "route_after_requirements",
    "route_after_scope",
    "warm_up_graph",
]


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("select_files", node_select_files)
    graph.add_node("load_next_scope_file", node_load_next_scope_file)
    graph.add_node("extract_scope", node_extract_scope)
    graph.add_node("load_next_requirement_file", node_load_next_requirement_file)
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
            "load_next_requirement_file": "load_next_requirement_file",
        },
    )
    graph.add_edge("load_next_requirement_file", "extract_requirements")
    graph.add_conditional_edges(
        "extract_requirements",
        route_after_requirements,
        {
            "load_next_requirement_file": "load_next_requirement_file",
            "build_assets_index": "build_assets_index",
        },
    )
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
    trace = None
    if settings.llm_trace_enabled:
        trace = start_trace(
            settings.llm_trace_dir,
            meta={
                "tender_path": str(tender_path),
                "assets_path": str(assets_path),
                "llm_model": settings.llm_model,
                "max_reqs_per_scope_item": settings.max_reqs_per_scope_item,
                "max_requirement_files": settings.max_requirement_files,
                "match_parallelism": settings.match_parallelism,
                "top_k": settings.top_k,
            },
        )
        if progress:
            progress(f"Трассировка LLM: {trace.path}", 0.02)

    result: dict[str, Any] = {}
    try:
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
                    "requirement_queue": [],
                    "requirement_files_tried": [],
                    "current_requirement_file": "",
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

        query_selection = dict(result.get("query_selection") or {})
        warnings = list(result.get("warnings") or [])
        if trace is not None:
            trace.finish(
                {
                    "positions": len(position_matches),
                    "covered": covered,
                    "requirements": len(all_requirements),
                    "requirements_stats": result.get("requirements_stats") or {},
                }
            )
            query_selection["llm_trace_dir"] = str(trace.path)
            warnings.append(f"Трассировка LLM/retrieval: {trace.path}")

        elapsed_seconds = round(perf_counter() - started, 2)
        return AnalysisReport(
            tender_path=str(tender_path.resolve()),
            assets_path=str(assets_path.resolve()),
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
            summary=summary,
            verdict=verdict,
            position_matches=position_matches,
            warnings=warnings,
            indexed_files=list(result.get("indexed_files") or []),
            index_reused=bool(result.get("index_reused")),
            elapsed_seconds=elapsed_seconds,
            query_selection=query_selection,
        )
    except Exception as exc:
        if trace is not None:
            trace.finish({"error": str(exc)})
        raise
    finally:
        clear_trace()
