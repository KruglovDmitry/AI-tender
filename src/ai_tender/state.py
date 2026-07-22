"""Состояние LangGraph-пайплайна."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from llama_index.core import Document

from .models import ExtractedRequirement, ScopePositionMatch


ProgressCallback = Callable[[str, float], None]


class PipelineState(TypedDict, total=False):
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
