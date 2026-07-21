from ai_tender.graph import (
    build_graph,
    route_after_coverage,
    route_after_scope,
)
from ai_tender.doc_select import TenderFileEntry


def test_build_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_route_after_scope_asks_more_when_empty() -> None:
    state = {
        "scope_items": [],
        "scope_meta": {"needs_more_docs": True},
        "loaded_labels": [],
        "ranked_paths": ["a.pdf", "b.pdf"],
        "scope_queue": ["a.pdf"],
    }
    assert route_after_scope(state) == "load_next_scope_file"


def test_route_after_scope_continues_when_ready() -> None:
    state = {
        "scope_items": [{"name": "ПКУ 6-10 кВ"}],
        "scope_meta": {"needs_more_docs": False},
        "loaded_labels": ["a.pdf"],
        "ranked_paths": ["a.pdf", "b.pdf"],
        "scope_queue": ["a.pdf"],
    }
    assert route_after_scope(state) == "load_remaining_for_reqs"


def test_route_after_coverage_expands_when_missing() -> None:
    state = {
        "expand_done": False,
        "missing_scope_items": ["ПКУ"],
        "loaded_labels": ["a.pdf"],
        "ranked_paths": ["a.pdf"],
        "catalog_entries": [
            TenderFileEntry(path="a.pdf", suffix=".pdf", size_bytes=10, parent=""),
            TenderFileEntry(path="b.pdf", suffix=".pdf", size_bytes=10, parent=""),
        ],
    }
    assert route_after_coverage(state) == "expand_docs"


def test_route_after_coverage_finalizes_when_done() -> None:
    state = {
        "expand_done": True,
        "missing_scope_items": ["ПКУ"],
        "loaded_labels": ["a.pdf"],
        "ranked_paths": ["a.pdf"],
        "catalog_entries": [],
    }
    assert route_after_coverage(state) == "finalize"
