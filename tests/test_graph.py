from pathlib import Path

from ai_tender.graph import build_graph, route_after_scope, warm_up_graph
from ai_tender.extract import scope_has_detailed_list
from ai_tender.utils import export_graph_diagram


def test_build_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_warm_up_graph_caches_compiled() -> None:
    g1 = warm_up_graph(export_diagram=False)
    g2 = warm_up_graph(export_diagram=False)
    assert g1 is g2


def test_export_graph_diagram(tmp_path: Path) -> None:
    written = export_graph_diagram(build_graph(), out_dir=tmp_path)
    assert written["mermaid"].is_file()
    text = written["mermaid"].read_text(encoding="utf-8")
    assert "select_files" in text
    assert "extract_scope" in text
    assert "match_positions" in text
    if "png" in written:
        assert written["png"].is_file()
        assert written["png"].stat().st_size > 0


def test_route_after_scope_asks_more_when_empty() -> None:
    state = {
        "scope_items": [],
        "scope_meta": {"needs_more_docs": True},
        "loaded_labels": [],
        "ranked_paths": ["a.pdf", "b.pdf"],
        "scope_queue": ["a.pdf"],
    }
    assert route_after_scope(state) == "load_next_scope_file"


def test_route_after_scope_asks_more_for_title_only() -> None:
    state = {
        "scope_items": [{"name": "Выполнение ПИР, СМР по титулу…", "qty": None}],
        "scope_meta": {"needs_more_docs": False},
        "loaded_labels": ["a.pdf"],
        "ranked_paths": ["a.pdf", "b.pdf"],
        "scope_queue": ["a.pdf"],
    }
    assert route_after_scope(state) == "load_next_scope_file"


def test_route_after_scope_continues_to_load_remaining() -> None:
    state = {
        "scope_items": [
            {"name": "замена ПКУ 6-10 кВ", "qty": 24, "unit": "шт."},
            {"name": "монтаж ПКУ 6-10 кВ", "qty": 174, "unit": "шт."},
        ],
        "scope_meta": {"needs_more_docs": False},
        "loaded_labels": ["a.pdf", "b.pdf"],
        "ranked_paths": ["a.pdf", "b.pdf"],
        "scope_queue": ["a.pdf"],
    }
    assert route_after_scope(state) == "load_remaining"


def test_scope_has_detailed_list() -> None:
    assert not scope_has_detailed_list([])
    assert not scope_has_detailed_list([{"name": "титул", "qty": None}])
    assert scope_has_detailed_list([{"name": "ПКУ", "qty": 24}])
    assert scope_has_detailed_list(
        [{"name": "a", "qty": None}, {"name": "b", "qty": None}]
    )
