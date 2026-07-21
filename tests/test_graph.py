from ai_tender.graph import build_graph, route_after_scope
from ai_tender.extract import scope_has_detailed_list


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


def test_route_after_scope_asks_more_for_title_only() -> None:
    state = {
        "scope_items": [{"name": "Выполнение ПИР, СМР по титулу…", "qty": None}],
        "scope_meta": {"needs_more_docs": False},
        "loaded_labels": ["a.pdf"],
        "ranked_paths": ["a.pdf", "b.pdf"],
        "scope_queue": ["a.pdf"],
    }
    assert route_after_scope(state) == "load_next_scope_file"


def test_route_after_scope_finalizes_when_detailed() -> None:
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
    assert route_after_scope(state) == "finalize"


def test_scope_has_detailed_list() -> None:
    assert not scope_has_detailed_list([])
    assert not scope_has_detailed_list([{"name": "титул", "qty": None}])
    assert scope_has_detailed_list([{"name": "ПКУ", "qty": 24}])
    assert scope_has_detailed_list(
        [{"name": "a", "qty": None}, {"name": "b", "qty": None}]
    )
