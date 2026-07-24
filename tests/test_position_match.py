from unittest.mock import MagicMock

from ai_tender.nodes.scope import scope_has_detailed_list
from ai_tender.models import Evidence, ExtractedRequirement, PositionMatchStatus, ScopePositionMatch
from ai_tender.nodes.common import dedupe_evidence_by_file
from ai_tender.nodes.match import (
    match_scope_position,
    position_to_query_text,
)
from ai_tender.nodes.verdict import build_tender_verdict


def test_scope_has_detailed_list() -> None:
    assert not scope_has_detailed_list([])
    assert not scope_has_detailed_list([{"name": "титул", "qty": None}])
    assert scope_has_detailed_list([{"name": "позиция А", "qty": 24}])
    assert scope_has_detailed_list([{"name": "a"}, {"name": "b"}])


def test_dedupe_evidence_by_file() -> None:
    hits = [
        Evidence(file="a.pdf", location="1", quote="x", score=0.1),
        Evidence(file="a.pdf", location="2", quote="y", score=0.9),
        Evidence(file="b.pdf", location="1", quote="z", score=0.5),
    ]
    out = dedupe_evidence_by_file(hits)
    assert len(out) == 2
    assert out[0].quote == "y"
    assert out[0].score == 0.9


def test_position_to_query_text() -> None:
    reqs = [
        ExtractedRequirement(
            text="Класс точности 0.5S",
            quote="Класс точности 0.5S",
            file="tz.docx",
            location="док",
            kind="specs",
        )
    ]
    text = position_to_query_text("позиция перечня", reqs)
    assert "позиция перечня" in text
    assert "Класс точности 0.5S" in text


def test_position_to_query_prefers_product_first() -> None:
    reqs = [
        ExtractedRequirement(
            text="Ток 5 А",
            quote="ток",
            file="tz.docx",
            location="док",
            kind="specs",
            priority=3,
        ),
        ExtractedRequirement(
            text="Изделие серии X",
            quote="серия X",
            file="tz.docx",
            location="док",
            kind="product",
            priority=2,
        ),
    ]
    text = position_to_query_text("позиция", reqs)
    assert text.index("Изделие серии X") < text.index("Ток 5 А")


def test_match_scope_position_no_hits() -> None:
    llm = MagicMock()
    match = match_scope_position(
        llm,
        scope_item={"name": "позиция", "qty": 24, "unit": "шт."},
        requirements=[],
        asset_hits=[],
    )
    assert match.status == PositionMatchStatus.none
    assert match.product_name == ""
    llm.complete.assert_not_called()


def test_match_scope_position_parses_llm() -> None:
    llm = MagicMock()
    llm.complete.return_value = (
        '{"matched": true, "status": "matched", "product_name": "Модель-10", '
        '"explanation": "Модель подходит по напряжению.", "confidence": 0.8}'
    )
    hit = Evidence(file="asset.pdf", location="стр. 1", quote="Модель-10 6-10 кВ")
    match = match_scope_position(
        llm,
        scope_item={"name": "замена оборудования 6-10 кВ", "qty": 24, "unit": "шт."},
        requirements=[],
        asset_hits=[hit],
    )
    assert match.status == PositionMatchStatus.matched
    assert match.product_name == "Модель-10"
    assert "напряжению" in match.explanation


def test_match_scope_position_bad_json_becomes_none() -> None:
    llm = MagicMock()
    llm.complete.return_value = "это не json {{{"
    hit = Evidence(file="asset.pdf", location="стр. 1", quote="Модель-10")
    match = match_scope_position(
        llm,
        scope_item={"name": "позиция", "qty": 1, "unit": "шт."},
        requirements=[],
        asset_hits=[hit],
    )
    assert match.status == PositionMatchStatus.none
    assert "разобрать" in match.explanation.lower() or "модели" in match.explanation.lower()
    assert llm.complete.call_count == 2  # extract + repair


def test_match_matched_true_with_none_status_becomes_partial() -> None:
    llm = MagicMock()
    llm.complete.return_value = (
        '{"matched": true, "status": "none", "product_name": "Серия A", '
        '"explanation": "Основное изделие есть, комплектующие не подтверждены.", '
        '"confidence": 0.6}'
    )
    hits = [
        Evidence(file="a.pdf", location="1", quote="Серия A", score=0.8),
        Evidence(file="a.pdf", location="2", quote="повтор", score=0.2),
    ]
    match = match_scope_position(
        llm,
        scope_item={"name": "комплект оборудования", "qty": 2, "unit": "компл."},
        requirements=[],
        asset_hits=hits,
    )
    assert match.status == PositionMatchStatus.partial
    assert match.product_name == "Серия A"
    assert len(match.asset_hits) == 1


def test_node_match_positions_parallel_preserves_order(monkeypatch) -> None:
    from ai_tender.nodes import match as match_mod
    from ai_tender.models import Settings

    calls: list[str] = []

    def fake_match_one(*, llm, scope_item, requirements, assets_index, top_k, user_instruction):
        name = str(scope_item.get("name") or "")
        calls.append(name)
        return ScopePositionMatch(scope_name=name, status=PositionMatchStatus.partial)

    monkeypatch.setattr(match_mod, "_match_one_position", fake_match_one)
    settings = Settings(match_parallelism=4)
    state = {
        "settings": settings,
        "llm": MagicMock(),
        "assets_index": object(),
        "scope_items": [{"name": f"поз-{i}"} for i in range(6)],
        "requirements_by_item": [[] for _ in range(6)],
        "progress": None,
    }
    out = match_mod.node_match_positions(state)
    names = [item.scope_name for item in out["position_matches"]]
    assert names == [f"поз-{i}" for i in range(6)]
    assert sorted(calls) == names


def test_node_match_positions_isolates_errors(monkeypatch) -> None:
    from ai_tender.nodes import match as match_mod
    from ai_tender.models import Settings

    def fake_match_one(*, llm, scope_item, requirements, assets_index, top_k, user_instruction):
        name = str(scope_item.get("name") or "")
        if name == "bad":
            raise RuntimeError("boom")
        return ScopePositionMatch(scope_name=name, status=PositionMatchStatus.matched)

    monkeypatch.setattr(match_mod, "_match_one_position", fake_match_one)
    settings = Settings(match_parallelism=2)
    state = {
        "settings": settings,
        "llm": MagicMock(),
        "assets_index": object(),
        "scope_items": [{"name": "ok"}, {"name": "bad"}, {"name": "ok2"}],
        "requirements_by_item": [[], [], []],
        "progress": None,
    }
    out = match_mod.node_match_positions(state)
    names = [item.scope_name for item in out["position_matches"]]
    assert names == ["ok", "bad", "ok2"]
    assert out["position_matches"][1].status == PositionMatchStatus.none
    assert any("bad" in w for w in out["warnings"])


def test_build_tender_verdict_fallback_on_empty_text() -> None:
    llm = MagicMock()
    llm.complete.return_value = '{"suitable": true, "label": "подходит", "verdict": ""}'
    matches = [
        ScopePositionMatch(
            scope_name="позиция A",
            status=PositionMatchStatus.matched,
            product_name="Модель-10",
        ),
        ScopePositionMatch(scope_name="позиция B", status=PositionMatchStatus.none),
    ]
    text = build_tender_verdict(llm, matches, scope_summary="тест")
    assert "1 из 2" in text or "подходит" in text.lower()


def test_build_tender_verdict_fallback_on_llm_error() -> None:
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("api down")
    matches = [
        ScopePositionMatch(scope_name="A", status=PositionMatchStatus.matched),
        ScopePositionMatch(scope_name="B", status=PositionMatchStatus.matched),
        ScopePositionMatch(scope_name="C", status=PositionMatchStatus.none),
    ]
    text = build_tender_verdict(llm, matches)
    assert "2 из 3" in text
