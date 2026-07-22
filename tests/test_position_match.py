from unittest.mock import MagicMock

from llama_index.core.schema import TextNode

from ai_tender.extract import parse_requirements_per_scope_payload, scope_has_detailed_list
from ai_tender.models import Evidence, PositionMatchStatus
from ai_tender.providers import build_tender_verdict, match_scope_position
from ai_tender.query_select import position_to_query_text


def test_scope_has_detailed_list() -> None:
    assert not scope_has_detailed_list([])
    assert not scope_has_detailed_list([{"name": "титул", "qty": None}])
    assert scope_has_detailed_list([{"name": "ПКУ", "qty": 24}])
    assert scope_has_detailed_list([{"name": "a"}, {"name": "b"}])


def test_parse_requirements_per_scope_payload() -> None:
    scope_items = [
        {"name": "замена ПКУ 6-10 кВ", "qty": 24, "unit": "шт."},
        {"name": "монтаж ПКУ", "qty": 174, "unit": "шт."},
    ]
    source = TextNode(
        text="Класс точности 0.5S\nНоминальное напряжение 6-10 кВ",
        metadata={"file_path": "tz.docx", "location": "документ"},
    )
    data = {
        "items": [
            {
                "scope_index": 0,
                "requirements": [
                    {
                        "text": "Класс точности 0.5S",
                        "quote": "Класс точности 0.5S",
                        "kind": "specs",
                        "priority": 3,
                        "confidence": 0.9,
                    },
                    {
                        "text": "Напряжение 6-10 кВ",
                        "quote": "Номинальное напряжение 6-10 кВ",
                        "kind": "specs",
                        "priority": 2,
                        "confidence": 0.8,
                    },
                ],
            },
            {"scope_index": 1, "requirements": []},
        ]
    }
    buckets = parse_requirements_per_scope_payload(
        data,
        scope_items=scope_items,
        source_node=source,
        max_per_item=10,
    )
    assert len(buckets) == 2
    assert len(buckets[0]) == 2
    assert buckets[0][0].scope_item == "замена ПКУ 6-10 кВ"
    assert buckets[1] == []


def test_parse_requirements_respects_max_per_item() -> None:
    scope_items = [{"name": "ПКУ"}]
    source = TextNode(text="a\nb\nc", metadata={"file_path": "a.docx"})
    data = {
        "items": [
            {
                "scope_index": 0,
                "requirements": [
                    {"text": f"req {i}", "quote": f"req {i}", "kind": "specs", "priority": 2}
                    for i in range(5)
                ],
            }
        ]
    }
    buckets = parse_requirements_per_scope_payload(
        data,
        scope_items=scope_items,
        source_node=source,
        max_per_item=2,
    )
    assert len(buckets[0]) == 2


def test_position_to_query_text() -> None:
    from ai_tender.models import ExtractedRequirement

    reqs = [
        ExtractedRequirement(
            text="Класс точности 0.5S",
            quote="Класс точности 0.5S",
            file="tz.docx",
            location="док",
        )
    ]
    text = position_to_query_text("замена ПКУ 6-10 кВ", reqs)
    assert "замена ПКУ 6-10 кВ" in text
    assert "Класс точности 0.5S" in text


def test_match_scope_position_no_hits() -> None:
    llm = MagicMock()
    match = match_scope_position(
        llm,
        scope_item={"name": "ПКУ", "qty": 24, "unit": "шт."},
        requirements=[],
        asset_hits=[],
    )
    assert match.status == PositionMatchStatus.none
    assert match.product_name == ""
    llm.complete.assert_not_called()


def test_match_scope_position_parses_llm() -> None:
    llm = MagicMock()
    llm.complete.return_value = (
        '{"matched": true, "status": "matched", "product_name": "ПКУ-10", '
        '"explanation": "Модель подходит по напряжению.", "confidence": 0.8}'
    )
    hit = Evidence(file="asset.pdf", location="стр. 1", quote="ПКУ-10 6-10 кВ")
    match = match_scope_position(
        llm,
        scope_item={"name": "замена ПКУ 6-10 кВ", "qty": 24, "unit": "шт."},
        requirements=[],
        asset_hits=[hit],
    )
    assert match.status == PositionMatchStatus.matched
    assert match.product_name == "ПКУ-10"
    assert "напряжению" in match.explanation


def test_build_tender_verdict_fallback_on_empty_text() -> None:
    llm = MagicMock()
    llm.complete.return_value = '{"suitable": true, "label": "подходит", "verdict": ""}'
    from ai_tender.models import ScopePositionMatch

    matches = [
        ScopePositionMatch(
            scope_name="ПКУ",
            status=PositionMatchStatus.matched,
            product_name="ПКУ-10",
        ),
        ScopePositionMatch(scope_name="ТТ", status=PositionMatchStatus.none),
    ]
    text = build_tender_verdict(llm, matches, scope_summary="тест")
    assert "1 из 2" in text or "подходит" in text.lower()
