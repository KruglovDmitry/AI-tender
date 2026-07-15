from ai_tender.providers import _parse_json


def test_parse_json_accepts_plain_object() -> None:
    assert _parse_json('{"status": "compliant"}') == {"status": "compliant"}


def test_parse_json_removes_markdown_fence() -> None:
    assert _parse_json('```json\n{"requirements": []}\n```') == {"requirements": []}
