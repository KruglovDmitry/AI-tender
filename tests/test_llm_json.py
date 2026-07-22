import json

from ai_tender.providers import parse_llm_json, try_parse_llm_json


def test_parse_llm_json_strips_fence() -> None:
    data = parse_llm_json('```json\n{"a": 1}\n```')
    assert data == {"a": 1}


def test_parse_llm_json_trailing_comma() -> None:
    data = parse_llm_json('{"items": [{"x": 1},],}')
    assert data["items"][0]["x"] == 1


def test_parse_llm_json_control_chars() -> None:
    raw = '{"text": "line1\x0bline2"}'
    data = parse_llm_json(raw)
    assert "line1" in data["text"]


def test_try_parse_llm_json_returns_none_on_garbage() -> None:
    assert try_parse_llm_json("not json at all {") is None


def test_parse_llm_json_still_raises_on_unfixable() -> None:
    try:
        parse_llm_json('{"items": [{"text": "broken "quote"}]}')
    except json.JSONDecodeError:
        return
    raise AssertionError("expected JSONDecodeError")
