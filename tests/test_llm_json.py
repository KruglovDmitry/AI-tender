import json

from ai_tender.providers import complete_llm_json, parse_llm_json, try_parse_llm_json


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


def test_parse_llm_json_missing_comma_between_objects() -> None:
    data = parse_llm_json('{"items": [{"x": 1}\n{"x": 2}]}')
    assert data["items"][0]["x"] == 1
    assert data["items"][1]["x"] == 2


def test_try_parse_llm_json_returns_none_on_garbage() -> None:
    assert try_parse_llm_json("not json at all {") is None


def test_salvage_match_json_partial() -> None:
    raw = (
        '{"matched": true, "status": "matched", "required_product": "ABC-100", '
        '"product_name": "Модель X-200", '
        '"explanation": "Аналог указан в каталоге", "confidence": 0.85'
    )
    data = try_parse_llm_json(raw)
    assert data is not None
    assert data["matched"] is True
    assert data["status"] == "matched"
    assert data["product_name"] == "Модель X-200"


def test_parse_llm_json_still_raises_on_unfixable() -> None:
    try:
        parse_llm_json('{"items": [{"text": "broken "quote"}]}')
    except json.JSONDecodeError:
        return
    raise AssertionError("expected JSONDecodeError")


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


def test_complete_llm_json_ok_first_try() -> None:
    class LLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> _Resp:
            self.calls += 1
            return _Resp('{"found": true, "requirements": []}')

    llm = LLM()
    data, n = complete_llm_json(llm, "prompt", trace_name=None)  # type: ignore[arg-type]
    assert data == {"found": True, "requirements": []}
    assert n == 1
    assert llm.calls == 1


def test_complete_llm_json_repairs_invalid() -> None:
    class LLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> _Resp:
            self.calls += 1
            if self.calls == 1:
                return _Resp("<<<not-json>>>")
            return _Resp('{"found": false, "requirements": []}')

    llm = LLM()
    data, n = complete_llm_json(
        llm,  # type: ignore[arg-type]
        "prompt",
        structure_hint="той же структуры (поля found и requirements)",
        trace_name=None,
    )
    assert data == {"found": False, "requirements": []}
    assert n == 2
    assert llm.calls == 2
