from pathlib import Path

from ai_tender.llm_trace import clear_trace, get_trace, start_trace, trace_llm, trace_retrieval


def test_llm_trace_writes_files(tmp_path: Path) -> None:
    clear_trace()
    trace = start_trace(tmp_path, meta={"test": True})
    assert get_trace() is trace

    trace_llm("requirements", prompt="PROMPT", response='{"items":[]}', meta={"phase": "extract"})
    trace_retrieval(
        "retrieve_position",
        query="Позиция: ПКУ",
        hits=[{"score": 0.9, "file": "a.pdf", "text_preview": "текст"}],
        meta={"scope_name": "ПКУ"},
    )
    trace.finish({"ok": True})
    clear_trace()
    assert get_trace() is None

    assert (trace.path / "meta.json").is_file()
    assert (trace.path / "events.jsonl").is_file()
    events = (trace.path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 2
    req_files = list(trace.path.glob("*_requirements_request.txt"))
    assert len(req_files) == 1
    assert req_files[0].read_text(encoding="utf-8") == "PROMPT"
    ret_files = list(trace.path.glob("*_retrieve_position.json"))
    assert len(ret_files) == 1
