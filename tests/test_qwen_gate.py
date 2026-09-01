"""Unit-тесты pre-flight gate и кэша Qwen extract."""

from pathlib import Path

import pytest

from ai_tender.extract.qwen_cache import QwenExtractCache, content_sha256
from ai_tender.extract.qwen_gate import ExtractRoute, can_send_to_qwen


def test_xlsx_never_goes_to_qwen_doc(tmp_path: Path) -> None:
    path = tmp_path / "spec.xlsx"
    path.write_bytes(b"fake")
    decision = can_send_to_qwen(path)
    assert not decision.ok
    assert decision.route == ExtractRoute.legacy
    assert "таблица" in decision.reason.lower()


def test_docx_goes_to_qwen_doc(tmp_path: Path) -> None:
    path = tmp_path / "tz.docx"
    path.write_bytes(b"pk")
    decision = can_send_to_qwen(path)
    assert decision.ok
    assert decision.route == ExtractRoute.qwen_doc


def test_oversize_goes_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "big.pdf"
    path.write_bytes(b"x" * 100)
    from ai_tender.extract import qwen_gate

    monkeypatch.setattr(qwen_gate, "MAX_FILE_BYTES", 10)
    decision = can_send_to_qwen(path)
    assert not decision.ok
    assert decision.route == ExtractRoute.legacy


def test_izveshchenie_338_is_qwen_doc() -> None:
    path = Path("sources/1/Извещение 338.pdf")
    if not path.is_file():
        pytest.skip("smoke fixture PDF missing")
    decision = can_send_to_qwen(path)
    assert decision.ok
    assert decision.route == ExtractRoute.qwen_doc
    assert decision.sends_to_qwen_doc


def test_cache_roundtrip(tmp_path: Path) -> None:
    sample = tmp_path / "a.txt"
    sample.write_text("hello", encoding="utf-8")
    digest = content_sha256(sample)
    cache = QwenExtractCache(tmp_path / "cache")
    cache.put(
        kind="tender",
        content_hash=digest,
        file_id="file-abc",
        route="qwen_doc",
        model="qwen-doc-turbo",
        result={"scope_summary": "t", "scope_items": []},
        schema_version="1",
    )
    hit = cache.get(kind="tender", content_hash=digest, schema_version="1")
    assert hit is not None
    assert hit["file_id"] == "file-abc"
    assert hit["result"]["scope_summary"] == "t"
    assert cache.get(kind="tender", content_hash=digest, schema_version="2") is None
