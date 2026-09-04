"""Unit-тесты pre-flight gate Qwen extract."""

from pathlib import Path

import pytest

from ai_tender.extract import base_extract
from ai_tender.extract.base_extract import ExtractRoute, can_send_to_qwen


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
    monkeypatch.setattr(base_extract, "MAX_FILE_BYTES", 10)
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
