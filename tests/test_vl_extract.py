"""Unit tests for VL extract routing/helpers."""

from pathlib import Path

from ai_tender.extract.qwen_extract import QwenExtractor, _merge_catalog_results
from ai_tender.extract.qwen_gate import ExtractRoute, GateDecision
from ai_tender.extract.schemas import CatalogExtractResult, ProductRecord
from ai_tender.models import Settings


def test_should_use_vl_for_scan_even_when_disabled() -> None:
    ex = QwenExtractor(cache_dir=Path("data/cache"), vl_enabled=False)
    scan = GateDecision(True, ExtractRoute.qwen_scan, "no text")
    doc = GateDecision(True, ExtractRoute.qwen_doc, "ok")
    assert ex.should_use_vl(scan) is True
    assert ex.should_use_vl(doc) is False


def test_should_use_vl_when_enabled_for_text_pdf() -> None:
    ex = QwenExtractor(cache_dir=Path("data/cache"), vl_enabled=True)
    doc = GateDecision(True, ExtractRoute.qwen_doc, "ok")
    assert ex.should_use_vl(doc) is True


def test_should_skip_vl_for_large_text_pdf(tmp_path: Path) -> None:
    import fitz

    pdf = tmp_path / "big.pdf"
    doc = fitz.open()
    for _ in range(20):
        doc.new_page()
    doc.save(pdf)
    doc.close()

    ex = QwenExtractor(cache_dir=Path("data/cache"), vl_enabled=True)
    gate = GateDecision(True, ExtractRoute.qwen_doc, "ok")
    assert ex.should_use_vl(gate, pdf) is False


def test_merge_catalog_dedupes_by_model() -> None:
    merged = _merge_catalog_results(
        [
            CatalogExtractResult(
                catalog_name="Тракт",
                products=[ProductRecord(model="A-1", manufacturer="X")],
            ),
            CatalogExtractResult(
                catalog_name="",
                products=[
                    ProductRecord(model="A-1", manufacturer="X"),
                    ProductRecord(model="B-2", manufacturer="Y"),
                ],
            ),
        ]
    )
    assert merged.catalog_name == "Тракт"
    assert [p.model for p in merged.products] == ["A-1", "B-2"]


def test_settings_vl_fields() -> None:
    settings = Settings(vl_enabled=True, qwen_vl_model="qwen-vl-plus")
    assert settings.vl_enabled is True
    assert settings.qwen_vl_model == "qwen-vl-plus"
    assert not hasattr(settings, "ocr_enabled") or "ocr_enabled" not in settings.model_fields
