from pathlib import Path

import fitz

from ai_tender.services.index_service import (
    delete_asset_file,
    file_fingerprint,
    scan_assets_files,
)
from ai_tender.services.catalog_service import (
    delete_product_artifacts,
    json_path_for,
    load_product_index,
    save_product_index,
)
from ai_tender.models import DocumentKind, ProductDocumentIndex


def _write_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(page.rect + (36, 36, -36, -36), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_scan_assets_files(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    _write_pdf(assets / "a.pdf", "model A")
    fps = scan_assets_files(assets)
    assert "a.pdf" in fps
    assert fps["a.pdf"]["size"] > 0


def test_file_fingerprint_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.pdf"
    _write_pdf(path, "hello")
    fp1 = file_fingerprint(path, relative="x.pdf")
    fp2 = file_fingerprint(path, relative="x.pdf")
    assert fp1 == fp2


def test_delete_asset_file_removes_disk_and_cache(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(assets / "cat.pdf", "UPS-1000")
    save_product_index(
        cache,
        ProductDocumentIndex(
            source_file="cat.pdf",
            doc_kind=DocumentKind.asset,
            products=[],
        ),
    )
    assert json_path_for(cache, "cat.pdf").is_file()

    warnings = delete_asset_file(assets, cache, "cat.pdf")
    assert not (assets / "cat.pdf").is_file()
    assert load_product_index(cache, "cat.pdf") is None
    assert not json_path_for(cache, "cat.pdf").exists()
    assert warnings == []

    delete_product_artifacts(cache, "missing.pdf")
