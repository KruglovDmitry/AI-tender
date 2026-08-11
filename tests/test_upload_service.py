import zipfile
from pathlib import Path

from ai_tender.services.upload_service import (
    cleanup_old_uploads,
    prepare_upload_dir,
    replace_shared_assets,
    safe_filename,
)


class _FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_safe_filename_strips_path() -> None:
    assert safe_filename(r"..\evil\doc.pdf") == "doc.pdf"


def test_prepare_upload_dir_saves_and_unpacks_zip(tmp_path: Path) -> None:
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("docs/a.txt", "hello")

    dest = tmp_path / "tender"
    path, warnings = prepare_upload_dir(
        [_FakeUpload("pack.zip", archive.read_bytes())],
        dest,
    )
    assert path == dest.resolve()
    assert not warnings
    assert (dest / "pack" / "docs" / "a.txt").is_file()
    assert not (dest / "pack.zip").exists()


def test_replace_shared_assets_clears_previous(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "old.txt").write_text("old", encoding="utf-8")

    replace_shared_assets([_FakeUpload("new.txt", b"new")], assets)
    assert not (assets / "old.txt").exists()
    assert (assets / "new.txt").read_bytes() == b"new"


def test_cleanup_old_uploads(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "uploads"
    old = root / "old_run"
    new = root / "new_run"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    monkeypatch.setattr(
        "ai_tender.services.upload_service.time.time",
        lambda: 1_000_000.0,
    )
    # old: mtime far in past
    import os

    os.utime(old, (0, 0))
    os.utime(new, (999_999, 999_999))
    removed = cleanup_old_uploads(root, max_age_hours=1)
    assert removed == 1
    assert not old.exists()
    assert new.exists()
