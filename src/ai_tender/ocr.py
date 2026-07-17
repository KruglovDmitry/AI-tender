"""OCR для PDF-сканов без текстового слоя."""

from __future__ import annotations

import shutil
from pathlib import Path

from llama_index.core import Document

MIN_TEXT_LEN = 20

_REPO_TESSDATA = Path(__file__).resolve().parents[2] / "data" / "tessdata"


def tesseract_path() -> Path | None:
    import os

    env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
    if env and Path(env).is_file():
        return Path(env)

    found = shutil.which("tesseract")
    if found:
        return Path(found)

    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def tessdata_dir() -> Path | None:
    import os

    for candidate in (
        Path(os.environ["TESSDATA_PREFIX"]) if os.environ.get("TESSDATA_PREFIX") else None,
        _REPO_TESSDATA,
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    ):
        if candidate and candidate.is_dir() and any(candidate.glob("*.traineddata")):
            return candidate
    return None


def _languages_available(languages: str) -> list[str]:
    folder = tessdata_dir()
    if folder is None:
        return list(languages.replace("+", " ").split())
    missing: list[str] = []
    for lang in languages.replace("+", " ").split():
        if not (folder / f"{lang.strip()}.traineddata").is_file():
            missing.append(lang.strip())
    return missing


def _ocr_config() -> str:
    """--tessdata-dir только для нестандартной папки (например data/tessdata в проекте)."""
    folder = tessdata_dir()
    if folder is None:
        return ""
    install_dirs = {
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    }
    if folder.resolve() in {path.resolve() for path in install_dirs}:
        return ""
    return f"--tessdata-dir {folder}"


def ocr_status(languages: str = "rus+eng") -> tuple[bool, str]:
    try:
        import fitz  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return False, "Установите пакеты: pip install pymupdf pytesseract pillow"

    tess = tesseract_path()
    if tess is None:
        return (
            False,
            "Tesseract не найден. Установите https://github.com/UB-Mannheim/tesseract/wiki "
            "(языки rus+eng) или задайте TESSERACT_CMD.",
        )

    missing = _languages_available(languages)
    if missing:
        data = tessdata_dir()
        hint = (
            f"Нет языковых файлов: {', '.join(missing)}. "
            "Переустановите Tesseract с Russian+English или положите "
            "*.traineddata в data/tessdata проекта."
        )
        if data:
            hint += f" (папка tessdata: {data})"
        return False, hint

    data = tessdata_dir()
    suffix = f", tessdata: {data}" if data else ""
    return True, f"{tess}{suffix}"


def extract_pdf_with_ocr(
    path: Path,
    label: str,
    corpus: str,
    languages: str = "rus+eng",
) -> tuple[list[Document], str | None]:
    """Извлекает текст постранично; для пустых страниц — OCR."""
    ok, detail = ocr_status(languages)
    if not ok:
        return [], detail

    import fitz
    import pytesseract
    from PIL import Image

    tess = tesseract_path()
    if tess:
        pytesseract.pytesseract.tesseract_cmd = str(tess)

    ocr_config = _ocr_config()
    documents: list[Document] = []
    ocr_pages = 0

    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = (page.get_text() or "").strip()
            used_ocr = False

            if len(text) < MIN_TEXT_LEN:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = (
                    pytesseract.image_to_string(
                        image,
                        lang=languages,
                        config=ocr_config,
                    )
                    or ""
                ).strip()
                used_ocr = bool(text)
                if used_ocr:
                    ocr_pages += 1

            if len(text) < MIN_TEXT_LEN:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": Path(label).name,
                        "file_path": label,
                        "corpus": corpus,
                        "location": f"стр. {page_index}",
                        "page_number": page_index,
                        "ocr": used_ocr,
                    },
                )
            )

    if not documents:
        return [], "OCR не извлёк текст"

    note = None
    if ocr_pages:
        note = f"Распознано OCR: {label} ({ocr_pages} стр.)"
    return documents, note
