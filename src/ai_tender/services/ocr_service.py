"""OCR для PDF-сканов без текстового слоя."""

from __future__ import annotations

import shutil
from pathlib import Path

from llama_index.core import Document

MIN_TEXT_LEN = 20

_REPO_TESSDATA = Path(__file__).resolve().parents[3] / "data" / "tessdata"


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


def _system_tessdata_candidates() -> list[Path]:
    """Стандартные пути tessdata: Windows installer + Debian/Ubuntu packages."""
    return [
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tessdata"),
    ]


def tessdata_dir() -> Path | None:
    import os

    prefix = os.environ.get("TESSDATA_PREFIX")
    candidates: list[Path | None] = [
        Path(prefix) if prefix else None,
        _REPO_TESSDATA,
        *_system_tessdata_candidates(),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        # TESSDATA_PREFIX иногда указывает на родителя tessdata/
        for folder in (candidate, candidate / "tessdata"):
            if folder.is_dir() and any(folder.glob("*.traineddata")):
                return folder
    return None


def _tesseract_listed_langs() -> set[str] | None:
    """Языки из `tesseract --list-langs` (если бинарь доступен)."""
    tess = tesseract_path()
    if tess is None:
        return None
    try:
        import subprocess

        result = subprocess.run(
            [str(tess), "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        langs = {
            line.strip()
            for line in output.splitlines()
            if line.strip() and not line.lower().startswith("list of")
        }
        return langs or None
    except Exception:
        return None


def _languages_available(languages: str) -> list[str]:
    wanted = [lang.strip() for lang in languages.replace("+", " ").split() if lang.strip()]
    folder = tessdata_dir()
    if folder is not None:
        return [
            lang
            for lang in wanted
            if not (folder / f"{lang}.traineddata").is_file()
        ]

    listed = _tesseract_listed_langs()
    if listed is not None:
        return [lang for lang in wanted if lang not in listed]

    # Не смогли найти tessdata и опросить tesseract — считаем все отсутствующими.
    return wanted


def _ocr_config() -> str:
    """Всегда передаём --tessdata-dir, если папка найдена.

    Иначе pytesseract опирается на TESSDATA_PREFIX, который в Docker иногда
    указывает на родителя (.../5/), а не на .../5/tessdata/.
    """
    folder = tessdata_dir()
    if folder is None:
        return ""
    return f'--tessdata-dir "{folder}"'


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

    try:
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
    except Exception as exc:
        return [], f"OCR ошибка для {label}: {exc}"

    if not documents:
        return [], "OCR не извлёк текст"

    note = None
    if ocr_pages:
        note = f"Распознано OCR: {label} ({ocr_pages} стр.)"
    return documents, note
