"""Рендер страниц PDF / одиночных изображений для VL-extract."""

from __future__ import annotations

import base64
import io
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"})


def render_document_images(
    path: Path,
    *,
    dpi_scale: float = 1.5,
    max_pages: int = 80,
    jpeg_quality: int = 85,
) -> list[tuple[int, str]]:
    """Возвращает список (page_number_1based, data_url) для multimodal VL."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        raw = path.read_bytes()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        b64 = base64.b64encode(raw).decode("ascii")
        return [(1, f"data:{mime};base64,{b64}")]

    if ext != ".pdf":
        raise ValueError(f"VL-рендер не поддержан для {ext}")

    import fitz
    from PIL import Image

    out: list[tuple[int, str]] = []
    matrix = fitz.Matrix(dpi_scale, dpi_scale)
    with fitz.open(path) as doc:
        total = min(doc.page_count, max_pages)
        for i in range(total):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            out.append((i + 1, f"data:image/jpeg;base64,{b64}"))
    if not out:
        raise ValueError(f"PDF без страниц: {path.name}")
    return out
