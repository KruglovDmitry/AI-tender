"""Пересчёт эмбеддингов VL-каталога из product_json (без VL-парсинга)."""

from __future__ import annotations

import sys

from ai_tender.models import get_settings
from ai_tender.services.catalog_retrieval import reembed_vl_catalogs


def main() -> int:
    settings = get_settings()
    messages = reembed_vl_catalogs(
        settings.cache_dir,
        embedding_model=settings.embedding_model,
        device=settings.embedding_device,
    )
    for line in messages:
        print(line)
    return 0 if messages and not messages[0].startswith("VL-каталог не найден") else 1


if __name__ == "__main__":
    sys.exit(main())
