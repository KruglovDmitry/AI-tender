"""Индексатор «прочее»: индексация не выполняется."""

from __future__ import annotations

from pathlib import Path

from ...models import (
    DOCUMENT_KIND_LABELS,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
)


class OtherDocumentIndexer:
    kind = DocumentKind.other

    def index(self, path: Path, *, relative_path: str, context: IndexingContext,) -> IndexingResult:
        del path, context
        label = DOCUMENT_KIND_LABELS[self.kind]
        name = Path(relative_path).name
        return IndexingResult(
            relative_path=relative_path,
            doc_kind=self.kind,
            status=IndexingStatus.skipped,
            message=(
                f"Тип документа «{label}»: файл «{name}» пропущен — "
                "индексация для этого типа не выполняется"
            ),
        )
