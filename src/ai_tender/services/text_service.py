"""Сборка и дедуп ExtractedRequirement."""

from __future__ import annotations

from ..models import ExtractedRequirement


def _strip_line_prefixes(quote: str) -> str:
    """Убрать префиксы `N|` из цитаты модели (если скопировала нумерацию)."""
    lines = []
    for line in (quote or "").splitlines() or [quote]:
        if "|" in line[:6]:
            prefix, _, rest = line.partition("|")
            if prefix.strip().isdigit():
                lines.append(rest)
                continue
        lines.append(line)
    return "\n".join(lines).strip()


def make_requirement(
    *,
    text: str,
    quote: str,
    file: str,
    page: int | None = None,
    priority: int = 2,
    confidence: float = 0.7,
    kind: str = "other",
) -> ExtractedRequirement:
    """Собрать требование без поиска якоря в исходном тексте."""
    cleaned_quote = _strip_line_prefixes(quote) or text
    cleaned_quote = " ".join(cleaned_quote.split())[:800]
    kind_norm = (kind or "other").strip().lower()
    if kind_norm not in {"product", "specs", "other"}:
        kind_norm = "other"
    location = f"стр. {page}" if page is not None else "документ"
    return ExtractedRequirement(
        text=text[:500],
        quote=cleaned_quote,
        file=file,
        location=location,
        page=page,
        line_start=None,
        line_end=None,
        kind=kind_norm,
        priority=priority,
        confidence=confidence,
    )


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe_requirements(
    items: list[ExtractedRequirement],
    limit: int,
) -> list[ExtractedRequirement]:
    """Сначала product, затем specs/other; внутри — по priority/confidence."""
    kind_rank = {"product": 0, "specs": 1, "other": 2}
    ranked = sorted(
        items,
        key=lambda item: (
            kind_rank.get(item.kind, 9),
            -item.priority,
            -item.confidence,
        ),
    )
    seen: set[str] = set()
    products: list[ExtractedRequirement] = []
    rest: list[ExtractedRequirement] = []
    for item in ranked:
        key = _normalize_key(item.text)
        if not key or key in seen:
            continue
        seen.add(key)
        if item.kind == "product":
            products.append(item)
        else:
            rest.append(item)

    output = list(products)
    room = max(0, limit - len(output))
    output.extend(rest[:room])
    return output[: max(len(products), limit)]
