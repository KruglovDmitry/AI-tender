"""Экспорт AnalysisReport в файлы для скачивания."""

from __future__ import annotations

import json
from pathlib import Path

from ai_tender.models import POSITION_STATUS_LABELS, AnalysisReport, PositionMatchStatus


def report_to_json_bytes(report: AnalysisReport) -> bytes:
    payload = report.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def report_to_markdown(report: AnalysisReport) -> str:
    lines: list[str] = ["# AI Tender — отчёт", ""]

    elapsed = report.elapsed_seconds
    if elapsed is not None:
        lines.append(f"- Время: {elapsed:.1f} с")
    lines.append(
        "- Индекс эталонов: "
        + ("из кэша" if report.index_reused else "построен заново")
    )
    lines.append(f"- LLM: `{report.llm_model}`")
    lines.append(f"- Embeddings: `{report.embedding_model}`")
    lines.append(f"- Тендер: `{report.tender_path}`")
    lines.append(f"- Эталоны: `{report.assets_path}`")
    lines.append("")

    if report.verdict:
        lines.extend(["## Итоговый вывод", "", report.verdict.strip(), ""])
    elif report.summary:
        lines.extend(["## Кратко", "", report.summary.strip(), ""])

    matches = list(report.position_matches or [])
    if matches:
        lines.extend(["## Позиции", ""])
        for index, match in enumerate(matches, start=1):
            status = POSITION_STATUS_LABELS.get(match.status.value, match.status.value)
            qty = ""
            if match.qty is not None:
                qty = f" — {match.qty} {match.unit}".rstrip()
            lines.append(f"### {index}. {match.scope_name}{qty}")
            lines.append(f"- Статус: **{status}** (conf={match.confidence:.2f})")
            if match.status == PositionMatchStatus.none or not match.product_name:
                lines.append("- Эталон: нет подходящего варианта")
            else:
                lines.append(f"- Эталон: {match.product_name}")
            if match.explanation:
                lines.append(f"- Пояснение: {match.explanation}")
            if match.requirements:
                lines.append("- Требования:")
                for req in match.requirements:
                    lines.append(f"  - {req.text}")
            if match.asset_hits:
                lines.append("- Фрагменты эталона:")
                for hit in match.asset_hits[:5]:
                    score = f", score={hit.score:.3f}" if hit.score is not None else ""
                    lines.append(
                        f"  - `{Path(hit.file).name}` · {hit.location}{score}: "
                        f"{hit.quote[:300]}"
                    )
            lines.append("")

    if report.indexed_files:
        lines.extend(["## Эталоны в индексе", ""])
        for path in report.indexed_files:
            lines.append(f"- `{Path(path).name}`")
        lines.append("")

    if report.warnings:
        lines.extend(["## Предупреждения", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
