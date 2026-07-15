import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .models import AnalysisReport

STATUS_LABELS = {
    "compliant": "Соответствует",
    "non_compliant": "Не соответствует",
    "partial": "Частично",
    "insufficient_evidence": "Недостаточно данных",
    "not_applicable": "Не применимо",
}


def save_report(report: AnalysisReport, output_root: Path) -> tuple[Path, Path]:
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report.report_dir = run_dir
    json_path = run_dir / "report.json"
    html_path = run_dir / "report.html"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path


def render_html(report: AnalysisReport) -> str:
    counts = Counter(item.status.value for item in report.comparisons)
    cards = "".join(
        f"<div class='card'><b>{html.escape(STATUS_LABELS[key])}</b><span>{counts[key]}</span></div>"
        for key in STATUS_LABELS
    )
    rows = []
    for item in report.comparisons:
        evidence = "<br>".join(
            f"<b>{html.escape(Path(e.file).name)} — {html.escape(e.location)}</b>: "
            f"«{html.escape(e.quote)}»"
            for e in item.evidence
        ) or "Подтверждающие фрагменты не найдены"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.requirement.id)}</td>"
            f"<td>{html.escape(item.requirement.text)}</td>"
            f"<td><span class='status {item.status.value}'>{STATUS_LABELS[item.status.value]}</span></td>"
            f"<td>{html.escape(item.explanation)}</td>"
            f"<td>{evidence}</td>"
            f"<td>{item.confidence:.0%}</td>"
            "</tr>"
        )
    warnings = "".join(f"<li>{html.escape(value)}</li>" for value in report.warnings)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт AI Tender</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;color:#202124}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{padding:14px 18px;border:1px solid #ddd;border-radius:10px;min-width:150px}}
.card span{{display:block;font-size:28px;margin-top:5px}} table{{border-collapse:collapse;width:100%;margin-top:24px}}
th,td{{border:1px solid #ddd;padding:10px;vertical-align:top;text-align:left}} th{{background:#f4f6f8}}
.status{{font-weight:bold}} .non_compliant{{color:#b42318}} .compliant{{color:#067647}}
.partial{{color:#b54708}} .insufficient_evidence{{color:#475467}} small{{color:#667085}}
</style></head><body>
<h1>Сопоставление технических требований</h1>
<p><small>Тендер: {html.escape(report.tender_path)}<br>Эталоны: {html.escape(report.assets_path)}
<br>Модель: {html.escape(report.model)}</small></p>
<div class="cards">{cards}</div>
<table><thead><tr><th>ID</th><th>Требование</th><th>Статус</th><th>Вывод</th>
<th>Доказательства</th><th>Уверенность</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Предупреждения извлечения</h2><ul>{warnings or '<li>Нет</li>'}</ul>
</body></html>"""
