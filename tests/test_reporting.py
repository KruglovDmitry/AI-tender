from pathlib import Path

from ai_tender.models import AnalysisReport, Comparison, Evidence, Requirement, Status
from ai_tender.reporting import render_html, save_report


def sample_report() -> AnalysisReport:
    requirement = Requirement(
        id="REQ-0001",
        text="Степень защиты не ниже IP54",
        category="защита",
        source_block_id="source-1",
        source_file="ТЗ.docx",
        source_location="таблица 1",
    )
    comparison = Comparison(
        requirement=requirement,
        status=Status.compliant,
        explanation="Эталон подтверждает IP54.",
        reference_value="IP54",
        confidence=0.95,
        evidence=[
            Evidence(
                file="РЭ.pdf",
                location="стр. 10",
                quote="Степень защиты IP54",
                block_id="asset-1",
            )
        ],
    )
    return AnalysisReport(
        tender_path="sources/1",
        assets_path="assets",
        model="test-model",
        comparisons=[comparison],
    )


def test_render_html_contains_requirement_and_evidence() -> None:
    result = render_html(sample_report())

    assert "Степень защиты не ниже IP54" in result
    assert "Соответствует" in result
    assert "Степень защиты IP54" in result


def test_save_report_creates_json_and_html(tmp_path: Path) -> None:
    json_path, html_path = save_report(sample_report(), tmp_path)

    assert json_path.exists()
    assert html_path.exists()
    assert '"status": "compliant"' in json_path.read_text(encoding="utf-8")
