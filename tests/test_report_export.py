from ai_tender.models import AnalysisReport, PositionMatchStatus, ScopePositionMatch
from ai_tender.services.report_export import report_to_json_bytes, report_to_markdown


def test_report_export_markdown_and_json() -> None:
    report = AnalysisReport(
        tender_path="/t",
        assets_path="/a",
        embedding_model="BAAI/bge-m3",
        llm_model="deepseek-chat",
        verdict="Подходит частично.",
        position_matches=[
            ScopePositionMatch(
                scope_name="Счётчик",
                qty=10,
                unit="шт",
                status=PositionMatchStatus.partial,
                product_name="Модель X",
                required_product="Модель X или аналог",
                explanation="Есть основное изделие.",
                confidence=0.7,
            )
        ],
        indexed_files=["/a/etalon.pdf"],
        warnings=["тест"],
        elapsed_seconds=12.5,
    )
    md = report_to_markdown(report)
    assert "Подходит частично" in md
    assert "Счётчик" in md
    assert "Требуется: Модель X или аналог" in md
    assert "Подобрано: Модель X" in md
    assert "Модель X" in md

    raw = report_to_json_bytes(report)
    assert b"deepseek-chat" in raw
    assert "Счётчик".encode("utf-8") in raw
