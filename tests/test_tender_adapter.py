from ai_tender.extract.schemas import RequirementRecord, ScopeItemExtract, TenderExtractResult
from ai_tender.extract.tender_adapter import (
    merge_scope_item_lists,
    tender_result_to_requirements,
    tender_result_to_scope,
)


def test_tender_result_to_scope_and_requirements() -> None:
    result = TenderExtractResult(
        scope_summary="Закупка счётчиков",
        scope_items=[
            ScopeItemExtract(
                name="Счетчик МИР С-05.10",
                qty=1,
                unit="шт.",
                quote="Счетчик … 1 шт.",
                requirements=[
                    RequirementRecord(
                        text="Класс точности 1",
                        quote="класс точности 1/1",
                        kind="specs",
                    )
                ],
            )
        ],
        overall_confidence=0.9,
        needs_more_docs=False,
    )
    items, meta = tender_result_to_scope(result, source_file="tz.pdf")
    assert len(items) == 1
    assert items[0]["qty"] == 1
    assert meta["needs_more_docs"] is False
    assert meta["extraction_mode"] == "qwen_whole_file"

    buckets = tender_result_to_requirements(
        result,
        source_file="tz.pdf",
        scope_items=items,
        max_per_item=10,
    )
    assert len(buckets) == 1
    assert len(buckets[0]) == 1
    assert buckets[0][0].scope_item == "Счетчик МИР С-05.10"


def test_merge_scope_items_by_name() -> None:
    merged = merge_scope_item_lists(
        [{"name": "А", "qty": None}],
        [{"name": "а", "qty": 2, "unit": "шт."}],
    )
    assert len(merged) == 1
    assert merged[0]["qty"] == 2
