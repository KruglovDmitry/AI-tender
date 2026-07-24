import json
from typing import Any

from ai_tender.models import ExtractedRequirement
from ai_tender.nodes.common import docs_for_label
from ai_tender.nodes.requirements import (
    extract_requirements_from_file,
    order_requirement_files,
)
from llama_index.core import Document


class CountingLLM:
    """LLM-заглушка: found/requirements; умеет отдавать пусто для части файлов."""

    def __init__(self, *, empty_labels: set[str] | None = None) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.empty_labels = empty_labels or set()

    def complete(self, prompt: str) -> Any:
        self.calls += 1
        self.prompts.append(prompt)
        if "ИСХОДНЫЙ ОТВЕТ" in prompt or "НЕВАЛИДНЫМ JSON" in prompt:
            return _Resp("<<<not-json>>>")

        label = ""
        for line in prompt.splitlines():
            if line.startswith("ФАЙЛ:"):
                label = line.split(":", 1)[1].strip()
                break

        if any(token in label for token in self.empty_labels):
            return _Resp(json.dumps({"found": False, "requirements": []}, ensure_ascii=False))

        return _Resp(
            json.dumps(
                {
                    "found": True,
                    "requirements": [
                        {
                            "text": f"требование из {label or 'doc'}",
                            "quote": f"требование из {label or 'doc'}",
                            "kind": "specs",
                            "priority": 2,
                            "confidence": 0.9,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


def test_order_requirement_files_specs_before_title() -> None:
    labels = [
        "title.pdf",
        "ТЗ ФЗ-522 детальное.docx",
        "Приложение №6 Регламент.docx",
        "Извещение.docx",
    ]
    doc_selection = {
        "files": [
            {"path": "title.pdf", "scope_level": 1, "role": "tz_main", "priority": 2},
            {
                "path": "ТЗ ФЗ-522 детальное.docx",
                "scope_level": 2,
                "role": "specs",
                "priority": 1,
            },
            {
                "path": "Приложение №6 Регламент.docx",
                "scope_level": 3,
                "role": "other",
                "priority": 3,
            },
            {"path": "Извещение.docx", "scope_level": 1, "role": "notice", "priority": 2},
        ]
    }
    ordered = order_requirement_files(labels, doc_selection=doc_selection)
    assert ordered == [
        "ТЗ ФЗ-522 детальное.docx",
        "title.pdf",
        "Приложение №6 Регламент.docx",
        "Извещение.docx",
    ]


def test_order_prefers_docx_over_pdf_same_stem() -> None:
    ordered = order_requirement_files(
        ["ТЗ ФЗ-522_6-20кВ.pdf", "ТЗ ФЗ-522_6-20кВ.docx"]
    )
    assert ordered == ["ТЗ ФЗ-522_6-20кВ.docx"]


def test_extract_one_call_per_position_parallel() -> None:
    llm = CountingLLM()
    scope_items = [{"name": f"позиция {i}", "qty": i + 1, "unit": "шт."} for i in range(4)]
    buckets, stats = extract_requirements_from_file(
        label="tz.docx",
        text="текст ТЗ с требованиями " * 50,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        parallelism=4,
    )
    assert llm.calls == 4
    assert stats["llm_calls"] == 4
    assert stats["parallelism"] == 4
    assert all(len(bucket) == 1 for bucket in buckets)


def test_extract_second_file_fills_empty_buckets() -> None:
    """Как два прохода графа: пустой файл → следующий."""
    llm = CountingLLM(empty_labels={"tz.docx"})
    scope_items = [{"name": "ПКУ", "qty": 1, "unit": "шт."}]

    buckets, stats1 = extract_requirements_from_file(
        label="tz.docx",
        text="файл без нужных требований " * 40,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
    )
    assert buckets == [[]]
    assert stats1["llm_calls"] == 1

    buckets, stats2 = extract_requirements_from_file(
        label="title.pdf",
        text="титул с общими требованиями " * 20,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        existing_buckets=buckets,
    )
    assert stats2["source_by_item"] == ["title.pdf"]
    assert len(buckets[0]) == 1
    assert llm.calls == 2


def test_extract_retries_when_sibling_found() -> None:
    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> Any:
            self.calls += 1
            if "ПОЗИЦИЯ: а" in prompt or "ПОЗИЦИЯ: а —" in prompt:
                return _Resp(
                    json.dumps(
                        {
                            "found": True,
                            "requirements": [
                                {
                                    "text": "req a",
                                    "quote": "req a",
                                    "kind": "specs",
                                    "priority": 2,
                                    "confidence": 0.9,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            if "ПОВТОРНЫЙ ЗАПРОС" in prompt:
                return _Resp(
                    json.dumps(
                        {
                            "found": True,
                            "requirements": [
                                {
                                    "text": "req b retry",
                                    "quote": "req b retry",
                                    "kind": "specs",
                                    "priority": 2,
                                    "confidence": 0.9,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            return _Resp(json.dumps({"found": False, "requirements": []}, ensure_ascii=False))

    llm = FlakyLLM()
    buckets, stats = extract_requirements_from_file(
        label="tz.docx",
        text="текст " * 40,
        scope_items=[{"name": "а"}, {"name": "б"}],
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        retry_if_sibling_found=True,
    )
    assert stats["retries"] == 1
    assert llm.calls == 3
    assert [len(b) for b in buckets] == [1, 1]


def test_extract_no_retry_when_only_previous_file_filled() -> None:
    """Retry только если сосед нашёлся в ТЕКУЩЕМ файле."""
    llm = CountingLLM(empty_labels={"title.pdf"})
    scope_items = [{"name": "а"}, {"name": "б"}]
    existing = [
        [
            ExtractedRequirement(
                text="from prev",
                quote="from prev",
                file="tz.docx",
                location="документ",
            )
        ],
        [],
    ]
    buckets, stats = extract_requirements_from_file(
        label="title.pdf",
        text="титул " * 20,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        existing_buckets=existing,
        retry_if_sibling_found=True,
    )
    assert stats["retries"] == 0
    assert llm.calls == 1
    assert len(buckets[0]) == 1
    assert buckets[1] == []


def test_docs_for_label_prefers_exact_path() -> None:
    docs = [
        Document(text="wrong", metadata={"file_path": "other/tz.docx"}),
        Document(text="right", metadata={"file_path": "folder/tz.docx"}),
    ]
    matched = docs_for_label(docs, "folder/tz.docx")
    assert len(matched) == 1
    assert matched[0].text == "right"
