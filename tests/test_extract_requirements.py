import json
from typing import Any

from llama_index.core import Document

from ai_tender.nodes.requirements import (
    _dedupe_equivalent_files,
    _order_requirement_files,
    extract_requirements_per_scope_items,
)


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


def _docs(*pairs: tuple[str, str]) -> list[Document]:
    return [
        Document(text=text, metadata={"file_path": label, "file_name": label})
        for label, text in pairs
    ]


def test_order_requirement_files_uses_doc_selection_not_length() -> None:
    files = [
        ("title.pdf", "short", None),
        ("ТЗ ФЗ-522 детальное.docx", "x" * 50, None),
        ("Приложение №6 Регламент.docx", "y" * 5000, None),
        ("Извещение.docx", "z" * 2000, None),
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
    ordered = _order_requirement_files(
        files,
        file_order=[f[0] for f in files],
        prefer_labels=["title.pdf", "ТЗ ФЗ-522 детальное.docx"],
        doc_selection=doc_selection,
    )
    assert [item[0] for item in ordered] == [
        "ТЗ ФЗ-522 детальное.docx",
        "title.pdf",
        "Приложение №6 Регламент.docx",
        "Извещение.docx",
    ]


def test_dedupe_docx_preferred_over_pdf_same_stem() -> None:
    files = [
        ("ТЗ ФЗ-522_6-20кВ.pdf", "same body text " * 40, None),
        ("ТЗ ФЗ-522_6-20кВ.docx", "same body text " * 40, None),
    ]
    kept, skipped = _dedupe_equivalent_files(files)
    assert len(kept) == 1
    assert kept[0][0].endswith(".docx")
    assert skipped


def test_extract_one_call_per_position_parallel() -> None:
    llm = CountingLLM()
    scope_items = [{"name": f"позиция {i}", "qty": i + 1, "unit": "шт."} for i in range(4)]
    docs = _docs(("tz.docx", "текст ТЗ с требованиями " * 50))

    buckets, stats = extract_requirements_per_scope_items(
        docs,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        max_files=2,
        parallelism=4,
    )

    assert llm.calls == 4
    assert stats["llm_calls"] == 4
    assert stats["parallelism"] == 4
    assert all(len(bucket) == 1 for bucket in buckets)


def test_extract_tries_next_file_when_empty() -> None:
    llm = CountingLLM(empty_labels={"tz.docx"})
    scope_items = [{"name": "ПКУ", "qty": 1, "unit": "шт."}]
    docs = _docs(
        ("title.pdf", "титул с общими требованиями " * 20),
        ("tz.docx", "файл без нужных требований " * 40),
    )
    doc_selection = {
        "files": [
            {"path": "title.pdf", "scope_level": 1, "role": "tz_main", "priority": 2},
            {"path": "tz.docx", "scope_level": 2, "role": "specs", "priority": 1},
        ]
    }

    buckets, stats = extract_requirements_per_scope_items(
        docs,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        doc_selection=doc_selection,
        max_files=2,
    )

    assert llm.calls == 2
    assert stats["files_tried_by_item"][0] == ["tz.docx", "title.pdf"]
    assert stats["source_by_item"] == ["title.pdf"]
    assert len(buckets[0]) == 1


def test_extract_retries_when_sibling_found() -> None:
    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> Any:
            self.calls += 1
            # first position always ok; second empty unless retry
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
    buckets, stats = extract_requirements_per_scope_items(
        _docs(("tz.docx", "текст " * 40)),
        scope_items=[{"name": "а"}, {"name": "б"}],
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        max_files=1,
        retry_if_sibling_found=True,
    )
    assert stats["retries"] == 1
    assert llm.calls == 3  # a + b empty + b retry
    assert [len(b) for b in buckets] == [1, 1]


def test_extract_skips_duplicate_pdf_when_docx_present() -> None:
    llm = CountingLLM()
    scope_items = [{"name": "ПКУ"}]
    body = "одинаковый текст требований ПКУ " * 30
    docs = _docs(
        ("ТЗ ФЗ-522.pdf", body),
        ("ТЗ ФЗ-522.docx", body),
    )

    _buckets, stats = extract_requirements_per_scope_items(
        docs,
        scope_items=scope_items,
        llm=llm,  # type: ignore[arg-type]
        max_per_item=1,
        max_chars_per_doc=5000,
        max_files=3,
    )

    assert stats["files_after_dedupe"] == 1
    assert llm.calls == 1
    assert stats["source_by_item"][0].endswith(".docx")
