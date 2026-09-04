from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core.llms import LLM

from ..providers import complete_llm_json
from ..services.loader_service import READABLE_SUFFIXES, TenderInventory, inventory_tender_folder
from ..models import PipelineState, Settings

SELECT_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект:
{
  "files": [
    {
      "path": "относительный/путь/к/файлу.pdf",
      "priority": 1,
      "scope_level": 1,
      "role": "tz_main|specs|notice|contract|nmck|other",
      "reason": "кратко, зачем этот файл"
    }
  ],
  "skip": [
    {"path": "...", "reason": "почему не нужен для предмета закупки"}
  ]
}

Правила:
- Нужны файлы для ПРЕДМЕТА ЗАКУПКИ: общий титул И детальный перечень позиций.
- priority: 1 = главный источник, 2 = дополнительный, 3 = запасной.
- scope_level: 1 = общее описание/титул закупки («ТЗ на проведение…», извещение),
  2 = детальное ТЗ с перечнем работ/оборудования и количествами (часто «ТЗ ФЗ-…»,
  файлы в папке «Техническое задание»), 3 = приложения/второстепенное.
- Включай ОБА слоя (scope_level 1 и 2), если они есть. Детальное ТЗ — priority 1.
- Не включай: проект договора, обеспечение заявки, реквизиты, НМЦ/расчёты цены,
  регламенты ЭДО, протоколы, шаблоны форм.
- Не выдумывай пути — только из списка ниже.
""".strip()


@dataclass(frozen=True)
class TenderFileEntry:
    path: str
    suffix: str
    size_bytes: int
    parent: str


def build_catalog_from_inventory(inventory: TenderInventory) -> list[TenderFileEntry]:
    entries: list[TenderFileEntry] = []
    for path, label in inventory.work_items:
        if path.suffix.lower() not in READABLE_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        parent = Path(label).parent.as_posix()
        if parent == ".":
            parent = ""
        entries.append(
            TenderFileEntry(
                path=label,
                suffix=path.suffix.lower(),
                size_bytes=size,
                parent=parent,
            )
        )
    return sorted(entries, key=lambda item: item.path.lower())


def format_catalog_tree(entries: list[TenderFileEntry]) -> str:
    """Плоский список файлов с папкой и размером (для промпта LLM)."""
    if not entries:
        return "(пусто)"
    lines: list[str] = []
    for entry in entries:
        kb = max(1, entry.size_bytes // 1024)
        lines.append(f"- {entry.path}  [{entry.suffix}, {kb} КБ]")
    return "\n".join(lines)


def select_tender_files_by_llm(
    entries: list[TenderFileEntry],
    catalog_text: str,
    llm: LLM,
    *,
    max_files: int,
) -> dict[str, Any]:
    valid_paths = {entry.path for entry in entries}
    prompt = (
        "Ты аналитик закупок. По списку тендерных файлов выбери документы "
        "для извлечения ПРЕДМЕТА ЗАКУПКИ: общий титул и детальный перечень позиций "
        f"(работы/оборудование с количествами). Верни не более {max_files} файлов "
        "в files, отсортированных по priority (1 = сначала).\n\n"
        f"{SELECT_SCHEMA_HINT}\n\n"
        f"СПИСОК ФАЙЛОВ:\n{catalog_text}"
    )
    data, _n_calls = complete_llm_json(
        llm,
        prompt,
        structure_hint="той же структуры (files, skip)",
        trace_name="select_files",
    )
    if data is None:
        raise ValueError("LLM вернул невалидный JSON при выборе файлов")

    files_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data.get("files", []):
        path = str(item.get("path", "")).strip().replace("\\", "/")
        if not path or path not in valid_paths or path in seen:
            continue
        seen.add(path)
        try:
            priority = int(item.get("priority", 2))
        except (TypeError, ValueError):
            priority = 2
        try:
            scope_level = int(item.get("scope_level", 2))
        except (TypeError, ValueError):
            scope_level = 2
        files_out.append(
            {
                "path": path,
                "priority": min(max(priority, 1), 3),
                "scope_level": min(max(scope_level, 1), 3),
                "role": str(item.get("role", "other")).strip().lower() or "other",
                "reason": str(item.get("reason", "")).strip()[:200],
            }
        )
        if len(files_out) >= max_files:
            break

    files_out.sort(
        key=lambda item: (
            item.get("scope_level", 2),
            item.get("priority", 9),
            item["path"].lower(),
        )
    )

    skip_out: list[dict[str, str]] = []
    for item in data.get("skip", []):
        path = str(item.get("path", "")).strip().replace("\\", "/")
        if path and path in valid_paths and path not in seen:
            skip_out.append(
                {
                    "path": path,
                    "reason": str(item.get("reason", "")).strip()[:200],
                }
            )

    return {
        "mode": "llm",
        "files": files_out,
        "skip": skip_out,
        "catalog_paths": sorted(valid_paths),
    }


def select_tender_files(
    folder: Path,
    llm: LLM,
    *,
    max_files: int = 6,
) -> tuple[TenderInventory, list[TenderFileEntry], dict[str, Any]]:
    """Инвентаризация папки + LLM-выбор файлов для extract."""
    inventory = inventory_tender_folder(folder)
    entries = build_catalog_from_inventory(inventory)
    catalog_text = format_catalog_tree(entries)

    if not entries:
        return inventory, entries, {
            "mode": "empty",
            "files": [],
            "skip": [],
            "catalog_paths": [],
            "tree": catalog_text,
            "warnings": list(inventory.warnings),
            "catalog_count": 0,
        }

    selection = select_tender_files_by_llm(
        entries,
        catalog_text,
        llm,
        max_files=max_files,
    )
    if not selection.get("files"):
        raise ValueError(
            "LLM не выбрал ни одного файла тендера для анализа. "
            "Проверьте состав документации."
        )

    selection["tree"] = catalog_text
    selection["warnings"] = list(inventory.warnings)
    selection["catalog_count"] = len(entries)
    return inventory, entries, selection


def ranked_file_paths(selection: dict[str, Any]) -> list[str]:
    files = selection.get("files") or []
    ranked = sorted(
        files,
        key=lambda item: (
            item.get("scope_level", 2),
            item.get("priority", 9),
            item.get("path", "").lower(),
        ),
    )
    return [str(item["path"]) for item in ranked if item.get("path")]


def node_select_files(state: PipelineState) -> dict[str, Any]:
    """Каталог тендера → LLM-выбор файлов."""
    settings: Settings = state["settings"]
    callback = state.get("progress")
    if callable(callback):
        callback("Каталог и выбор файлов тендера", 0.1)
    inventory, catalog_entries, doc_selection = select_tender_files(
        Path(state["tender_path"]),
        state["llm"],
        max_files=settings.max_tender_files_total,
    )
    box = state.get("cleanup_box")
    if isinstance(box, dict):
        box["inventory"] = inventory

    ranked = ranked_file_paths(doc_selection)
    if not ranked:
        raise ValueError("Не выбрано ни одного файла тендера для анализа")

    initial_queue = ranked[: max(1, settings.max_tender_files_initial)]
    if callable(callback):
        callback(
            (
                f"Выбрано {len(ranked)} из {doc_selection.get('catalog_count', 0)} "
                f"файлов (llm)"
            ),
            0.2,
        )
    return {
        "inventory": inventory,
        "catalog_entries": catalog_entries,
        "doc_selection": doc_selection,
        "ranked_paths": ranked,
        "scope_queue": initial_queue,
        "loaded_labels": [],
        "documents": [],
        "scope_files_used": [],
        "scope_items": [],
        "scope_meta": {},
        "warnings": list(doc_selection.get("warnings") or []),
        "requirements_by_item": [],
        "requirements_stats": {},
        "qwen_extracted_files": [],
        "position_matches": [],
        "verdict": "",
        "query_selection": {},
        "indexed_files": [],
        "index_reused": False,
    }
