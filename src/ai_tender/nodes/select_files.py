"""Выбор тендерных файлов для extract: каталог → LLM/heuristic → приоритет."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core.llms import LLM

from ..providers import parse_llm_json
from ..loaders import READABLE_SUFFIXES, TenderInventory, inventory_tender_folder
from ..models import Settings
from ..state import PipelineState

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
    """Дерево папок + список файлов с размером."""
    if not entries:
        return "(пусто)"

    by_parent: dict[str, list[TenderFileEntry]] = {}
    for entry in entries:
        by_parent.setdefault(entry.parent, []).append(entry)

    lines: list[str] = []

    def walk(parent: str, indent: int) -> None:
        prefix = "  " * indent
        children = sorted(by_parent.get(parent, []), key=lambda item: item.path.lower())
        for entry in children:
            name = Path(entry.path).name
            kb = max(1, entry.size_bytes // 1024)
            lines.append(f"{prefix}{name}  ({entry.suffix}, {kb} КБ)")
        subdirs = sorted(
            {
                key
                for key in by_parent
                if key and (parent == "" and "/" not in key or key.startswith(parent + "/"))
            }
        )
        seen_dirs: set[str] = set()
        for key in by_parent:
            if parent == "":
                if "/" in key:
                    top = key.split("/")[0]
                else:
                    continue
            elif key.startswith(parent + "/"):
                rest = key[len(parent) + 1 :]
                top = rest.split("/")[0] if "/" in rest else rest
            else:
                continue
            dir_path = f"{parent}/{top}".strip("/") if parent else top
            if dir_path in seen_dirs:
                continue
            seen_dirs.add(dir_path)
            lines.append(f"{prefix}{top}/")
            walk(dir_path, indent + 1)

    roots = [e for e in entries if not e.parent]
    for entry in sorted(roots, key=lambda item: item.path.lower()):
        kb = max(1, entry.size_bytes // 1024)
        lines.append(f"{Path(entry.path).name}  ({entry.suffix}, {kb} КБ)")

    subdirs = sorted({e.parent.split("/")[0] for e in entries if e.parent})
    for subdir in subdirs:
        if any(not e.parent for e in entries if Path(e.path).name == subdir):
            continue
        lines.append(f"{subdir}/")
        walk(subdir, 1)

    if not lines:
        for entry in entries:
            kb = max(1, entry.size_bytes // 1024)
            lines.append(f"{entry.path}  ({entry.suffix}, {kb} КБ)")
    return "\n".join(lines)


def format_catalog_list(entries: list[TenderFileEntry]) -> str:
    lines = []
    for entry in entries:
        kb = max(1, entry.size_bytes // 1024)
        parent = f"{entry.parent}/" if entry.parent else ""
        lines.append(f"- {parent}{Path(entry.path).name}  [{entry.suffix}, {kb} КБ]")
    return "\n".join(lines)


def select_files_heuristic(
    entries: list[TenderFileEntry],
    *,
    max_files: int,
) -> dict[str, Any]:
    """Fallback: ранжирование по маркерам в пути (титул + детальное ТЗ)."""
    if not entries:
        return {"files": [], "skip": [], "mode": "heuristic_empty"}

    paths = {entry.path for entry in entries}
    scored: list[tuple[int, int, str, str]] = []
    for entry in entries:
        lower = entry.path.lower().replace("\\", "/")
        name = Path(entry.path).name.lower()
        score = 10
        role = "other"
        scope_level = 3

        in_tz_folder = "техническ" in lower
        is_detailed_tz = (
            ("тз" in name or name.startswith("тз"))
            and (
                "фз" in name
                or "522" in name
                or in_tz_folder
            )
        )
        is_title_tz = "провед" in lower and "закуп" in lower
        is_object_desc = any(
            x in lower
            for x in ("предмет", "объект закуп", "объект закупки", "описание объекта")
        )

        if is_detailed_tz:
            # Детальный перечень позиций — тот же приоритет, что и титул.
            score = 1
            role = "specs"
            scope_level = 2
        elif is_title_tz or is_object_desc:
            score = 1
            role = "tz_main"
            scope_level = 1
        elif "техническ" in lower or "тз" in name or "/тз " in lower or " тз " in lower:
            score = 2
            role = "specs"
            scope_level = 2
        elif "тз" in lower:
            score = 2
            role = "specs"
            scope_level = 2
        elif "извещение" in lower:
            score = 4
            role = "notice"
            scope_level = 1
        elif any(x in lower for x in ("нмц", "расценк", "реквизит", "договор", "обеспечен")):
            score = 8
            role = "other"
        scored.append((score, scope_level, entry.path, role))

    scored.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    preferred = [item for item in scored if item[0] <= 4]
    chosen = preferred if preferred else scored
    selected = chosen[:max_files]
    selected_paths = {path for _, _, path, _ in selected}

    return {
        "mode": "heuristic",
        "files": [
            {
                "path": path,
                "priority": min(score, 3),
                "scope_level": scope_level,
                "role": role,
                "reason": "эвристика по имени/пути",
            }
            for score, scope_level, path, role in selected
        ],
        "skip": [
            {"path": entry.path, "reason": "не прошёл эвристический отбор"}
            for entry in entries
            if entry.path not in selected_paths
        ],
        "catalog_paths": sorted(paths),
    }


def select_tender_files_by_llm(
    entries: list[TenderFileEntry],
    tree_text: str,
    llm: LLM,
    *,
    max_files: int,
) -> dict[str, Any]:
    valid_paths = {entry.path for entry in entries}
    catalog_list = format_catalog_list(entries)
    prompt = (
        "Ты аналитик закупок. По структуре тендерной документации выбери файлы "
        "для извлечения ПРЕДМЕТА ЗАКУПКИ: общий титул и детальный перечень позиций "
        f"(работы/оборудование с количествами). Верни не более {max_files} файлов "
        "в files, отсортированных по priority (1 = сначала).\n\n"
        f"{SELECT_SCHEMA_HINT}\n\n"
        f"СТРУКТУРА ПАПОК:\n{tree_text}\n\n"
        f"СПИСОК ФАЙЛОВ:\n{catalog_list}"
    )
    response = llm.complete(prompt)
    data = parse_llm_json(str(response))

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
        scope_level = min(max(scope_level, 1), 3)
        role = str(item.get("role", "other")).strip().lower() or "other"
        files_out.append(
            {
                "path": path,
                "priority": min(max(priority, 1), 3),
                "scope_level": scope_level,
                "role": role,
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
        if path and path in valid_paths:
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
    llm: LLM | None,
    *,
    use_llm: bool = True,
    max_files: int = 6,
) -> tuple[TenderInventory, list[TenderFileEntry], dict[str, Any]]:
    """
    Инвентаризация папки + выбор файлов для extract.
    Возвращает inventory (нужно закрыть через cleanup), entries и результат выбора.
    """
    inventory = inventory_tender_folder(folder)
    entries = build_catalog_from_inventory(inventory)
    tree_text = format_catalog_tree(entries)

    if not entries:
        return inventory, entries, {
            "mode": "empty",
            "files": [],
            "skip": [],
            "catalog_paths": [],
            "tree": tree_text,
            "warnings": list(inventory.warnings),
        }

    if use_llm and llm is not None:
        try:
            selection = select_tender_files_by_llm(
                entries, tree_text, llm, max_files=max_files
            )
        except Exception as exc:
            selection = select_files_heuristic(entries, max_files=max_files)
            selection["mode"] = "heuristic_llm_error"
            selection["error"] = str(exc)
    else:
        selection = select_files_heuristic(entries, max_files=max_files)

    if not selection.get("files"):
        selection = select_files_heuristic(entries, max_files=max_files)
        selection["mode"] = selection.get("mode", "heuristic") + "_fallback"

    selection["tree"] = tree_text
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
    """Каталог тендера → LLM/heuristic выбор файлов."""
    from .common import progress

    settings: Settings = state["settings"]
    progress(state, "Каталог и выбор файлов тендера", 0.1)
    inventory, catalog_entries, doc_selection = select_tender_files(
        Path(state["tender_path"]),
        state["llm"],
        use_llm=True,
        max_files=settings.max_tender_files_total,
    )
    box = state.get("cleanup_box")
    if isinstance(box, dict):
        box["inventory"] = inventory

    ranked = ranked_file_paths(doc_selection)
    if not ranked:
        raise ValueError("Не выбрано ни одного файла тендера для анализа")

    initial_queue = ranked[: max(1, settings.max_tender_files_initial)]
    progress(
        state,
        (
            f"Выбрано {len(ranked)} из {doc_selection.get('catalog_count', 0)} "
            f"файлов ({doc_selection.get('mode', 'llm')})"
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
        "requirement_queue": [],
        "requirement_files_tried": [],
        "current_requirement_file": "",
        "position_matches": [],
        "verdict": "",
        "query_selection": {},
        "indexed_files": [],
        "index_reused": False,
    }
