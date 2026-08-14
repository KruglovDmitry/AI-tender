import os
import subprocess
import sys
from html import escape
from pathlib import Path

import streamlit as st

from ai_tender.models import (
    POSITION_STATUS_LABELS,
    AnalysisReport,
    PositionMatchStatus,
    get_settings,
)
from ai_tender.services.index_service import indexed_file_paths, load_or_build_assets_index
from ai_tender.services.ocr_service import ocr_status
from ai_tender.services.report_export import report_to_json_bytes, report_to_markdown
from ai_tender.services.upload_service import (
    cleanup_old_uploads,
    new_run_dir,
    prepare_upload_dir,
    replace_shared_assets,
)
from ai_tender.graph import analyze, warm_up_graph

# Компиляция структуры графа при старте приложения (один раз).
warm_up_graph()

st.set_page_config(page_title="AI Tender", page_icon="📋", layout="wide")
st.title("AI Tender")
st.caption("Предмет закупки → требования → эталон")

st.markdown(
    """
    <style>
    .folder-input-label {
        font-size: 0.875rem;
        margin: 0 0 0.35rem 0;
        color: rgba(250, 250, 250, 0.85);
    }
    .folder-input-spacer {
        margin-bottom: 1rem;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group) {
        display: flex !important;
        align-items: stretch !important;
        gap: 0 !important;
        margin: 0 0 1rem 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div[data-testid="column"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div[data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div[data-testid="column"]:last-child {
        flex: 0 0 2.75rem !important;
        width: 2.75rem !important;
        min-width: 2.75rem !important;
        max-width: 2.75rem !important;
        margin-left: -1px !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:first-child [data-testid="stTextInput"] {
        margin: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:first-child [data-testid="stTextInput"] > div,
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:first-child [data-testid="stTextInput"] > div > div,
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:first-child [data-testid="stTextInput"] input {
        border-top-right-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
        min-height: 2.5rem !important;
        height: 2.5rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:first-child [data-testid="stTextInput"] > div {
        border-right: 0 !important;
        border-top-left-radius: 0.5rem !important;
        border-bottom-left-radius: 0.5rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:first-child [data-testid="stTextInput"] input {
        border-top-left-radius: 0.5rem !important;
        border-bottom-left-radius: 0.5rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:last-child [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        justify-content: flex-end !important;
        height: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:last-child [data-testid="stButton"] {
        width: 100% !important;
        margin: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:last-child [data-testid="stButton"] > button {
        width: 2.75rem !important;
        min-height: 2.5rem !important;
        height: 2.5rem !important;
        padding: 0 !important;
        margin: 0 !important;
        border-top-left-radius: 0 !important;
        border-bottom-left-radius: 0 !important;
        border-top-right-radius: 0.5rem !important;
        border-bottom-right-radius: 0.5rem !important;
        border: 1px solid rgba(128, 128, 128, 0.35) !important;
        border-left: 1px solid rgba(128, 128, 128, 0.35) !important;
        background-color: rgba(250, 250, 250, 0.06) !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ff4b4b'%3E%3Cpath d='M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 1.1rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:last-child [data-testid="stButton"] > button p {
        display: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.folder-input-group)
    > div:last-child [data-testid="stButton"] > button:hover {
        background-color: rgba(250, 250, 250, 0.1) !important;
        border-color: rgba(128, 128, 128, 0.45) !important;
    }
    .pos-match-row {
        margin: 0.35rem 0 0.15rem 0;
    }
    .pos-match-title {
        margin: 0;
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .pos-match-status {
        margin: 0;
        text-align: right;
        white-space: nowrap;
        font-size: 0.9rem;
        opacity: 0.92;
    }
    .pos-match-status.is-none { color: #ff6b6b; }
    .pos-match-status.is-partial { color: #ffd166; }
    .pos-match-status.is-matched { color: #6bcb77; }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_running_in_docker() -> bool:
    return os.environ.get("RUNNING_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


def folder_picker_available() -> bool:
    """GUI-выбор папки доступен только при локальном дисплее (не headless/стенд)."""
    if is_running_in_docker():
        return False
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def default_tender_path() -> str:
    if is_running_in_docker():
        for candidate in (Path("/data/tender/1"), Path("/data/tender")):
            if candidate.is_dir():
                return str(candidate)
        return "/data/tender"
    return str((Path.cwd() / "sources" / "1").resolve())


def default_assets_path() -> str:
    if is_running_in_docker():
        return "/data/assets"
    return str((Path.cwd() / "assets").resolve())


def pick_folder_dialog(initial: str | None = None) -> str | None:
    """Диалог выбора папки в отдельном процессе (совместимо со Streamlit)."""
    if not folder_picker_available():
        raise RuntimeError(
            "На стенде нет GUI-дисплея. Введите путь к папке на сервере вручную."
        )

    initialdir = ""
    if initial:
        initial_path = Path(initial).expanduser()
        if initial_path.is_dir():
            initialdir = str(initial_path.resolve())

    if sys.platform == "win32":
        safe_initial = initialdir.replace("'", "''")
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Выберите папку'
$initial = '{safe_initial}'
if ($initial -and (Test-Path $initial)) {{ $dialog.SelectedPath = $initial }}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.SelectedPath
}}
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 and result.stderr.strip():
            raise RuntimeError(result.stderr.strip())
        folder = result.stdout.strip()
        return folder or None

    tk_script = (
        "import sys, tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "kwargs = {}\n"
        "if len(sys.argv) > 1 and sys.argv[1]:\n"
        "    kwargs['initialdir'] = sys.argv[1]\n"
        "folder = filedialog.askdirectory(**kwargs)\n"
        "print(folder or '')\n"
        "root.destroy()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", tk_script, initialdir],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 and result.stderr.strip():
        raise RuntimeError(result.stderr.strip())
    folder = result.stdout.strip()
    return folder or None


def _pick_folder_callback(state_key: str, error_key: str) -> None:
    try:
        chosen = pick_folder_dialog(st.session_state.get(state_key))
        if chosen:
            st.session_state[state_key] = chosen
    except Exception as exc:
        st.session_state[error_key] = f"Не удалось открыть диалог: {exc}"


def folder_path_input(label: str, state_key: str, pick_key: str) -> str:
    if not folder_picker_available():
        if is_running_in_docker():
            st.caption(
                "Docker: папки с хоста смонтированы как "
                "`sources` → `/data/tender`, `assets` → `/data/assets`."
            )
        else:
            st.caption(
                "Стенд без GUI: укажите путь к папке на сервере вручную "
                "(например `sources` или абсолютный путь)."
            )
        return st.text_input(label, key=state_key)

    error_key = f"{pick_key}__error"
    if error_key in st.session_state:
        st.error(st.session_state.pop(error_key))

    st.markdown(f'<p class="folder-input-label">{label}</p>', unsafe_allow_html=True)
    col_input, col_pick = st.columns([24, 1], gap=None)
    with col_input:
        st.markdown('<span class="folder-input-group"></span>', unsafe_allow_html=True)
        st.text_input(label, key=state_key, label_visibility="collapsed")
    with col_pick:
        st.button(
            " ",
            key=pick_key,
            on_click=_pick_folder_callback,
            args=(state_key, error_key),
        )
    return str(st.session_state[state_key])


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes} мин {rest:.0f} с"


def render_report(report: AnalysisReport, tender_root: str, assets_root: str) -> None:
    del tender_root, assets_root
    elapsed = format_elapsed(report.elapsed_seconds)
    cache_note = (
        "индекс эталонов из кэша"
        if report.index_reused
        else "индекс эталонов построен заново"
    )
    st.success(f"Готово за {elapsed}. {cache_note.capitalize()}.")
    trace_dir = (report.query_selection or {}).get("llm_trace_dir")
    if trace_dir:
        st.caption(f"Логи LLM/retrieval: `{trace_dir}`")

    md_bytes = report_to_markdown(report).encode("utf-8")
    json_bytes = report_to_json_bytes(report)
    col_md, col_json = st.columns(2)
    with col_md:
        st.download_button(
            "Скачать отчёт (.md)",
            data=md_bytes,
            file_name="ai-tender-report.md",
            mime="text/markdown",
            width="stretch",
        )
    with col_json:
        st.download_button(
            "Скачать отчёт (.json)",
            data=json_bytes,
            file_name="ai-tender-report.json",
            mime="application/json",
            width="stretch",
        )

    if report.verdict:
        st.subheader("Итоговый вывод")
        st.info(report.verdict.replace("\n", "  \n"))
    elif report.summary:
        st.info(report.summary.replace("\n", "  \n"))

    qs = report.query_selection or {}
    doc_sel = qs.get("doc_selection") or {}
    if doc_sel.get("selected"):
        with st.expander(
            f"Выбор файлов тендера ({len(doc_sel.get('loaded') or [])} загружено "
            f"из {doc_sel.get('catalog_count', '?')})"
        ):
            st.caption(f"Режим: {doc_sel.get('mode', '—')}")
            for item in doc_sel.get("selected") or []:
                path = item.get("path", "")
                loaded = path in (doc_sel.get("loaded") or [])
                mark = "✓" if loaded else "·"
                reason = item.get("reason", "")
                st.markdown(
                    f"{mark} **p{item.get('priority', '?')}** `{Path(path).name}` "
                    f"(scope={item.get('scope_level', '—')}, {item.get('role', '—')})"
                    + (f" — {reason}" if reason else "")
                )
            skipped = doc_sel.get("skipped") or []
            if skipped:
                st.markdown("**Пропущены:**")
                for item in skipped[:12]:
                    st.markdown(
                        f"- `{Path(item.get('path', '')).name}` — {item.get('reason', '')}"
                    )
            if doc_sel.get("error"):
                st.warning(f"Выбор файлов: fallback — {doc_sel['error']}")

    scope = qs.get("scope") or {}
    matches = list(getattr(report, "position_matches", None) or [])
    with st.expander("Предмет закупки", expanded=True):
        st.caption(
            f"confidence={scope.get('overall_confidence', '—')} · "
            f"needs_more_docs={scope.get('needs_more_docs', False)}"
        )
        summary = (scope.get("summary") or "").strip()
        if summary:
            st.markdown(f"**Титул:** {summary}")

        if matches:
            for index, match in enumerate(matches, start=1):
                qty_part = (
                    f" — {match.qty} {match.unit}".rstrip()
                    if match.qty is not None
                    else ""
                )
                status_label = POSITION_STATUS_LABELS.get(
                    match.status.value, match.status.value
                )
                status_class = {
                    PositionMatchStatus.matched.value: "is-matched",
                    PositionMatchStatus.partial.value: "is-partial",
                    PositionMatchStatus.none.value: "is-none",
                }.get(match.status.value, "")

                st.markdown('<div class="pos-match-row"></div>', unsafe_allow_html=True)
                col_title, col_status = st.columns([5, 1], gap="small")
                with col_title:
                    st.markdown(
                        f'<p class="pos-match-title"><strong>{index}. '
                        f"{escape(str(match.scope_name))}</strong>"
                        f"{escape(qty_part)}</p>",
                        unsafe_allow_html=True,
                    )
                with col_status:
                    st.markdown(
                        f'<p class="pos-match-status {status_class}">'
                        f"{escape(status_label)}</p>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Детали", expanded=(index == 1)):
                    st.caption(f"Статус: {status_label} · conf={match.confidence:.2f}")

                    if match.requirements:
                        st.markdown("**Требования:**")
                        for req in match.requirements:
                            st.markdown(f"- {req.text}")
                    else:
                        st.caption("Требования не найдены.")

                    if match.status == PositionMatchStatus.none or not match.product_name:
                        st.markdown("**Эталон:** нет подходящего варианта")
                    else:
                        st.markdown(f"**Эталон:** {match.product_name}")
                    if match.explanation:
                        st.write(match.explanation)
                    if match.asset_hits:
                        with st.expander(f"Фрагменты эталона ({len(match.asset_hits)})"):
                            for hit in match.asset_hits[:5]:
                                score = (
                                    f" · score={hit.score:.3f}"
                                    if hit.score is not None
                                    else ""
                                )
                                st.caption(f"{Path(hit.file).name} · {hit.location}{score}")
                                st.write(hit.quote[:400])
        else:
            items = scope.get("items") or []
            if items:
                st.markdown("**Перечень позиций:**")
                for index, item in enumerate(items, start=1):
                    if isinstance(item, dict):
                        name = str(item.get("name") or "").strip() or "—"
                        qty = item.get("qty")
                        unit = str(item.get("unit") or "").strip()
                        qty_part = f" — {qty} {unit}".rstrip() if qty is not None else ""
                        st.markdown(f"{index}. {name}{qty_part}")
                    else:
                        st.markdown(f"{index}. {item}")
            elif scope.get("missing_signals"):
                st.warning(scope.get("missing_signals"))
            else:
                st.warning("Перечень позиций не извлечён.")

        files_used = scope.get("files_used") or []
        if files_used:
            st.caption(
                "Файлы scope: " + ", ".join(Path(path).name for path in files_used)
            )

    if report.indexed_files:
        with st.expander(f"Эталоны в индексе ({len(report.indexed_files)} файлов)"):
            st.write("\n".join(f"- {Path(path).name}" for path in report.indexed_files))

    if report.warnings:
        with st.expander(f"Предупреждения ({len(report.warnings)})"):
            st.write("\n".join(f"- {warning}" for warning in report.warnings))


default_tender = default_tender_path()
default_assets = default_assets_path()
if "tender_folder" not in st.session_state:
    st.session_state.tender_folder = default_tender
if "assets_folder" not in st.session_state:
    st.session_state.assets_folder = default_assets

cleanup_old_uploads()

settings = get_settings()
with st.sidebar:
    st.header("Настройки")
    llm_provider = st.selectbox(
        "LLM-провайдер",
        options=["deepseek", "openai"],
        index=0 if settings.llm_provider == "deepseek" else 1,
    )
    llm_model = st.text_input("Модель LLM", value=settings.llm_model)
    if llm_provider == "deepseek":
        api_key = st.text_input(
            "DeepSeek API key",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
        )
    else:
        api_key = st.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )

    _assets_update_message = st.session_state.pop("assets_update_message", None)
    if _assets_update_message:
        st.success(_assets_update_message)

    with st.expander("Эталоны", expanded=False):
        st.caption(
            "Эталоны — справочные ТС продукции на сервере. "
            "По ним сравнивается тендер. Обновление заменяет общий комплект "
            "для стенда и сразу строит индекс."
        )
        info = st.session_state.get("assets_index_info")
        if info and info.get("files"):
            st.caption(f"В индексе: {len(info['files'])} файл(ов)")
        assets_uploads = st.file_uploader(
            "Новый пакет (файлы или ZIP/RAR)",
            accept_multiple_files=True,
            key="assets_uploader",
        )
        if st.button(
            "Обновить эталоны",
            key="update_assets_btn",
            type="secondary",
            width="stretch",
            disabled=not assets_uploads,
        ):
            try:
                with st.spinner("Сохранение эталонов..."):
                    assets_root = Path(st.session_state.assets_folder).expanduser()
                    _, upload_warnings = replace_shared_assets(
                        list(assets_uploads), assets_root
                    )
                with st.spinner(
                    "Построение индекса эталонов (может занять несколько минут)..."
                ):
                    _index, nodes, index_warnings, reused = load_or_build_assets_index(
                        assets_root.resolve(),
                        cache_dir=settings.cache_dir,
                        embedding_model=settings.embedding_model,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        device=settings.embedding_device,
                        ocr_enabled=settings.ocr_enabled,
                        ocr_languages=settings.ocr_languages,
                    )
                files = indexed_file_paths(nodes)
                st.session_state["assets_index_info"] = {
                    "files": files,
                    "reused": reused,
                    "warnings": upload_warnings + index_warnings,
                }
                st.session_state["assets_update_message"] = (
                    f"Эталоны обновлены. Индекс: {len(files)} файлов"
                    + (" (кэш)" if reused else " (заново)")
                    + "."
                )
                st.rerun()
            except Exception as exc:
                st.exception(exc)
        for warning in (info or {}).get("warnings") or []:
            st.warning(warning)

    ocr_enabled = st.checkbox(
        "OCR для сканов PDF",
        value=settings.ocr_enabled,
    )
    ocr_ok, ocr_hint = ocr_status()
    if ocr_enabled and not ocr_ok:
        st.warning(ocr_hint)

    max_reqs_per_scope_item = st.slider(
        "Макс. требований на позицию",
        min_value=1,
        max_value=20,
        value=min(max(settings.max_reqs_per_scope_item, 1), 20),
        help="Верхний лимит извлечённых требований на каждую позицию перечня. Минимум не ограничен.",
    )

tender_source = st.radio(
    "Источник документов тендера",
    options=["upload", "folder"],
    format_func=lambda value: (
        "Загрузить файлы" if value == "upload" else "Папка на сервере"
    ),
    horizontal=True,
    key="tender_source",
)

tender_uploads = None
tender_input = st.session_state.tender_folder
if tender_source == "upload":
    tender_uploads = st.file_uploader(
        "Документы тендера (файлы или ZIP/RAR)",
        accept_multiple_files=True,
        key="tender_uploader",
    )
    if tender_uploads:
        st.caption(f"Выбрано файлов: {len(tender_uploads)}")
else:
    if is_running_in_docker():
        st.info(
            "Docker: папки с хоста смонтированы как "
            "`sources` → `/data/tender`, `assets` → `/data/assets`."
        )
    tender_input = folder_path_input(
        "Папка с документами тендера",
        state_key="tender_folder",
        pick_key="pick_tender_folder",
    )

if st.button("Начать сравнение", type="primary", width="stretch"):
    assets_path = Path(st.session_state.assets_folder).expanduser()

    if not api_key:
        st.error("Укажите API key для выбранного LLM-провайдера.")
    elif not assets_path.is_dir():
        st.error("Каталог эталонов не существует.")
    elif tender_source == "upload" and not tender_uploads:
        st.error("Загрузите файлы тендера или выберите папку на сервере.")
    else:
        if llm_provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key

        progress_bar = st.progress(0.0)
        status = st.empty()

        def update(message: str, value: float) -> None:
            status.info(message)
            progress_bar.progress(min(max(value, 0.0), 1.0))

        runtime_settings = settings.model_copy(
            update={
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "ocr_enabled": ocr_enabled,
                "max_reqs_per_scope_item": max_reqs_per_scope_item,
            }
        )

        try:
            if tender_source == "upload":
                update("Сохранение загруженного тендера...", 0.02)
                run_dir = new_run_dir("tender")
                tender_path, upload_warnings = prepare_upload_dir(
                    list(tender_uploads),
                    run_dir / "tender",
                )
            else:
                tender_path = Path(tender_input).expanduser()
                upload_warnings = []
                if not tender_path.is_dir():
                    raise ValueError("Папка тендера не существует.")

            report = analyze(
                tender_path,
                assets_path,
                settings=runtime_settings,
                progress=update,
            )
            if upload_warnings:
                report.warnings = list(report.warnings or []) + upload_warnings
        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.exception(exc)
        else:
            progress_bar.empty()
            status.empty()
            st.session_state["last_report"] = report
            st.session_state["last_tender_root"] = str(tender_path.resolve())
            st.session_state["last_assets_root"] = str(assets_path.resolve())

if "last_report" in st.session_state:
    render_report(
        st.session_state["last_report"],
        st.session_state.get("last_tender_root", ""),
        st.session_state.get("last_assets_root", ""),
    )
