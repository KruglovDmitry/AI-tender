import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from ai_tender.models import (
    AnalysisReport,
    get_settings,
)
from ai_tender.ocr import ocr_status
from ai_tender.graph import analyze

st.set_page_config(page_title="AI Tender", page_icon="📋", layout="wide")
st.title("AI Tender")
st.caption("Предмет закупки (перечень позиций)")

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
    </style>
    """,
    unsafe_allow_html=True,
)


def is_running_in_docker() -> bool:
    return os.environ.get("RUNNING_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


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
    if is_running_in_docker():
        st.caption(
            "Docker: папки с хоста смонтированы как "
            "`sources` → `/data/tender`, `assets` → `/data/assets`."
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
    del tender_root, assets_root  # этап сверки с эталоном пока отключён
    elapsed = format_elapsed(report.elapsed_seconds)
    st.success(f"Готово за {elapsed}.")

    if report.summary:
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
    items = scope.get("items") or []
    with st.expander("Предмет закупки", expanded=True):
        st.caption(
            f"confidence={scope.get('overall_confidence', '—')} · "
            f"needs_more_docs={scope.get('needs_more_docs', False)}"
        )
        summary = (scope.get("summary") or "").strip()
        if summary:
            st.markdown(f"**Титул:** {summary}")
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
                "Файлы: " + ", ".join(Path(path).name for path in files_used)
            )

    if report.warnings:
        with st.expander(f"Предупреждения ({len(report.warnings)})"):
            st.write("\n".join(f"- {warning}" for warning in report.warnings))


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

    ocr_enabled = st.checkbox(
        "OCR для сканов PDF",
        value=settings.ocr_enabled,
    )
    ocr_ok, ocr_hint = ocr_status()
    if ocr_enabled and not ocr_ok:
        st.warning(ocr_hint)

default_tender = default_tender_path()
default_assets = default_assets_path()
if "tender_folder" not in st.session_state:
    st.session_state.tender_folder = default_tender
if "assets_folder" not in st.session_state:
    st.session_state.assets_folder = default_assets

if is_running_in_docker():
    st.info(
        "Режим Docker. Положите документы в папки `sources` и `assets` рядом с проектом "
        "или укажите путь внутри контейнера (`/data/tender`, `/data/assets`)."
    )

tender_input = folder_path_input(
    "Папка с документами тендера",
    state_key="tender_folder",
    pick_key="pick_tender_folder",
)
assets_input = folder_path_input(
    "Папка с эталонными документами",
    state_key="assets_folder",
    pick_key="pick_assets_folder",
)

if st.button("Начать сравнение", type="primary", width="stretch"):
    tender_path = Path(tender_input).expanduser()
    assets_path = Path(assets_input).expanduser()

    if not api_key:
        st.error("Укажите API key для выбранного LLM-провайдера.")
    elif not tender_path.is_dir() or not assets_path.is_dir():
        st.error("Один из указанных путей не существует или не является папкой.")
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
            }
        )

        try:
            report = analyze(
                tender_path,
                assets_path,
                settings=runtime_settings,
                progress=update,
            )
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
