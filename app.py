import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from ai_tender.models import (
    STATUS_LABELS,
    AnalysisReport,
    Evidence,
    get_settings,
)
from ai_tender.ocr import ocr_status
from ai_tender.pipeline import analyze
from ai_tender.viewer import build_document_view

st.set_page_config(page_title="AI Tender", page_icon="📋", layout="wide")
st.title("AI Tender")
st.caption("Требование тендера → подтверждение в эталоне (hybrid RAG + LLM)")

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


@st.dialog("Фрагмент в документе", width="large")
def show_evidence_dialog(evidence: Evidence, root: str | None, role: str) -> None:
    view = build_document_view(evidence, root, role=role)
    st.markdown(f"**{view.title}**")
    st.caption(f"{view.location}" + (f" · `{view.path}`" if view.path else ""))
    if view.note:
        st.info(view.note)
    st.markdown(view.body_html, unsafe_allow_html=True)


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes} мин {rest:.0f} с"


def render_report(report: AnalysisReport, tender_root: str, assets_root: str) -> None:
    cache_note = (
        "индекс эталонов из кэша"
        if report.index_reused
        else "индекс эталонов построен заново"
    )
    elapsed = format_elapsed(report.elapsed_seconds)
    st.success(f"Готово за {elapsed}. {cache_note.capitalize()}.")

    if report.summary:
        # st.info поддерживает markdown; двойной пробел+\\n сохраняет переносы.
        st.info(report.summary.replace("\n", "  \n"))

    qs = report.query_selection or {}
    truncated = qs.get("truncated_files") or []
    if truncated:
        st.warning(
            "Документ(ы) обрезаны по лимиту длины при extract: "
            + ", ".join(Path(name).name for name in truncated)
        )
    if qs.get("error"):
        st.warning(f"Extract: fallback из‑за ошибки — {qs['error']}")

    top_reqs = qs.get("top_requirements") or []
    if top_reqs:
        with st.expander("Извлечённые требования (топ)"):
            for item in top_reqs:
                loc = item.get("location") or ""
                st.markdown(
                    f"- **p{item.get('priority', '?')}** "
                    f"({item.get('confidence', 0):.2f}): {item.get('text', '')}"
                    + (f"  \n  _{loc}_" if loc else "")
                )

    if getattr(report, "extracted_requirements", None):
        with st.expander(
            f"Все извлечённые требования ({len(report.extracted_requirements)})"
        ):
            for index, req in enumerate(report.extracted_requirements, start=1):
                st.markdown(
                    f"**{index}. {req.text}**  \n"
                    f"`{req.location}` · `{Path(req.file).name}`  \n"
                    f"> {req.quote[:300]}"
                )

    if report.indexed_files:
        with st.expander(f"Эталоны в индексе ({len(report.indexed_files)} файлов)"):
            st.write("\n".join(f"- {Path(path).name}" for path in report.indexed_files))

    if not report.findings:
        st.warning("Результатов нет — увеличьте число требований или top-k.")
    else:
        rows = [
            {
                "Статус": STATUS_LABELS[item.status.value],
                "Тип": item.kind or "—",
                "Требование тендера": item.tender.quote[:200],
                "Файл тендера": Path(item.tender.file).name,
                "Эталон": (
                    Path(item.asset_hits[0].file).name if item.asset_hits else "—"
                ),
                "Хитов эталона": len(item.asset_hits),
                "Пояснение": item.explanation[:200],
                "Уверенность": round(item.confidence, 2),
            }
            for item in report.findings
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

        st.subheader("Детали")
        for index, item in enumerate(report.findings, start=1):
            title = (
                f"{index}. {STATUS_LABELS[item.status.value]} — "
                f"{Path(item.tender.file).name}"
            )
            with st.expander(title, expanded=(index == 1)):
                st.markdown(f"**Пояснение:** {item.explanation or '—'}")
                left, right = st.columns(2)
                with left:
                    st.markdown("**Тендер**")
                    if item.query_text and item.query_text != item.tender.quote[:300]:
                        st.markdown(f"**Требование:** {item.query_text}")
                    st.caption(f"{item.tender.file} · {item.tender.location}")
                    st.write(item.tender.quote)
                    if st.button(
                        "Показать в документе",
                        key=f"view_tender_{index}",
                        width="stretch",
                    ):
                        show_evidence_dialog(item.tender, tender_root, "Тендер")
                with right:
                    st.markdown("**Эталон**")
                    if not item.asset_hits:
                        st.write("Нет хитов")
                    for hit_index, hit in enumerate(item.asset_hits):
                        score = (
                            f" · score={hit.score:.3f}"
                            if hit.score is not None
                            else ""
                        )
                        st.caption(f"{hit.file} · {hit.location}{score}")
                        st.write(hit.quote)
                        if st.button(
                            "Показать в документе",
                            key=f"view_asset_{index}_{hit_index}",
                            width="stretch",
                        ):
                            show_evidence_dialog(hit, assets_root, "Эталон")

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

    strategy_options = {
        "hybrid": "Гибридный",
        "product": "По продукту",
        "specs": "По техсоответствию",
    }
    default_strategy = (
        settings.match_strategy if settings.match_strategy in strategy_options else "hybrid"
    )
    match_strategy = st.radio(
        "Стратегия поиска",
        options=list(strategy_options.keys()),
        format_func=lambda key: strategy_options[key],
        index=list(strategy_options.keys()).index(default_strategy),
        help=(
            "Гибридный: сначала артикул/название в эталоне; если нашёл — "
            "несколько ключевых ТТХ; если нет — полный разбор ТТХ.\n\n"
            "По продукту: только явные позиции (например МИР С-05…).\n\n"
            "По техсоответствию: проверка технических требований без акцента на артикул."
        ),
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
                "match_strategy": match_strategy,
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
