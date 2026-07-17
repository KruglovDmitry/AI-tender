import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from ai_tender.models import DEFAULT_USER_INSTRUCTION, STATUS_LABELS, get_settings
from ai_tender.ocr import ocr_status
from ai_tender.pipeline import analyze

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

    embedding_model = st.text_input("Модель embeddings", value=settings.embedding_model)
    st.caption("Embeddings локально (bge-m3). Индекс эталонов кэшируется на диске.")

    top_k = st.slider("Top-K фрагментов эталона на требование", 1, 10, settings.top_k)
    max_tender_queries = st.number_input(
        "Макс. требований из тендера",
        min_value=5,
        max_value=50,
        value=settings.max_tender_queries,
    )
    max_findings = st.number_input(
        "Макс. строк в результате",
        min_value=3,
        max_value=40,
        value=settings.max_findings,
    )

    chunk_size = st.number_input(
        "Размер чанка",
        min_value=256,
        max_value=4000,
        value=settings.chunk_size,
        step=128,
    )
    chunk_overlap = st.number_input(
        "Overlap чанка",
        min_value=0,
        max_value=500,
        value=settings.chunk_overlap,
        step=32,
    )

    st.subheader("Инструкция для LLM")
    user_instruction = st.text_area(
        "Задача оценки (префикс промпта)",
        value=settings.user_instruction or DEFAULT_USER_INSTRUCTION,
        height=180,
        help="Этот текст подставляется в промпт перед схемой ответа и данными.",
    )

    st.subheader("OCR для сканов PDF")
    ocr_enabled = st.checkbox("Распознавать сканы (Tesseract)", value=settings.ocr_enabled)
    ocr_languages = st.text_input("Языки OCR", value=settings.ocr_languages)
    ocr_ok, ocr_hint = ocr_status()
    if ocr_enabled and not ocr_ok:
        st.warning(ocr_hint)
    elif ocr_enabled:
        st.caption(f"Tesseract: {ocr_hint}")

default_tender = str((Path.cwd() / "sources" / "1").resolve())
default_assets = str((Path.cwd() / "assets").resolve())
if "tender_folder" not in st.session_state:
    st.session_state.tender_folder = default_tender
if "assets_folder" not in st.session_state:
    st.session_state.assets_folder = default_assets

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
                "embedding_model": embedding_model,
                "top_k": top_k,
                "max_tender_queries": int(max_tender_queries),
                "max_findings": int(max_findings),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "user_instruction": user_instruction.strip() or DEFAULT_USER_INSTRUCTION,
                "ocr_enabled": ocr_enabled,
                "ocr_languages": ocr_languages.strip() or "rus+eng",
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
            progress_bar.progress(1.0)
            cache_note = (
                "индекс эталонов из кэша"
                if report.index_reused
                else "индекс эталонов построен заново"
            )
            status.success(f"Готово. {cache_note.capitalize()}.")

            if report.summary:
                st.info(report.summary)

            if report.indexed_files:
                with st.expander(
                    f"Эталоны в индексе ({len(report.indexed_files)} файлов)"
                ):
                    st.write(
                        "\n".join(f"- {Path(path).name}" for path in report.indexed_files)
                    )

            if not report.findings:
                st.warning(
                    "Результатов нет — увеличьте число требований или top-k."
                )
            else:
                rows = [
                    {
                        "Статус": STATUS_LABELS[item.status.value],
                        "Требование тендера": item.tender.quote[:200],
                        "Файл тендера": Path(item.tender.file).name,
                        "Эталон": (
                            Path(item.asset_hits[0].file).name
                            if item.asset_hits
                            else "—"
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
                    with st.expander(title):
                        st.markdown(f"**Пояснение:** {item.explanation or '—'}")
                        left, right = st.columns(2)
                        left.markdown("**Тендер**")
                        left.caption(f"{item.tender.file} · {item.tender.location}")
                        left.write(item.tender.quote)

                        right.markdown("**Эталон**")
                        if not item.asset_hits:
                            right.write("Нет хитов")
                        for hit in item.asset_hits:
                            score = (
                                f" · score={hit.score:.3f}"
                                if hit.score is not None
                                else ""
                            )
                            right.caption(f"{hit.file} · {hit.location}{score}")
                            right.write(hit.quote)

            if report.warnings:
                with st.expander(f"Предупреждения ({len(report.warnings)})"):
                    st.write("\n".join(f"- {warning}" for warning in report.warnings))
