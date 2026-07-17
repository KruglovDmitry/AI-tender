import os
from pathlib import Path

import streamlit as st

from ai_tender.models import DEFAULT_USER_INSTRUCTION, STATUS_LABELS, get_settings
from ai_tender.ocr import ocr_status
from ai_tender.pipeline import analyze

st.set_page_config(page_title="AI Tender", page_icon="📋", layout="wide")
st.title("AI Tender")
st.caption("Требование тендера → подтверждение в эталоне (hybrid RAG + LLM)")

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
tender_input = st.text_input("Папка с документами тендера", value=default_tender)
assets_input = st.text_input("Папка с эталонными документами", value=default_assets)

if st.button("Начать сравнение", type="primary", use_container_width=True):
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
                st.dataframe(rows, use_container_width=True, hide_index=True)

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
