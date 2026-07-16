import os
from pathlib import Path

import streamlit as st

from ai_tender.models import STATUS_LABELS, get_settings
from ai_tender.pipeline import analyze

st.set_page_config(page_title="AI Tender", page_icon="📋", layout="wide")
st.title("AI Tender")
st.caption("LlamaIndex RAG: поиск вхождений эталона в тендер и оценка LLM")

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

    top_k = st.slider("Top-K фрагментов тендера на запрос", 1, 10, settings.top_k)
    max_asset_queries = st.number_input(
        "Макс. запросов из эталона",
        min_value=5,
        max_value=200,
        value=settings.max_asset_queries,
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
                "max_asset_queries": int(max_asset_queries),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
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
            cache_note = "индекс эталонов из кэша" if report.index_reused else "индекс эталонов построен заново"
            status.success(f"Готово. {cache_note.capitalize()}.")

            if report.summary:
                st.info(report.summary)

            if not report.findings:
                st.warning(
                    "Результатов нет — попробуйте увеличить max запросов или top-k, а также снизить порог min_retrieval_score."
                )
            else:
                rows = [
                    {
                        "Статус": STATUS_LABELS[item.status.value],
                        "Эталон": Path(item.asset.file).name,
                        "Место эталона": item.asset.location,
                        "Фрагмент эталона": item.asset.quote[:200],
                        "Хитов в тендере": len(item.tender_hits),
                        "Пояснение LLM": item.explanation[:240],
                        "Уверенность": round(item.confidence, 2),
                    }
                    for item in report.findings
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

                st.subheader("Детали")
                for index, item in enumerate(report.findings, start=1):
                    title = f"{index}. {STATUS_LABELS[item.status.value]} — {Path(item.asset.file).name}"
                    with st.expander(title):
                        st.markdown(f"**Пояснение:** {item.explanation or '—'}")
                        left, right = st.columns(2)
                        left.markdown("**Эталон**")
                        left.caption(f"{item.asset.file} · {item.asset.location}")
                        left.write(item.asset.quote)

                        right.markdown("**Тендер**")
                        if not item.tender_hits:
                            right.write("Нет хитов")

                        for hit in item.tender_hits:
                            score = f" · score={hit.score:.3f}" if hit.score is not None else ""
                            right.caption(f"{hit.file} · {hit.location}{score}")
                            right.write(hit.quote)

            if report.warnings:
                with st.expander(f"Предупреждения ({len(report.warnings)})"):
                    st.write("\n".join(f"- {warning}" for warning in report.warnings))

