import os
from pathlib import Path

import streamlit as st

from ai_tender.config import get_settings
from ai_tender.pipeline import analyze
from ai_tender.reporting import STATUS_LABELS, save_report

st.set_page_config(page_title="AI Tender", page_icon="📋", layout="wide")
st.title("AI Tender")
st.caption("Сопоставление технических требований закупки с эталонной документацией")

settings = get_settings()
with st.sidebar:
    st.header("Настройки")
    model = st.text_input("Модель сравнения", value=settings.model)
    embedding_model = st.text_input("Модель embeddings", value=settings.embedding_model)
    api_key = st.text_input(
        "DeepSeek API key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        help="Ключ используется только для текущего запуска и не сохраняется приложением.",
    )
    st.caption("BAAI/bge-m3 выполняется локально; при первом запуске модель будет загружена.")
    max_requirements = st.number_input(
        "Максимум требований", min_value=1, max_value=500, value=settings.max_requirements
    )
    top_k = st.slider("Фрагментов эталона на требование", 1, 10, settings.top_k)

default_tender = str((Path.cwd() / "sources" / "1").resolve())
default_assets = str((Path.cwd() / "assets").resolve())
tender_input = st.text_input("Папка с документами тендера", value=default_tender)
assets_input = st.text_input("Папка с эталонными документами", value=default_assets)

if st.button("Начать сравнение", type="primary", use_container_width=True):
    tender_path = Path(tender_input).expanduser()
    assets_path = Path(assets_input).expanduser()
    if not api_key:
        st.error("Укажите DeepSeek API key или переменную DEEPSEEK_API_KEY.")
    elif not tender_path.is_dir() or not assets_path.is_dir():
        st.error("Один из указанных путей не существует или не является папкой.")
    else:
        progress_bar = st.progress(0.0)
        status = st.empty()

        def update(message: str, value: float) -> None:
            status.info(message)
            progress_bar.progress(min(max(value, 0.0), 1.0))

        runtime_settings = settings.model_copy(
            update={
                "model": model,
                "embedding_model": embedding_model,
                "max_requirements": int(max_requirements),
                "top_k": top_k,
            }
        )
        try:
            report = analyze(
                tender_path,
                assets_path,
                api_key=api_key,
                settings=runtime_settings,
                progress=update,
            )
            json_path, html_path = save_report(report, runtime_settings.output_dir)
        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.exception(exc)
        else:
            progress_bar.progress(1.0)
            status.success(f"Готово. Отчёт сохранён в {html_path.parent}")
            rows = [
                {
                    "ID": item.requirement.id,
                    "Категория": item.requirement.category,
                    "Требование": item.requirement.text,
                    "Статус": STATUS_LABELS[item.status.value],
                    "Объяснение": item.explanation,
                    "Уверенность": round(item.confidence, 2),
                }
                for item in report.comparisons
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            left, right = st.columns(2)
            left.download_button(
                "Скачать JSON",
                data=json_path.read_bytes(),
                file_name="report.json",
                mime="application/json",
                use_container_width=True,
            )
            right.download_button(
                "Скачать HTML",
                data=html_path.read_bytes(),
                file_name="report.html",
                mime="text/html",
                use_container_width=True,
            )
            if report.warnings:
                with st.expander(f"Предупреждения извлечения ({len(report.warnings)})"):
                    st.write("\n".join(f"- {warning}" for warning in report.warnings))
