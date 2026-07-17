# AI Tender

LlamaIndex RAG для оценки применимости эталонной продукции к обобщённым
требованиям тендерной документации (hybrid retrieval + LLM).

## Как работает

1. Документы читаются через LlamaIndex `SimpleDirectoryReader` (PDF, DOCX, XLSX, CSV, TXT, MD); ZIP/RAR распаковываются перед чтением.
2. Строится векторный индекс **эталонов** (`BAAI/bge-m3`, локально) и кэшируется на диске.
3. Из тендера берутся требования (чанки) как запросы.
4. По эталону идёт **hybrid retrieval** (vector + BM25): «это требование ↔ какие фрагменты эталона».
5. Пары отдаются в LLM с пользовательской инструкцией; в UI показываются только
   наиболее значимые совпадения и короткое резюме по счётчикам статусов.
6. PDF без текстового слоя: при включённом OCR распознаются через Tesseract (медленнее).
7. RAR: нужен WinRAR (UnRAR) или 7-Zip; путь можно задать через `UNRAR_TOOL`.

## Запуск

Требуется Python 3.12.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
Copy-Item .env.example .env
# заполните DEEPSEEK_API_KEY или OPENAI_API_KEY
streamlit run app.py --server.fileWatcherType none
```

UI: `streamlit run app.py --server.fileWatcherType none`.
Кэш индекса эталонов: `data/cache/llama_assets/`.

## Переменные окружения

См. `.env.example`. Ключевые:

- `AI_TENDER_LLM_PROVIDER=deepseek|openai`
- `AI_TENDER_LLM_MODEL`
- `AI_TENDER_EMBEDDING_MODEL=BAAI/bge-m3`
- `AI_TENDER_MAX_TENDER_QUERIES` / `AI_TENDER_MAX_FINDINGS`
- `AI_TENDER_OCR_ENABLED` / `AI_TENDER_OCR_LANGUAGES`
- `UNRAR_TOOL` / `TESSERACT_CMD`
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`

### RAR и OCR (Windows)

```powershell
# RAR — обычно хватает установленного WinRAR; иначе:
$env:UNRAR_TOOL = "C:\Program Files\WinRAR\UnRAR.exe"

# OCR — установите Tesseract с языками rus+eng:
# https://github.com/UB-Mannheim/tesseract/wiki
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Ограничения

- Высокий retrieval score ≠ юридическое соответствие; финальную оценку даёт LLM по цитатам.
- PDF без текста: включите OCR и установите Tesseract (rus+eng).
- RAR без WinRAR/7-Zip не распакуется — см. `UNRAR_TOOL`.
- Legacy `.doc` может читаться нестабильно.
- Фрагменты уходят в выбранный LLM API — учитывайте политику данных.
