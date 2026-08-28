# AI Tender

LlamaIndex RAG для оценки применимости эталонной продукции к обобщённым
требованиям тендерной документации (hybrid retrieval + LLM).

## Как работает

1. Документы читаются через LlamaIndex `SimpleDirectoryReader` (PDF, DOCX, DOC, XLSX, CSV, TXT, MD); ZIP/RAR распаковываются перед чтением.
2. Строится векторный индекс **эталонов** (`BAAI/bge-m3`, локально) и кэшируется на диске.
3. Тендер обрабатывается через **LangGraph** (scope-first): сначала предмет закупки (перечень), затем требования к пунктам, с дочитыванием доп. файлов при необходимости.
4. По эталону идёт **hybrid retrieval** (vector + BM25): «это требование ↔ какие фрагменты эталона».
5. Пары отдаются в LLM с пользовательской инструкцией; в UI показываются только
   наиболее значимые совпадения и короткое резюме по счётчикам статусов.
6. PDF без текстового слоя: при включённом OCR распознаются через Tesseract (медленнее).
7. RAR: нужен WinRAR (UnRAR) или 7-Zip; путь можно задать через `UNRAR_TOOL`.

## Запуск в Docker (Windows, одной кнопкой)

Требуется [Docker Desktop](https://www.docker.com/products/docker-desktop/).

1. Заполните `.env` (или он создастся из `.env.example` при первом запуске).
2. Положите документы в `sources/` и `assets/` рядом с проектом.
3. Дважды щёлкните **`start.bat`** — Docker + **нативное окно** (WebView2).
4. Без Docker: **`start-native.bat`** — локальный API + UI в том же окне.
5. Остановка Docker: **`stop.bat`**.

Пути внутри контейнера:
- тендер: `/data/tender` (папка `sources` на хосте)
- эталоны: `/data/assets`

Кэш эмбеддингов и индекса: `data/` на хосте.

```powershell
# Docker + нативное окно:
docker compose up -d --build
pip install "pywebview>=5.0"
python scripts/native_window.py

# Локально без Docker:
pip install -e ".[native]"
python scripts/native_window.py --serve
```

UI: http://localhost:8000 (React + FastAPI).

## Запуск локально (без Docker)

Требуется Python 3.12 и Node.js 20+.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
cd ui && npm install && npm run build && cd ..
ai-tender-api
# или для разработки UI:
# терминал 1: uvicorn ai_tender.api.main:app --reload --app-dir src
# терминал 2: cd ui && npm run dev
```

Кэш индекса эталонов: `data/cache/llama_assets/`.

## Веб на стенде (доступ из браузера)

Поднимите приложение на машине стенда (Docker или native) так, чтобы порт `8000`
был доступен в LAN/VPN. Эталоны лежат в `assets/` на сервере.

В UI:

1. **Эталоны** — просмотр VL-индекса, загрузка и удаление PDF.
2. **Анализ тендера** — загрузка файлов из браузера или папка на сервере.
3. После анализа — **скачать отчёт** (`.md` / `.json`).

Загрузки тендера сохраняются в `data/uploads/` и периодически очищаются.
Не публикуйте `8000` в открытый интернет без VPN/пароля.

## Переменные окружения

См. `.env.example`. Обычно достаточно:

- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`
- `AI_TENDER_LLM_PROVIDER` / `AI_TENDER_LLM_MODEL`
- `AI_TENDER_MAX_TENDER_FILES_INITIAL` / `AI_TENDER_MAX_TENDER_FILES_TOTAL`
- `AI_TENDER_MAX_REQS_PER_SCOPE_ITEM`
- `AI_TENDER_OCR_ENABLED` / `AI_TENDER_OCR_LANGUAGES`
- при необходимости: `UNRAR_TOOL`, `TESSERACT_CMD`

Остальные параметры (эмбеддинги, chunk size, top-k, cache) имеют значения по умолчанию в коде.

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
- Legacy `.doc` читается через `sharepoint-to-text` (локально) или `antiword` (Docker/Linux).
- Фрагменты уходят в выбранный LLM API — учитывайте политику данных.
