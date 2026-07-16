# AI Tender

LlamaIndex RAG для поиска вхождений эталонных технических средств в тендерную
документацию с последующей оценкой соответствия через LLM.

## Как работает

1. Документы читаются через LlamaIndex `SimpleDirectoryReader` (PDF, DOCX, XLSX, CSV, TXT, MD); ZIP/RAR распаковываются перед чтением.
2. Строится векторный индекс эталонов (`BAAI/bge-m3`, локально) и кэшируется на диске.
3. Индекс тендера строится на каждый запуск.
4. Чанки эталона используются как запросы; по тендеру идёт **hybrid retrieval**
   (vector + BM25).
5. Найденные пары отдаются в LLM (`DeepSeek` через OpenAILike или `OpenAI`) с вопросом
   оценить вхождения и соответствие строго по цитатам.

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

UI запускается через `streamlit run app.py --server.fileWatcherType none`.
Кэш индекса эталонов: `data/cache/llama_assets/`.

## Переменные окружения

См. `.env.example`. Ключевые:

- `AI_TENDER_LLM_PROVIDER=deepseek|openai`
- `AI_TENDER_LLM_MODEL`
- `AI_TENDER_EMBEDDING_MODEL=BAAI/bge-m3`
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`

## Ограничения

- Высокий retrieval score ≠ юридическое соответствие; финальную оценку даёт LLM по цитатам.
- PDF без текста требуют OCR (пока нет).
- Legacy `.doc` может читаться нестабильно.
- Фрагменты уходят в выбранный LLM API — учитывайте политику данных.
