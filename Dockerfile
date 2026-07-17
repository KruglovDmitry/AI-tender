FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNNING_IN_DOCKER=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    UNRAR_TOOL=/usr/bin/unrar-free \
    SEVEN_ZIP_CMD=/usr/bin/7z \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/ \
    TESSERACT_CMD=/usr/bin/tesseract

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    p7zip-full \
    unrar-free \
    && rm -rf /var/lib/apt/lists/* \
    && tesseract --list-langs

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit

RUN pip install --upgrade pip && pip install .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.fileWatcherType", "none"]
