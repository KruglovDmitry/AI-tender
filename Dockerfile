# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm install
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNNING_IN_DOCKER=1 \
    UNRAR_TOOL=/usr/bin/unrar-free \
    SEVEN_ZIP_CMD=/usr/bin/7z

RUN apt-get update && apt-get install -y --no-install-recommends \
    p7zip-full \
    unrar-free \
    antiword \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY --from=ui-build /ui/dist ./ui/dist

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["uvicorn", "ai_tender.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
