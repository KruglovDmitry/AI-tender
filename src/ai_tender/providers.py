"""LLM-провайдеры и разбор JSON-ответов."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from llama_index.core.llms import LLM

from .models import Settings


def _strip_llm_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = (
            text.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return text


def _repair_json_text(text: str) -> str:
    """Типичные починки ответов LLM: запятые, кавычки, управляющие символы."""
    repaired = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    # Управляющие символы внутри строк ломают json.loads.
    repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", repaired)
    # Пропущенные запятые между элементами: }{  ]{  }[  ][
    repaired = re.sub(r"([}\]])(\s*)([{\[])", r"\1,\2\3", repaired)
    # Висячие запятые перед } или ]
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def parse_llm_json(content: str) -> dict[str, Any]:
    """Разбор JSON-ответа LLM (fence, хвостовые запятые, битые control chars)."""
    text = _strip_llm_json_fence(content)
    candidates = [text, _repair_json_text(text)]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(data, dict):
            raise ValueError("Ответ LLM должен быть JSON-объектом")
        return data
    assert last_error is not None
    raise last_error


def try_parse_llm_json(content: str) -> dict[str, Any] | None:
    """Как parse_llm_json, но без исключения при ошибке."""
    try:
        return parse_llm_json(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def complete_llm_json(
    llm: LLM,
    prompt: str,
    *,
    structure_hint: str = "той же структуры",
    trace_name: str | None = None,
    repair: bool = True,
) -> tuple[dict[str, Any] | None, int]:
    """
    LLM complete → разбор JSON; при битом ответе — один repair-запрос.

    Возвращает (data|None, число complete-вызовов).
    """
    from .services.logging_service import trace_llm

    response = llm.complete(prompt)
    raw = str(response)
    if trace_name:
        trace_llm(trace_name, prompt=prompt, response=raw, meta={"phase": "extract"})
    data = try_parse_llm_json(raw)
    if data is not None or not repair:
        return data, 1

    repair_prompt = (
        "Предыдущий ответ был НЕВАЛИДНЫМ JSON. "
        f"Верни ТОЛЬКО исправленный валидный JSON-объект {structure_hint}. "
        "Без markdown, без комментариев. Экранируй кавычки в строках.\n\n"
        f"ИСХОДНЫЙ ОТВЕТ:\n{raw[:12000]}"
    )
    repaired = llm.complete(repair_prompt)
    repaired_raw = str(repaired)
    if trace_name:
        trace_llm(
            f"{trace_name}_repair",
            prompt=repair_prompt,
            response=repaired_raw,
            meta={"phase": "repair"},
        )
    return try_parse_llm_json(repaired_raw), 2


def build_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Не указан OPENAI_API_KEY")
        return OpenAI(
            model=settings.llm_model,
            api_key=api_key,
            api_base=settings.openai_base_url,
            temperature=0,
        )

    from llama_index.llms.openai_like import OpenAILike

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Не указан DEEPSEEK_API_KEY")

    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.deepseek_base_url,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0,
    )
