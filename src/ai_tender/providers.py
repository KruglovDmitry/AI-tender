"""LLM-провайдеры и разбор JSON-ответов."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from llama_index.core.llms import LLM

from .models import Settings, _openai_compatible_base_url, resolve_llm_provider


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


def _extract_balanced_object(text: str, start: int) -> tuple[str | None, int]:
    """Вырезает один JSON-объект {...} начиная с start. → (obj|None, end_index)."""
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    return None, start


def _salvage_partial_products_json(content: str) -> dict[str, Any] | None:
    """Достаёт полные элементы products из обрезанного JSON ответа VL."""
    text = content.strip()
    if text.startswith("```"):
        text = (
            text.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    catalog = ""
    catalog_match = re.search(
        r'"catalog_name"\s*:\s*"((?:\\.|[^"\\])*)"', text
    )
    if catalog_match:
        catalog = json.loads(f'"{catalog_match.group(1)}"')

    products_match = re.search(r'"products"\s*:\s*\[', text)
    if not products_match:
        return None

    products: list[Any] = []
    i = products_match.end()
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        if text[i] != "{":
            break
        obj_text, end = _extract_balanced_object(text, i)
        if obj_text is None:
            break
        try:
            item = json.loads(_repair_json_text(obj_text))
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            products.append(item)
        i = end

    if not products:
        return None
    return {"catalog_name": catalog, "products": products}


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
    salvaged = _salvage_partial_products_json(content)
    if salvaged is not None:
        return salvaged
    assert last_error is not None
    raise last_error


def try_parse_llm_json(content: str) -> dict[str, Any] | None:
    """Как parse_llm_json, но без исключения при ошибке."""
    try:
        return parse_llm_json(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        salvaged = _salvage_partial_products_json(content or "")
        if salvaged is not None:
            return salvaged
        return _salvage_match_json(content or "")


def _salvage_match_json(content: str) -> dict[str, Any] | None:
    """Достаёт поля match_position из обрезанного/битого JSON."""
    text = _strip_llm_json_fence(content or "")
    if not text.strip():
        return None
    result: dict[str, Any] = {}
    matched = re.search(r'"matched"\s*:\s*(true|false)', text, re.IGNORECASE)
    if matched:
        result["matched"] = matched.group(1).lower() == "true"
    fits = re.search(r'"fits"\s*:\s*(true|false)', text, re.IGNORECASE)
    if fits:
        result["fits"] = fits.group(1).lower() == "true"
        if "matched" not in result:
            result["matched"] = result["fits"]
    for key in ("status", "required_product", "product_name", "explanation"):
        field = re.search(
            rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"',
            text,
        )
        if field:
            try:
                result[key] = json.loads(f'"{field.group(1)}"')
            except json.JSONDecodeError:
                result[key] = field.group(1)
    conf = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if conf:
        try:
            result["confidence"] = float(conf.group(1))
        except ValueError:
            pass
    if "status" in result or "matched" in result:
        return result
    return None


def complete_llm_json(
    llm: LLM,
    prompt: str,
    *,
    structure_hint: str = "той же структуры",
    trace_name: str | None = None,
    repair: bool = True,
    repair_context: str | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """
    LLM complete → разбор JSON; при битом ответе — один repair-запрос.

    Возвращает (data|None, число complete-вызовов).
    """
    from .services.logging_service import trace_llm

    response = llm.complete(prompt)
    raw = str(response).strip()
    if trace_name:
        trace_llm(trace_name, prompt=prompt, response=raw, meta={"phase": "extract"})
    if not raw or raw in {"```", "```json", "```JSON"}:
        raw = ""
    data = try_parse_llm_json(raw) if raw else None
    if data is not None or not repair:
        return data, 1

    repair_prompt = (
        "Предыдущий ответ был пустым или невалидным JSON. "
        f"Верни ТОЛЬКО исправленный валидный JSON-объект {structure_hint}. "
        "Без markdown, без ```, без комментариев. Начни ответ с символа {.\n\n"
        f"ЗАПРОС:\n{(repair_context or prompt)[-8000:]}"
    )
    if raw:
        repair_prompt += f"\n\nИСХОДНЫЙ ОТВЕТ:\n{raw[:12000]}"
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
    provider = resolve_llm_provider(settings.llm_provider)

    if provider == "local":
        from llama_index.llms.openai_like import OpenAILike

        api_key = (
            os.getenv("LOCAL_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        return OpenAILike(
            model=settings.llm_model,
            api_base=_openai_compatible_base_url(settings.local_llm_base_url),
            api_key=api_key,
            is_chat_model=True,
            is_function_calling_model=False,
            temperature=0,
            timeout=settings.llm_timeout_sec,
            max_tokens=settings.llm_max_output_tokens,
        )

    if provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        return OpenAI(
            model=settings.llm_model,
            api_key=api_key,
            api_base=settings.openai_base_url,
            temperature=0,
            timeout=settings.llm_timeout_sec,
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
        timeout=settings.llm_timeout_sec,
    )
