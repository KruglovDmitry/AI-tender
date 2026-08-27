"""Клиент Vision-LLM (OpenAI-compatible, напр. vLLM Qwen2.5-VL)."""

from __future__ import annotations

import base64
from typing import Any

from ...providers import try_parse_llm_json


def _log_raw(label: str, raw: str) -> None:
    preview = (raw or "").replace("\r", "")[:500]
    print(f"[vl] {label}: len={len(raw or '')} raw={preview!r}", flush=True)


def complete_vl_json(
    *,
    image_bytes: bytes,
    prompt: str,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    image_mime: str = "image/jpeg",
    max_tokens: int = 2048,
    timeout_sec: float = 120.0,
    structure_hint: str = "той же структуры",
    repair: bool = True,
) -> tuple[dict[str, Any] | None, int]:
    """Мультимодальный chat.completions → JSON. → (data|None, n_calls)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "EMPTY",
        timeout=timeout_sec,
    )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{image_mime};base64,{b64}"
    # Картинка первой — стабильнее для Qwen2.5-VL.
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    raw = (response.choices[0].message.content or "") if response.choices else ""
    data = try_parse_llm_json(raw)
    if data is not None:
        return data, 1
    _log_raw("json_parse_failed", raw)
    if not repair:
        return None, 1

    repair_prompt = (
        "Предыдущий ответ был НЕВАЛИДНЫМ JSON. "
        f"Верни ТОЛЬКО исправленный валидный JSON-объект {structure_hint}. "
        "Без markdown, без комментариев.\n\n"
        f"ИСХОДНЫЙ ОТВЕТ:\n{raw[:8000]}"
    )
    repaired = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": repair_prompt}],
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    repaired_raw = (
        (repaired.choices[0].message.content or "") if repaired.choices else ""
    )
    data = try_parse_llm_json(repaired_raw)
    if data is None:
        _log_raw("repair_parse_failed", repaired_raw)
    return data, 2
