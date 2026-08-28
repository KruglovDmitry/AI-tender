"""Клиент Vision-LLM (OpenAI-compatible, напр. vLLM Qwen2.5-VL)."""

from __future__ import annotations

import base64
import time
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
    log_context: str = "",
) -> tuple[dict[str, Any] | None, int]:
    """Мультimодальный chat.completions → JSON. → (data|None, n_calls)."""
    from openai import OpenAI

    tag = log_context or "call"
    print(
        f"[vl] {tag} → POST {base_url.rstrip('/')}/chat/completions "
        f"model={model!r} image={len(image_bytes)}B mime={image_mime} "
        f"prompt={len(prompt)}ch max_tokens={max_tokens} timeout={timeout_sec}s",
        flush=True,
    )
    prompt_preview = prompt.replace("\n", " ")[:160]
    print(f"[vl] {tag} prompt≈ {prompt_preview!r}", flush=True)

    client = OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "EMPTY",
        timeout=timeout_sec,
    )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{image_mime};base64,{b64}"
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(
            f"[vl] {tag} ← ERROR after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - t0
    choice = response.choices[0] if response.choices else None
    raw = (choice.message.content or "") if choice else ""
    finish = choice.finish_reason if choice else "?"
    data = try_parse_llm_json(raw)
    print(
        f"[vl] {tag} ← {elapsed:.1f}s finish={finish} raw={len(raw)}ch "
        f"parsed={'ok' if data is not None else 'fail'}",
        flush=True,
    )
    if data is not None:
        return data, 1
    _log_raw(f"{tag} json_parse_failed", raw)
    if not repair:
        return None, 1

    repair_prompt = (
        "Предыдущий ответ был НЕВАЛИДНЫМ JSON. "
        f"Верни ТОЛЬКО исправленный валидный JSON-объект {structure_hint}. "
        "Без markdown, без комментариев.\n\n"
        f"ИСХОДНЫЙ ОТВЕТ:\n{raw[:8000]}"
    )
    print(f"[vl] {tag} → repair (text-only) max_tokens={max_tokens}", flush=True)
    t1 = time.perf_counter()
    repaired = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": repair_prompt}],
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    repair_elapsed = time.perf_counter() - t1
    repaired_raw = (
        (repaired.choices[0].message.content or "") if repaired.choices else ""
    )
    data = try_parse_llm_json(repaired_raw)
    print(
        f"[vl] {tag} ← repair {repair_elapsed:.1f}s raw={len(repaired_raw)}ch "
        f"parsed={'ok' if data is not None else 'fail'}",
        flush=True,
    )
    if data is None:
        _log_raw(f"{tag} repair_parse_failed", repaired_raw)
    return data, 2
