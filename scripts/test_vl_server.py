"""Диагностика vLLM / Qwen2.5-VL: что модель реально возвращает.

Примеры:
  python scripts/test_vl_server.py
  python scripts/test_vl_server.py --pdf assets/SR33-2024-5.pdf --page 3
  python scripts/test_vl_server.py --scale 1.0 --max-tokens 512
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import fitz  # noqa: E402
from openai import OpenAI  # noqa: E402

from ai_tender.models import get_settings  # noqa: E402
from ai_tender.providers import try_parse_llm_json  # noqa: E402


PROBES: list[tuple[str, str]] = [
    (
        "plain_read_en",
        "Read all text visible on the image. Reply with the exact text only.",
    ),
    (
        "list_models_en",
        "List every product model / article code visible on the image, one per line.",
    ),
    (
        "json_simple",
        'Return ONLY a JSON object, no markdown fences:\n'
        '{"catalog_name":"","products":[{"model":"","canonical_desc":""}]}',
    ),
    (
        "json_anti_fence",
        "Extract products from the image. "
        "Output a raw JSON object starting with { and ending with }. "
        "Do not use markdown. Do not write ```.",
    ),
]


def _jpeg_from_pdf(path: Path, page: int, scale: float) -> bytes:
    doc = fitz.open(path)
    try:
        idx = max(0, min(page - 1, doc.page_count - 1))
        pix = doc.load_page(idx).get_pixmap(
            matrix=fitz.Matrix(scale, scale), alpha=False
        )
        return pix.tobytes("jpeg")
    finally:
        doc.close()


def _jpeg_synthetic(scale: float) -> bytes:
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_textbox(
            page.rect + (36, 36, -36, -36),
            "Catalog SR33 Online UPS\n"
            "Model: UPS-1000  Power: 1000 VA  Voltage: 220 V\n"
            "Model: UPS-2000  Power: 2000 VA  Voltage: 220 V\n"
            "Manufacturer: ACME\n",
            fontsize=16,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pix.tobytes("jpeg")
    finally:
        doc.close()


def _call(
    client: OpenAI,
    *,
    model: str,
    image_jpeg: bytes,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str, float]:
    b64 = base64.b64encode(image_jpeg).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    choice = resp.choices[0] if resp.choices else None
    raw = (choice.message.content or "") if choice else ""
    finish = choice.finish_reason if choice else "?"
    return raw, finish, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка ответа VL/vLLM")
    parser.add_argument("--pdf", type=Path, default=None, help="PDF для страницы")
    parser.add_argument("--page", type=int, default=1, help="Номер страницы (1-based)")
    parser.add_argument("--scale", type=float, default=1.2, help="Масштаб рендера PDF")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/cache/_vl_server_test.txt"),
        help="Куда сохранить полный лог",
    )
    args = parser.parse_args()

    settings = get_settings()
    base_url = settings.vl_base_url
    model = settings.vl_model
    timeout = settings.vl_timeout_sec

    print(f"base_url     = {base_url}")
    print(f"model        = {model}")
    print(f"timeout_sec  = {timeout}")
    print(f"temperature  = {args.temperature}")
    print(f"top_p        = {args.top_p}")
    print(f"max_tokens   = {args.max_tokens}")

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)

    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        print(f"models.list  = {ids}")
    except Exception as exc:
        print(f"models.list FAILED: {type(exc).__name__}: {exc}")
        return 1

    if args.pdf is not None:
        pdf = args.pdf.expanduser().resolve()
        if not pdf.is_file():
            print(f"PDF not found: {pdf}")
            return 1
        image = _jpeg_from_pdf(pdf, args.page, args.scale)
        source = f"{pdf.name} page={args.page}"
    else:
        image = _jpeg_synthetic(args.scale)
        source = "synthetic page (UPS-1000 / UPS-2000)"

    print(f"image        = {source}, jpeg_bytes={len(image)}, scale={args.scale}")
    print()

    lines: list[str] = [
        f"base_url={base_url}",
        f"model={model}",
        f"source={source}",
        f"jpeg_bytes={len(image)}",
        f"temperature={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}",
        "",
    ]

    for name, prompt in PROBES:
        print("=" * 72)
        print(f"PROBE: {name}")
        print(f"PROMPT: {prompt[:120]}{'…' if len(prompt) > 120 else ''}")
        try:
            raw, finish, elapsed = _call(
                client,
                model=model,
                image_jpeg=image,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as exc:
            msg = f"ERROR {type(exc).__name__}: {exc}"
            print(msg)
            lines.extend([f"=== {name}", msg, ""])
            continue

        parsed = try_parse_llm_json(raw)
        print(f"finish={finish}  elapsed={elapsed:.1f}s  len={len(raw)}")
        print(f"repr : {raw!r}")
        print("text :")
        print(raw if raw else "<empty>")
        if name.startswith("json"):
            print(f"parsed_json = {parsed}")

        lines.extend(
            [
                f"=== {name}",
                f"finish={finish} elapsed={elapsed:.1f}s len={len(raw)}",
                f"repr={raw!r}",
                "text:",
                raw if raw else "<empty>",
                f"parsed_json={parsed}",
                "",
            ]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("=" * 72)
    print(f"лог сохранён: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
