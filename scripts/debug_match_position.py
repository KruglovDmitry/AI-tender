"""Отладка match: retrieval, промпт и ответ LLM для одной позиции перечня.

Пример:
  py scripts/debug_match_position.py
  py scripts/debug_match_position.py --position "MOXA NPORT 5150" --candidate 0
  py scripts/debug_match_position.py --prompt-only --output data/debug_match.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ai_tender.models import get_settings
from ai_tender.nodes.match import (
    MATCH_LLM_EVAL_MAX,
    POSITION_MATCH_PROMPT_HINT,
    _parse_candidate_evaluation,
    build_match_candidate_prompt,
    retrieve_hits_for_position,
)
from ai_tender.providers import build_llm, complete_llm_json, try_parse_llm_json
from ai_tender.services.catalog_retrieval import catalog_hit_to_evidence, load_vl_catalog


def _divider(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Отладка match для позиции перечня")
    parser.add_argument(
        "--position",
        default="MOXA NPORT 5150",
        help="Название позиции (как в перечне закупки)",
    )
    parser.add_argument("--qty", type=float, default=24)
    parser.add_argument("--unit", default="шт.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help=f"Сколько топ-кандидатов из retrieval оценить (макс. {MATCH_LLM_EVAL_MAX})",
    )
    parser.add_argument(
        "--candidate",
        type=int,
        help="Оценить только одного кандидата по индексу (0 = top-1 retrieval)",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Только показать промпт, без вызова LLM",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Сохранить полный отчёт в JSON",
    )
    args = parser.parse_args()

    settings = get_settings()
    scope_item = {"name": args.position, "qty": args.qty, "unit": args.unit}
    requirements: list = []

    catalog, warnings = load_vl_catalog(
        settings.cache_dir,
        embedding_model=settings.embedding_model,
        device=settings.embedding_device,
    )
    if warnings:
        for w in warnings:
            print(f"WARN: {w}", file=sys.stderr)
    if catalog.size == 0:
        print("VL-каталог пуст. Переиндексируйте эталоны.", file=sys.stderr)
        return 1

    hits = retrieve_hits_for_position(
        args.position,
        requirements,
        catalog,
        top_k=max(settings.top_k, 5),
        embedding_model=settings.embedding_model,
        device=settings.embedding_device,
    )
    evidence = [catalog_hit_to_evidence(hit) for hit in hits]
    if not evidence:
        print("Retrieval не вернул кандидатов.", file=sys.stderr)
        return 1

    max_n = min(max(1, args.max_candidates), MATCH_LLM_EVAL_MAX, len(evidence))
    if args.candidate is not None:
        if args.candidate < 0 or args.candidate >= len(evidence):
            print(
                f"Индекс кандидата {args.candidate} вне диапазона 0..{len(evidence) - 1}",
                file=sys.stderr,
            )
            return 1
        indices = [args.candidate]
    else:
        indices = list(range(max_n))

    llm = None if args.prompt_only else build_llm(settings)

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "local_llm_base_url": settings.local_llm_base_url,
        "position": scope_item,
        "requirements": [],
        "schema_hint": POSITION_MATCH_PROMPT_HINT,
        "retrieval": [
            {
                "index": i,
                "score": ev.score,
                "file": ev.file,
                "location": ev.location,
                "quote_preview": (ev.quote or "")[:400],
            }
            for i, ev in enumerate(evidence[: max_n + 2])
        ],
        "candidates": [],
    }

    _divider(f"ПОЗИЦИЯ: {args.position}")
    print(f"Каталог: {catalog.size} продуктов, retrieval: {len(evidence)} хитов")
    print(f"Оцениваем кандидатов: {indices}")

    for index in indices:
        hit = evidence[index]
        prompt, payload = build_match_candidate_prompt(
            scope_item=scope_item,
            requirements=requirements,
            hit=hit,
            instruction=settings.user_instruction,
        )

        entry: dict = {
            "index": index,
            "retrieval_score": hit.score,
            "file": hit.file,
            "location": hit.location,
            "payload": payload,
            "prompt_chars": len(prompt),
            "prompt": prompt,
        }

        _divider(f"КАНДИДАТ #{index}  score={hit.score}  {hit.file} {hit.location}")
        print("\n--- PAYLOAD (JSON в блоке ДАННЫЕ) ---\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\n--- PROMPT ({0} символов) ---\n".format(len(prompt)))
        print(prompt)

        if args.prompt_only:
            report["candidates"].append(entry)
            continue

        assert llm is not None
        print("\n--- LLM: первый ответ (raw) ---\n")
        try:
            raw_response = str(llm.complete(prompt)).strip()
        except Exception as exc:
            raw_response = ""
            entry["llm_error"] = str(exc)
            print(f"ERROR: {exc}")
        else:
            print(raw_response or "(пусто)")

        entry["llm_raw"] = raw_response
        entry["llm_raw_parse"] = try_parse_llm_json(raw_response) if raw_response else None

        print("\n--- LLM: complete_llm_json (как в пайплайне, с repair) ---\n")
        try:
            parsed_data, n_calls = complete_llm_json(
                llm,
                prompt,
                structure_hint=POSITION_MATCH_PROMPT_HINT,
                trace_name=f"debug_match_{index}",
                repair_context=prompt,
            )
        except Exception as exc:
            parsed_data, n_calls = None, 0
            entry["pipeline_error"] = str(exc)
            print(f"ERROR: {exc}")
        else:
            print(json.dumps(parsed_data, ensure_ascii=False, indent=2))
            print(f"\n(вызовов LLM: {n_calls})")

        entry["llm_parsed"] = parsed_data
        entry["llm_calls"] = n_calls

        eval_result = _parse_candidate_evaluation(parsed_data, hit) if parsed_data else None
        if eval_result:
            entry["match_eval"] = {
                "fits": eval_result.fits,
                "status": eval_result.status.value,
                "product_name": eval_result.product_name,
                "explanation": eval_result.explanation,
                "confidence": eval_result.confidence,
            }
            print("\n--- Итог после _parse_candidate_evaluation ---\n")
            print(json.dumps(entry["match_eval"], ensure_ascii=False, indent=2))

        report["candidates"].append(entry)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nОтчёт сохранён: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
