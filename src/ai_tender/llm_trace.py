"""Трассировка LLM-запросов и retrieval для отладки нестабильных прогонов."""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

_TRACE: ContextVar["LlmTrace | None"] = ContextVar("ai_tender_llm_trace", default=None)


class LlmTrace:
    def __init__(self, run_dir: Path, *, meta: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._jsonl = self.run_dir / "events.jsonl"
        self._meta_path = self.run_dir / "meta.json"
        payload = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            **(meta or {}),
        }
        self._meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def path(self) -> Path:
        return self.run_dir

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _append_event(self, event: dict[str, Any]) -> None:
        with self._jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def log_llm(
        self,
        stage: str,
        *,
        prompt: str,
        response: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        seq = self._next_seq()
        stem = f"{seq:03d}_{_safe_stage(stage)}"
        (self.run_dir / f"{stem}_request.txt").write_text(prompt, encoding="utf-8")
        (self.run_dir / f"{stem}_response.txt").write_text(response, encoding="utf-8")
        event = {
            "seq": seq,
            "kind": "llm",
            "stage": stage,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "request_file": f"{stem}_request.txt",
            "response_file": f"{stem}_response.txt",
            **(meta or {}),
        }
        self._append_event(event)

    def log_retrieval(
        self,
        stage: str,
        *,
        query: str,
        hits: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> None:
        seq = self._next_seq()
        stem = f"{seq:03d}_{_safe_stage(stage)}"
        payload = {
            "stage": stage,
            "query": query,
            "hits": hits,
            **(meta or {}),
        }
        (self.run_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        event = {
            "seq": seq,
            "kind": "retrieval",
            "stage": stage,
            "query_chars": len(query),
            "hits_count": len(hits),
            "file": f"{stem}.json",
            **(meta or {}),
        }
        self._append_event(event)

    def log_note(self, stage: str, message: str, meta: dict[str, Any] | None = None) -> None:
        seq = self._next_seq()
        event = {
            "seq": seq,
            "kind": "note",
            "stage": stage,
            "message": message,
            **(meta or {}),
        }
        self._append_event(event)
        (self.run_dir / f"{seq:03d}_{_safe_stage(stage)}_note.txt").write_text(
            message if not meta else f"{message}\n\n{json.dumps(meta, ensure_ascii=False, indent=2, default=str)}",
            encoding="utf-8",
        )

    def finish(self, extra: dict[str, Any] | None = None) -> None:
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        data["events"] = self._seq
        if extra:
            data.update(extra)
        self._meta_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _safe_stage(stage: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stage.strip())
    return (cleaned or "event")[:80]


def get_trace() -> LlmTrace | None:
    return _TRACE.get()


def start_trace(
    base_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> LlmTrace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{stamp}"
    trace = LlmTrace(run_dir, meta=meta)
    _TRACE.set(trace)
    return trace


def clear_trace() -> None:
    _TRACE.set(None)


def trace_llm(
    stage: str,
    *,
    prompt: str,
    response: str,
    meta: dict[str, Any] | None = None,
) -> None:
    active = get_trace()
    if active is not None:
        active.log_llm(stage, prompt=prompt, response=response, meta=meta)


def trace_retrieval(
    stage: str,
    *,
    query: str,
    hits: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    active = get_trace()
    if active is not None:
        active.log_retrieval(stage, query=query, hits=hits, meta=meta)


def trace_note(stage: str, message: str, meta: dict[str, Any] | None = None) -> None:
    active = get_trace()
    if active is not None:
        active.log_note(stage, message, meta=meta)
