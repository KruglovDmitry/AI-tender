"""Провайдер local LLM и resolve_llm_provider."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from ai_tender.models import Settings, resolve_llm_provider, settings_for_llm_provider
from ai_tender.providers import build_llm


@pytest.mark.parametrize(
    ("raw", "has_local_url", "expected"),
    [
        ("deepseek", True, "deepseek"),
        ("local", True, "local"),
        ("local", False, "local"),
        ("openai", True, "local"),
        ("openai", False, "openai"),
    ],
)
def test_resolve_llm_provider(raw: str, has_local_url: bool, expected: str) -> None:
    env = {"LOCAL_LLM_BASE_URL": "http://10.0.0.1:8000"} if has_local_url else {}
    with patch.dict(os.environ, env, clear=False):
        if not has_local_url:
            os.environ.pop("LOCAL_LLM_BASE_URL", None)
        assert resolve_llm_provider(raw) == expected


def test_settings_for_local_caps_parallelism() -> None:
    base = Settings(requirements_parallelism=6, match_parallelism=4)
    env = {
        "LOCAL_LLM_BASE_URL": "http://10.0.0.1:8000",
        "LOCAL_LLM_DEFAULT_MODEL": "test-model",
    }
    with patch.dict(os.environ, env, clear=False):
        runtime = settings_for_llm_provider("local", base)
    assert runtime.llm_provider == "local"
    assert runtime.llm_model == "test-model"
    assert runtime.local_llm_base_url == "http://10.0.0.1:8000/v1"
    assert runtime.requirements_parallelism == 1
    assert runtime.match_parallelism == 1


def test_build_llm_local_openai_like() -> None:
    settings = Settings(
        llm_provider="local",
        llm_model="qwen-test",
        local_llm_base_url="http://127.0.0.1:8000/v1",
    )
    llm = build_llm(settings)
    assert llm.model == "qwen-test"
    assert llm.api_base == "http://127.0.0.1:8000/v1"
