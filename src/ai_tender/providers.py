import json
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from openai import OpenAI


class LLMProvider(Protocol):
    model: str

    def json_response(self, instructions: str, payload: str, schema: dict) -> dict: ...


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class DeepSeekProvider:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def json_response(self, instructions: str, payload: str, schema: dict) -> dict:
        schema_text = json.dumps(schema, ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{instructions}\nВерни только JSON без Markdown. "
                        f"Ответ обязан соответствовать JSON Schema:\n{schema_text}"
                    ),
                },
                {"role": "user", "content": payload},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek вернул пустой ответ")
        return _parse_json(content)


class LocalBGEProvider:
    def __init__(self, model: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self.encoder: Any = SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.encoder.encode(
            list(texts),
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def _parse_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Ответ модели должен быть JSON-объектом")
    return result
