from collections.abc import Callable, Sequence

from .models import Block, Comparison, Evidence, Requirement, Status
from .providers import LLMProvider


REQUIREMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {"type": "string"},
                    "parameter": {"type": ["string", "null"]},
                    "operator": {"type": ["string", "null"]},
                    "value": {"type": ["string", "number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "mandatory": {"type": "boolean"},
                    "source_block_id": {"type": "string"},
                },
                "required": [
                    "text",
                    "category",
                    "parameter",
                    "operator",
                    "value",
                    "unit",
                    "mandatory",
                    "source_block_id",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["requirements"],
    "additionalProperties": False,
}


COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [status.value for status in Status],
        },
        "explanation": {"type": "string"},
        "reference_value": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_block_ids": {"type": "array", "items": {"type": "string"}},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "status",
        "explanation",
        "reference_value",
        "confidence",
        "evidence_block_ids",
        "quotes",
    ],
    "additionalProperties": False,
}


def extract_requirements(
    provider: LLMProvider,
    blocks: Sequence[Block],
    max_requirements: int,
    progress: Callable[[str], None] | None = None,
) -> list[Requirement]:
    requirements: list[Requirement] = []
    batches: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in blocks:
        if current and size + len(block.text) > 30_000:
            batches.append(current)
            current, size = [], 0
        current.append(block)
        size += len(block.text)
    if current:
        batches.append(current)

    instructions = (
        "Ты технический аналитик закупок. Извлеки только атомарные требования к продукции, "
        "оборудованию, его функциям, параметрам, комплектности и подтверждающим документам. "
        "Игнорируй юридические, ценовые и процедурные условия. Не выдумывай требования. "
        "Каждый пункт должен проверять одно условие. Сохрани id блока-источника."
    )
    block_map = {block.id: block for block in blocks}
    for batch_number, batch in enumerate(batches, start=1):
        if len(requirements) >= max_requirements:
            break
        if progress:
            progress(f"Извлечение требований: пакет {batch_number} из {len(batches)}")
        payload = "\n\n".join(
            f"[BLOCK_ID={block.id}] [{block.file}; {block.location}]\n{block.text}"
            for block in batch
        )
        data = provider.json_response(instructions, payload, REQUIREMENTS_SCHEMA)
        for item in data["requirements"]:
            source = block_map.get(item["source_block_id"])
            if source is None:
                continue
            item["id"] = f"REQ-{len(requirements) + 1:04d}"
            item["source_file"] = source.file
            item["source_location"] = source.location
            requirements.append(Requirement.model_validate(item))
            if len(requirements) >= max_requirements:
                break
    return requirements


def compare_requirement(
    provider: LLMProvider,
    requirement: Requirement,
    candidates: Sequence[Block],
) -> Comparison:
    instructions = (
        "Сопоставь техническое требование только с предоставленными эталонными фрагментами. "
        "Для чисел учитывай оператор, диапазон и единицу измерения. compliant означает полное "
        "подтверждение, non_compliant — явное противоречие, partial — частичное выполнение, "
        "insufficient_evidence — доказательств недостаточно, not_applicable — требование не к "
        "рассматриваемой продукции. Не делай выводов из общих знаний. Цитируй дословно."
    )
    payload = (
        f"ТРЕБОВАНИЕ:\n{requirement.model_dump_json()}\n\nЭТАЛОНЫ:\n"
        + "\n\n".join(
            f"[BLOCK_ID={block.id}] [{block.file}; {block.location}]\n{block.text}"
            for block in candidates
        )
    )
    data = provider.json_response(instructions, payload, COMPARISON_SCHEMA)
    candidate_map = {block.id: block for block in candidates}
    evidence = []
    for index, block_id in enumerate(data["evidence_block_ids"]):
        block = candidate_map.get(block_id)
        if block:
            quote = data["quotes"][index] if index < len(data["quotes"]) else ""
            evidence.append(
                Evidence(
                    file=block.file,
                    location=block.location,
                    quote=quote,
                    block_id=block.id,
                )
            )
    return Comparison(
        requirement=requirement,
        status=data["status"],
        explanation=data["explanation"],
        reference_value=data["reference_value"],
        confidence=data["confidence"],
        evidence=evidence,
    )
