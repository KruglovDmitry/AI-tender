import os
from collections.abc import Callable
from pathlib import Path

from .config import Settings, get_settings
from .ingestion import extract_folder
from .llm import compare_requirement, extract_requirements
from .models import AnalysisReport
from .providers import DeepSeekProvider, LocalBGEProvider


ProgressCallback = Callable[[str, float], None]


def analyze(
    tender_path: Path,
    assets_path: Path,
    api_key: str | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    settings = settings or get_settings()
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Не указан DEEPSEEK_API_KEY")
    llm = DeepSeekProvider(api_key, settings.model, settings.deepseek_base_url)

    def update(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    update("Чтение закупочной документации", 0.05)
    tender_blocks, tender_warnings = extract_folder(tender_path, technical_only=True)
    update("Чтение эталонной документации", 0.15)
    asset_blocks, asset_warnings = extract_folder(assets_path)
    if not tender_blocks:
        raise ValueError("Не удалось извлечь текст из тендерной документации")
    if not asset_blocks:
        raise ValueError("Не удалось извлечь текст из эталонной документации")

    update("Извлечение атомарных требований", 0.25)
    requirements = extract_requirements(
        llm,
        tender_blocks,
        settings.max_requirements,
        lambda message: update(message, 0.3),
    )
    if not requirements:
        raise ValueError("Модель не нашла технических требований")

    update("Загрузка локальной embedding-модели и построение индекса", 0.4)
    embeddings = LocalBGEProvider(settings.embedding_model, settings.embedding_device)
    asset_vectors = embeddings.embed([block.text for block in asset_blocks])
    requirement_vectors = embeddings.embed([requirement.text for requirement in requirements])

    comparisons = []
    for index, requirement in enumerate(requirements):
        score = asset_vectors @ requirement_vectors[index]
        candidate_indices = score.argsort()[-settings.top_k :][::-1]
        candidates = [asset_blocks[int(item)] for item in candidate_indices]
        progress_value = 0.5 + (0.45 * (index + 1) / len(requirements))
        update(
            f"Сопоставление {index + 1} из {len(requirements)}: {requirement.category}",
            progress_value,
        )
        comparisons.append(compare_requirement(llm, requirement, candidates))

    update("Подготовка отчёта", 0.98)
    return AnalysisReport(
        tender_path=str(tender_path.resolve()),
        assets_path=str(assets_path.resolve()),
        model=settings.model,
        comparisons=comparisons,
        warnings=tender_warnings + asset_warnings,
    )
