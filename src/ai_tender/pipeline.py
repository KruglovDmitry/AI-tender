from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from .anchors import refine_requirement_anchors
from .index import (
    indexed_file_paths,
    load_or_build_assets_index,
    node_to_evidence,
    retrieve_for_queries,
)
from .doc_select import ranked_file_paths, select_tender_files
from .loaders import load_documents
from .models import AnalysisReport, ExtractedRequirement, Settings, get_settings
from .providers import assess_findings, build_compact_summary, build_llm, select_important_findings
from .query_select import (
    extract_tender_requirements_from_documents,
    product_match_succeeded,
    split_products_and_specs,
)

ProgressCallback = Callable[[str, float], None]


def _retrieve_candidates(
    requirements: list[ExtractedRequirement],
    assets_index,
    top_k: int,
) -> list[tuple[ExtractedRequirement, list]]:
    if not requirements:
        return []
    queries = [
        f"{item.text}\n{item.quote}".strip() if item.quote else item.text
        for item in requirements
    ]
    hit_lists = retrieve_for_queries(assets_index, queries, top_k=top_k)
    return [
        (req, hits)
        for req, hits in zip(requirements, hit_lists, strict=True)
        if hits
    ]


def analyze(
    tender_path: Path,
    assets_path: Path,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    settings = settings or get_settings()
    started = perf_counter()
    strategy = (settings.match_strategy or "hybrid").strip().lower()
    if strategy not in {"hybrid", "product", "specs"}:
        strategy = "hybrid"

    def update(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    update(
        "Индекс эталонов: чтение PDF и эмбеддинги (после смены файлов — полная пересборка, "
        "это долго; дальше будет из кэша)",
        0.08,
    )
    assets_index, asset_nodes, asset_warnings, index_reused = load_or_build_assets_index(
        assets_path,
        settings.cache_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_device,
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )
    indexed_files = indexed_file_paths(asset_nodes)
    update(
        (
            f"Индекс эталонов готов ({len(asset_nodes)} чанков, "
            f"{'из кэша' if index_reused else 'построен заново'})"
        ),
        0.3,
    )

    update("Каталог и выбор файлов тендера", 0.35)
    llm = build_llm(settings)
    use_extract = settings.llm_extract_requirements and settings.llm_select_queries
    use_doc_select = settings.llm_select_tender_files and use_extract

    tender_inventory = None
    doc_selection: dict = {}
    ranked_paths: list[str] = []

    try:
        if use_doc_select:
            tender_inventory, _catalog_entries, doc_selection = select_tender_files(
                tender_path,
                llm,
                use_llm=True,
                max_files=settings.max_tender_files_total,
            )
            ranked_paths = ranked_file_paths(doc_selection)
            update(
                (
                    f"Выбрано {len(ranked_paths)} из {doc_selection.get('catalog_count', 0)} "
                    f"файлов ({doc_selection.get('mode', 'llm')})"
                ),
                0.38,
            )
            update(f"Чтение выбранных файлов ({len(ranked_paths)})", 0.4)
            tender_documents, tender_warnings = load_documents(
                tender_path,
                corpus="tender",
                inventory=tender_inventory,
                only_labels=set(ranked_paths),
                ocr_enabled=settings.ocr_enabled,
                ocr_languages=settings.ocr_languages,
            )
        else:
            update("Чтение тендерной документации (эвристика)", 0.38)
            tender_documents, tender_warnings = load_documents(
                tender_path,
                corpus="tender",
                technical_only=True,
                ocr_enabled=settings.ocr_enabled,
                ocr_languages=settings.ocr_languages,
            )
            ranked_paths = sorted(
                {
                    str(doc.metadata.get("file_path") or doc.metadata.get("file_name") or "")
                    for doc in tender_documents
                }
                - {""}
            )

        update("Извлечение требований из документов (LLM)", 0.45)
        requirements, query_selection = extract_tender_requirements_from_documents(
            tender_documents,
            settings.max_tender_queries,
            llm=llm,
            use_llm=use_extract,
            max_chars_per_doc=settings.max_extract_chars_per_doc,
            file_order=ranked_paths or None,
            early_stop=settings.extract_early_stop and use_doc_select,
            early_stop_min_specs=settings.extract_early_stop_min_specs,
            early_stop_min_confidence=settings.extract_early_stop_min_confidence,
            early_stop_min_files=settings.extract_early_stop_min_files,
            max_files_to_process=(
                settings.max_tender_files_total if use_doc_select else None
            ),
        )
        if doc_selection:
            query_selection["doc_selection"] = {
                "mode": doc_selection.get("mode"),
                "catalog_count": doc_selection.get("catalog_count"),
                "selected": doc_selection.get("files"),
                "skipped": doc_selection.get("skip"),
                "loaded": ranked_paths,
            }
            if doc_selection.get("error"):
                query_selection["doc_selection"]["error"] = doc_selection["error"]
    finally:
        if tender_inventory is not None:
            tender_inventory.cleanup()
    requirements = refine_requirement_anchors(requirements, tender_path)
    products, specs = split_products_and_specs(requirements)
    query_selection["strategy"] = strategy
    query_selection["anchored"] = sum(
        1 for item in requirements if item.line_start is not None
    )
    query_selection["products"] = len(products)
    query_selection["specs"] = len(specs)
    query_selection["top_requirements"] = [
        {
            "priority": item.priority,
            "confidence": item.confidence,
            "kind": item.kind,
            "text": item.text[:160],
            "location": item.location,
            "line_start": item.line_start,
            "line_end": item.line_end,
        }
        for item in requirements[:5]
    ]

    all_findings = []
    match_mode = "specs_only"
    specs_to_check: list[ExtractedRequirement] = []

    if strategy == "product":
        match_mode = "product_only"
        targets = products or []
        update(f"Стратегия «продукт»: поиск {len(targets)} позиций", 0.65)
        if targets:
            product_candidates = _retrieve_candidates(
                targets, assets_index, top_k=max(settings.top_k, 5)
            )
            _, product_findings = assess_findings(
                llm,
                product_candidates,
                node_to_evidence,
                user_instruction=settings.user_instruction,
                max_findings=max(len(targets), settings.max_findings),
                mode="product",
                match_mode=match_mode,
                select_important=False,
            )
            all_findings.extend(product_findings)
        else:
            update("Артикулы не выделены — нечего проверять в режиме «продукт»", 0.7)

    elif strategy == "specs":
        match_mode = "specs_only"
        specs_to_check = specs or [item for item in requirements if item.kind != "product"]
        if not specs_to_check:
            specs_to_check = list(requirements)
        update(f"Стратегия «техсоответствие»: разбор ТТХ ({len(specs_to_check)})", 0.65)

    else:  # hybrid
        if products:
            update(
                f"Гибрид: поиск {len(products)} артикулов/названий в эталоне",
                0.6,
            )
            product_candidates = _retrieve_candidates(
                products, assets_index, top_k=max(settings.top_k, 5)
            )
            _, product_findings = assess_findings(
                llm,
                product_candidates,
                node_to_evidence,
                user_instruction=settings.user_instruction,
                max_findings=max(len(products), settings.max_findings),
                mode="product",
                match_mode="product_first",
                select_important=False,
            )
            all_findings.extend(product_findings)

            if product_match_succeeded(
                product_findings,
                min_confidence=settings.product_match_confidence,
            ):
                match_mode = "product_first"
                specs_to_check = sorted(
                    specs,
                    key=lambda item: (-item.priority, -item.confidence),
                )[: settings.max_specs_when_product_matched]
                update(
                    (
                        f"Артикул подтверждён — выборочная проверка ТТХ "
                        f"({len(specs_to_check)} из {len(specs)})"
                    ),
                    0.75,
                )
            else:
                match_mode = "specs_fallback"
                specs_to_check = specs
                update(
                    f"Артикул не подтверждён — полный разбор ТТХ ({len(specs_to_check)})",
                    0.75,
                )
        else:
            match_mode = "specs_only"
            specs_to_check = specs or requirements
            update(f"Артикул не выделен — разбор ТТХ ({len(specs_to_check)})", 0.65)

    if specs_to_check:
        update("Оценка ТТХ по обогащённым требованиям (LLM)", 0.85)
        specs_candidates = _retrieve_candidates(
            specs_to_check, assets_index, top_k=settings.top_k
        )
        _, specs_findings = assess_findings(
            llm,
            specs_candidates,
            node_to_evidence,
            user_instruction=settings.user_instruction,
            max_findings=settings.max_findings,
            mode="specs",
            match_mode=match_mode,
            select_important=False,
        )
        all_findings.extend(specs_findings)

    for item in all_findings:
        item.match_mode = match_mode

    findings = select_important_findings(all_findings, max_findings=settings.max_findings)
    summary = build_compact_summary(
        all_findings,
        findings,
        match_mode=match_mode,
        strategy=strategy,
    )
    query_selection["match_mode"] = match_mode
    query_selection["checked_specs"] = len(specs_to_check)

    truncated = query_selection.get("truncated_files") or []
    extra = f", обрезано файлов: {len(truncated)}" if truncated else ""
    update(
        (
            f"Готово: strategy=`{strategy}`, ход=`{match_mode}`, "
            f"product={len(products)}, specs_checked="
            f"{query_selection.get('checked_specs', 0)}, "
            f"якоря={query_selection.get('anchored', 0)}{extra}"
        ),
        0.95,
    )

    elapsed_seconds = round(perf_counter() - started, 2)
    return AnalysisReport(
        tender_path=str(tender_path.resolve()),
        assets_path=str(assets_path.resolve()),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        summary=summary,
        findings=findings,
        warnings=asset_warnings + tender_warnings,
        indexed_files=indexed_files,
        index_reused=index_reused,
        elapsed_seconds=elapsed_seconds,
        query_selection=query_selection,
        extracted_requirements=requirements,
    )
