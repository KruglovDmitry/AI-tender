"""REST-маршруты."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ai_tender.graph import analyze
from ai_tender.models import AnalysisReport, get_settings
from ai_tender.services.index_service import (
    get_assets_index_status,
    remove_asset_from_index,
    scan_assets_files,
)
from ai_tender.services.indexing import index_asset_files
from ai_tender.services.indexing.persistance import (
    catalog_is_indexed,
    load_product_index,
)
from ai_tender.services.ocr_service import ocr_status
from ai_tender.services.report_export import report_to_json_bytes, report_to_markdown
from ai_tender.services.upload_service import (
    append_uploaded_files,
    new_run_dir,
    prepare_upload_dir,
)

from .deps import (
    default_assets_path,
    default_tender_path,
    is_running_in_docker,
    resolve_assets_path,
)
from .jobs import job_manager

router = APIRouter(prefix="/api")


class DeleteAssetRequest(BaseModel):
    path: str


class ReindexAssetRequest(BaseModel):
    path: str


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None


def _job_to_response(job) -> JobResponse:
    return JobResponse(
        id=job.id,
        kind=job.kind,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
    )


def _indexing_job_extra(
    job_id: str, *, progress_start: float, progress_span: float
) -> dict[str, Any]:
    progress = job_manager.make_progress(job_id)

    def on_index_progress(phase: str, page: int, total: int, detail: str) -> None:
        if phase == "start":
            pct = progress_start
        elif phase == "persist":
            pct = progress_start + progress_span * 0.92
        elif phase == "done":
            pct = min(progress_start + progress_span, 0.99)
        elif phase == "scan":
            pct = progress_start + progress_span * 0.45 * (page / max(total, 1))
        else:
            pct = progress_start + progress_span * (
                0.45 + 0.45 * (page / max(total, 1))
            )
        progress(detail, min(max(pct, progress_start), 0.99))

    return {"on_index_progress": on_index_progress}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
def get_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "ocr_enabled": settings.ocr_enabled,
        "max_reqs_per_scope_item": settings.max_reqs_per_scope_item,
        "embedding_model": settings.embedding_model,
        "default_tender_path": default_tender_path(),
        "default_assets_path": default_assets_path(),
        "running_in_docker": is_running_in_docker(),
    }


@router.get("/ocr-status")
def api_ocr_status() -> dict[str, Any]:
    ok, hint = ocr_status()
    return {"ok": ok, "hint": hint}


@router.get("/assets")
def list_assets(assets_path: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    root = resolve_assets_path(assets_path)
    try:
        status = get_assets_index_status(
            root,
            settings.cache_dir,
            settings.embedding_model,
            settings.chunk_size,
            settings.chunk_overlap,
            ocr_enabled=settings.ocr_enabled,
            ocr_languages=settings.ocr_languages,
        )
    except Exception:
        status = {"files": [], "warnings": []}

    disk_files = sorted(scan_assets_files(root))
    items: list[dict[str, Any]] = []
    for rel in disk_files:
        product_index = load_product_index(settings.cache_dir, rel)
        indexed = catalog_is_indexed(product_index)
        items.append(
            {
                "path": rel,
                "product_count": len(product_index.products) if product_index else 0,
                "indexed": indexed,
                "doc_kind": product_index.doc_kind.value if product_index else None,
                "catalog_name": product_index.catalog_name if product_index else "",
            }
        )

    return {
        "assets_path": str(root),
        "files": items,
        "index_status": status,
    }


@router.get("/assets/products")
def get_asset_products(path: str, assets_path: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    root = resolve_assets_path(assets_path)
    rel = path.replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise HTTPException(status_code=400, detail="Некорректный путь")

    product_index = load_product_index(settings.cache_dir, rel)
    if product_index is None:
        raise HTTPException(status_code=404, detail="Индекс продуктов не найден")

    return product_index.model_dump(mode="json")


@router.post("/assets/upload")
async def upload_assets(
    files: Annotated[list[UploadFile], File(...)],
    assets_path: str | None = Form(None),
) -> JobResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Нет файлов")

    settings = get_settings()
    root = resolve_assets_path(assets_path)
    root.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, bytes]] = []
    for item in files:
        content = await item.read()
        payloads.append((item.filename or "file.pdf", content))

    job = job_manager.create("assets_upload")

    def task() -> dict[str, Any]:
        progress = job_manager.make_progress(job.id)

        class _Upload:
            def __init__(self, name: str, data: bytes) -> None:
                self.name = name
                self._data = data

            def getvalue(self) -> bytes:
                return self._data

        uploads = [_Upload(name, data) for name, data in payloads]
        progress("Сохранение на диск…", 0.15)
        _, _, changed = append_uploaded_files(uploads, root)
        progress("VL-индексация по страницам…", 0.35)
        results, _ = index_asset_files(
            root,
            changed,
            cache_dir=settings.cache_dir,
            settings=settings,
            extra=_indexing_job_extra(job.id, progress_start=0.35, progress_span=0.6),
        )
        indexed_n = sum(1 for r in results if r.status.value == "indexed")
        skipped_n = sum(1 for r in results if r.status.value == "skipped")
        failed_n = sum(1 for r in results if r.status.value == "failed")
        for result in results:
            if result.status.value == "failed" and result.message:
                print(f"[assets index] {result.message}", flush=True)
            for warning in result.details.get("warnings") or []:
                print(f"[assets index] {result.relative_path}: {warning}", flush=True)
        progress("Готово", 1.0)
        return {
            "processed": len(changed),
            "indexed": indexed_n,
            "skipped": skipped_n,
            "failed": failed_n,
        }

    job_manager.run_in_background(job.id, task)
    return _job_to_response(job)


@router.post("/assets/delete")
def delete_asset(body: DeleteAssetRequest, assets_path: str | None = None) -> dict[str, str]:
    settings = get_settings()
    root = resolve_assets_path(assets_path)
    try:
        remove_asset_from_index(
            root,
            settings.cache_dir,
            body.path,
            settings.embedding_model,
            settings.chunk_size,
            settings.chunk_overlap,
            device=settings.embedding_device,
            ocr_enabled=settings.ocr_enabled,
            ocr_languages=settings.ocr_languages,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "deleted", "path": body.path}


@router.post("/assets/reindex")
def reindex_asset(body: ReindexAssetRequest, assets_path: str | None = None) -> JobResponse:
    settings = get_settings()
    root = resolve_assets_path(assets_path)
    rel = body.path.replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise HTTPException(status_code=400, detail="Некорректный путь эталона")

    target = (root / rel).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise HTTPException(status_code=400, detail="Путь вне каталога эталонов")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл эталона не найден")

    job = job_manager.create("assets_reindex")

    def task() -> dict[str, Any]:
        progress = job_manager.make_progress(job.id)
        progress(f"VL-индексация «{Path(rel).name}»…", 0.2)
        results, _ = index_asset_files(
            root,
            [rel],
            cache_dir=settings.cache_dir,
            settings=settings,
            extra=_indexing_job_extra(job.id, progress_start=0.2, progress_span=0.75),
        )
        indexed_n = sum(1 for r in results if r.status.value == "indexed")
        failed_n = sum(1 for r in results if r.status.value == "failed")
        for result in results:
            if result.status.value == "failed" and result.message:
                print(f"[assets reindex] {result.message}", flush=True)
            for warning in result.details.get("warnings") or []:
                print(f"[assets reindex] {result.relative_path}: {warning}", flush=True)
        progress("Готово", 1.0)
        return {
            "path": rel,
            "indexed": indexed_n,
            "failed": failed_n,
        }

    job_manager.run_in_background(job.id, task)
    return _job_to_response(job)


@router.post("/analyze")
async def start_analyze(
    llm_provider: Annotated[str, Form()] = "deepseek",
    ocr_enabled: Annotated[bool, Form()] = True,
    max_reqs_per_scope_item: Annotated[int, Form()] = 10,
    tender_source: Annotated[str, Form()] = "upload",
    tender_folder: Annotated[str, Form()] = "",
    assets_path: Annotated[str, Form()] = "",
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> JobResponse:
    settings = get_settings()
    assets_root = resolve_assets_path(assets_path or None)
    if not assets_root.is_dir():
        raise HTTPException(status_code=400, detail="Каталог эталонов не существует")

    provider = llm_provider.lower().strip()
    if provider == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise HTTPException(status_code=400, detail="Не задан DEEPSEEK_API_KEY в окружении")
    elif not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    if tender_source == "upload" and not files:
        raise HTTPException(status_code=400, detail="Загрузите файлы тендера")

    upload_payloads: list[tuple[str, bytes]] = []
    if files:
        for item in files:
            upload_payloads.append((item.filename or "file", await item.read()))

    job = job_manager.create("analyze")

    def task() -> dict[str, Any]:
        progress = job_manager.make_progress(job.id)
        runtime_settings = settings.model_copy(
            update={
                "llm_provider": provider,
                "ocr_enabled": ocr_enabled,
                "max_reqs_per_scope_item": max_reqs_per_scope_item,
            }
        )

        if tender_source == "upload":
            progress("Сохранение загруженного тендера…", 0.02)

            class _Upload:
                def __init__(self, name: str, data: bytes) -> None:
                    self.name = name
                    self._data = data

                def getvalue(self) -> bytes:
                    return self._data

            uploads = [_Upload(name, data) for name, data in upload_payloads]
            run_dir = new_run_dir("tender")
            tender_path, upload_warnings = prepare_upload_dir(
                uploads,
                run_dir / "tender",
            )
        else:
            tender_path = Path(tender_folder or default_tender_path()).expanduser()
            upload_warnings = []
            if not tender_path.is_dir():
                raise ValueError("Папка тендера не существует")

        report = analyze(
            tender_path,
            assets_root,
            settings=runtime_settings,
            progress=progress,
        )
        if upload_warnings:
            report.warnings = list(report.warnings or []) + upload_warnings
        return {
            "report": report.model_dump(mode="json"),
            "tender_root": str(tender_path.resolve()),
            "assets_root": str(assets_root.resolve()),
        }

    job_manager.run_in_background(job.id, task)
    return _job_to_response(job)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> JobResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return _job_to_response(job)


@router.post("/reports/download")
def download_report(report: dict[str, Any], format: str = "md") -> Response:
    try:
        model = AnalysisReport.model_validate(report)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if format == "json":
        data = report_to_json_bytes(model)
        return Response(
            content=data,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="ai-tender-report.json"'},
        )
    if format == "md":
        data = report_to_markdown(model).encode("utf-8")
        return Response(
            content=data,
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="ai-tender-report.md"'},
        )
    raise HTTPException(status_code=400, detail="format must be md or json")
