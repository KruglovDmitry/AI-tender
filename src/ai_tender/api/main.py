"""Точка входа FastAPI: API + статика React."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_tender.graph import warm_up_graph
from ai_tender.services.upload_service import cleanup_old_uploads

from .routes import router

UI_DIST = Path(__file__).resolve().parents[3] / "ui" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="AI Tender", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.on_event("startup")
    def _warm_up() -> None:
        cleanup_old_uploads()
        warm_up_graph()

    if UI_DIST.is_dir():
        assets_dir = UI_DIST / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api"):
                from fastapi import HTTPException

                raise HTTPException(status_code=404)
            candidate = UI_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(UI_DIST / "index.html")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("ai_tender.api.main:app", host="0.0.0.0", port=8000, reload=False)
