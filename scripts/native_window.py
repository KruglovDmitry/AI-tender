"""Нативное окно для React UI (WebView2 на Windows)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_server(url: str, timeout_sec: int = 120) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    return False


def start_api(root: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ai_tender.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(root),
        env={**os.environ.copy(), "PYTHONPATH": str(root / "src")},
    )


def open_native_window(url: str, title: str = "AI Tender") -> None:
    import webview

    window = webview.create_window(
        title,
        url,
        width=1320,
        height=900,
        min_size=(960, 640),
    )
    webview.start(gui="edgechromium")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Tender native window")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Запустить API локально (без Docker)",
    )
    parser.add_argument("--title", default="AI Tender")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    process: subprocess.Popen | None = None

    if args.serve:
        print("Запуск API…")
        process = start_api(root)

    print(f"Ожидание {args.url} …")
    if not wait_for_server(args.url):
        if process is not None:
            process.terminate()
        print("Сервер не ответил вовремя.", file=sys.stderr)
        return 1

    try:
        print("Открытие окна приложения...")
        open_native_window(args.url, title=args.title)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
