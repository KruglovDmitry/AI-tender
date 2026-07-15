from pathlib import Path

import typer

from .config import get_settings
from .pipeline import analyze
from .reporting import save_report

app = typer.Typer(help="Сопоставление технических требований тендера с эталонами.")


@app.command()
def run(
    tender: Path = typer.Option(..., exists=True, file_okay=False, help="Папка тендера"),
    assets: Path = typer.Option(..., exists=True, file_okay=False, help="Папка эталонов"),
) -> None:
    settings = get_settings()

    def progress(message: str, _: float) -> None:
        typer.echo(message)

    report = analyze(tender, assets, settings=settings, progress=progress)
    json_path, html_path = save_report(report, settings.output_dir)
    typer.echo(f"JSON: {json_path}")
    typer.echo(f"HTML: {html_path}")


if __name__ == "__main__":
    app()
