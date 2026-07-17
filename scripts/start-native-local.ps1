# Local run without Docker: Streamlit + native window.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-ProjectPython {
    $venv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    return "python"
}

$python = Get-ProjectPython

foreach ($dir in @("sources", "assets", "data", "sources\1")) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - set DEEPSEEK_API_KEY or OPENAI_API_KEY" -ForegroundColor Yellow
}

Write-Host "Installing pywebview if needed..." -ForegroundColor Cyan
& $python -m pip install "pywebview>=5.0" -q

Write-Host "Starting AI Tender..." -ForegroundColor Cyan
& $python scripts\native_window.py --serve
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }