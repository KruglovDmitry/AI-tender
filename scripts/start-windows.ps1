# AI Tender: Docker + native window (WebView2).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Test-DockerReady {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not running. Start Docker Desktop and wait until status is Running."
    }
}

function Get-ProjectPython {
    $venv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    return "python"
}

Write-Host "AI Tender - starting container..." -ForegroundColor Cyan
Test-DockerReady

foreach ($dir in @("sources", "assets", "data", "sources\1")) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - set DEEPSEEK_API_KEY or OPENAI_API_KEY" -ForegroundColor Yellow
}

docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start docker compose"
}

$url = "http://localhost:8501"
$python = Get-ProjectPython

Write-Host "Installing pywebview if needed..." -ForegroundColor Cyan
& $python -m pip install "pywebview>=5.0" -q

Write-Host "Opening native window..." -ForegroundColor Cyan
& $python scripts\native_window.py --url $url
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not open window. Open manually: $url" -ForegroundColor Yellow
    Start-Process $url
}

Write-Host "Container is running: $url" -ForegroundColor Green
Write-Host "Stop with: stop.bat" -ForegroundColor DarkGray