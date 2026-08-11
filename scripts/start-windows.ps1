# AI Tender: Docker + native window (WebView2).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Test-DockerReady {
    # Docker Desktop часто ещё поднимается: CLI уже есть, engine отдаёт 500 на /info.
    $output = & docker info 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        return
    }
    $hint = @"
Docker Desktop не готов (docker info завершился с ошибкой).

Что сделать:
1. Откройте Docker Desktop и дождитесь статуса Running (зелёный).
2. Если висит Starting / Engine stopped — Restart Docker Desktop.
3. Проверьте в PowerShell: docker version   (должны быть и Client, и Server).
4. Ошибка '500 ... dockerDesktopLinuxEngine' / 'API version' = движок не поднялся
   или CLI новее Desktop: обновите Docker Desktop или переустановите.

Детали:
$output
"@
    throw $hint.Trim()
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

# --build ходит в Docker Hub за base image; при TLS timeout / офлайне падает.
# По умолчанию поднимаем существующий образ; FORCE_BUILD=1 — принудительная сборка.
$forceBuild = $env:FORCE_BUILD -eq "1"
$imageExists = $false
docker image inspect ai-tender-ai-tender:latest *> $null
if ($LASTEXITCODE -eq 0) { $imageExists = $true }

if ($forceBuild -or -not $imageExists) {
    Write-Host "Building image (needs Docker Hub)..." -ForegroundColor Cyan
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) {
        if ($imageExists) {
            Write-Host "Build failed (often Docker Hub TLS). Starting existing image..." -ForegroundColor Yellow
            docker compose up -d --no-build
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start docker compose"
        }
    }
}
else {
    Write-Host "Using existing image (skip build). FORCE_BUILD=1 to rebuild." -ForegroundColor DarkGray
    docker compose up -d --no-build
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start docker compose"
    }
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