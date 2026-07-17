$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
docker compose down
Write-Host "AI Tender stopped." -ForegroundColor Green