# start-webapp.ps1
# Khởi động webapp (chỉ docker core services + backend + frontend, KHÔNG data pipeline)
# Data pipeline services bị loại: seaweedfs, seaweedfs-init, mlflow, spark-master, spark-worker, polaris

param(
    [switch]$Down  # Dùng -Down để tắt tất cả
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Services cần cho webapp ---
$WebappServices = @("postgres", "redis", "qdrant", "seaweedfs", "prometheus")

if ($Down) {
    Write-Host "`n[*] Stopping webapp services..." -ForegroundColor Yellow
    docker compose -f "$ProjectRoot\docker-compose.yml" stop $WebappServices
    Write-Host "[OK] Stopped." -ForegroundColor Green
    exit 0
}

# 1. Start infra containers
Write-Host "`n[1/3] Starting infra: $($WebappServices -join ', ')..." -ForegroundColor Cyan
docker compose -f "$ProjectRoot\docker-compose.yml" up -d $WebappServices

# Wait for postgres healthy
Write-Host "[*] Waiting for Postgres..." -ForegroundColor Gray
$maxWait = 30
for ($i = 0; $i -lt $maxWait; $i++) {
    $health = docker inspect --format='{{.State.Health.Status}}' movie-recsys-db 2>$null
    if ($health -eq "healthy") { break }
    Start-Sleep -Seconds 1
}
Write-Host "[OK] Infra ready." -ForegroundColor Green

# 2. Start backend (FastAPI)
Write-Host "`n[2/3] Starting backend (FastAPI)..." -ForegroundColor Cyan
$backendDir = "$ProjectRoot\apps\api"

# Tạo venv nếu chưa có
if (-not (Test-Path "$backendDir\.venv\Scripts\Activate.ps1")) {
    Write-Host "[*] Creating Python venv..." -ForegroundColor Gray
    python -m venv "$backendDir\.venv"
}

$activateScript = "$backendDir\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}
pip install -q -r "$backendDir\requirements.txt" 2>$null

$backendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    & "$dir\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
} -ArgumentList $backendDir

Write-Host "[OK] Backend started (PID: $($backendJob.Id)) -> http://localhost:8000" -ForegroundColor Green

# 3. Start frontend (Vite dev server)
Write-Host "`n[3/3] Starting frontend (Vite)..." -ForegroundColor Cyan
$frontendDir = "$ProjectRoot\apps\web"

if (-not (Test-Path "$frontendDir\node_modules")) {
    Write-Host "[*] Installing npm dependencies..." -ForegroundColor Gray
    Push-Location $frontendDir
    npm install
    Pop-Location
}

$frontendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    npm run dev
} -ArgumentList $frontendDir

Write-Host "[OK] Frontend started (PID: $($frontendJob.Id)) -> http://localhost:5173" -ForegroundColor Green

# Summary
Write-Host "`n========================================" -ForegroundColor White
Write-Host "  MovieNex WebApp Running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor White
Write-Host "  Frontend : http://localhost:5173"
Write-Host "  Backend  : http://localhost:8000"
Write-Host "  API Docs : http://localhost:8000/docs"
Write-Host "========================================" -ForegroundColor White
Write-Host "`n  Press Ctrl+C to stop, or run:"
Write-Host "  .\start-webapp.ps1 -Down" -ForegroundColor Yellow
Write-Host ""

# Keep alive — stream backend logs
try {
    $ErrorActionPreference = "Continue"
    Receive-Job -Job $backendJob -Wait
} finally {
    Write-Host "`n[*] Cleaning up jobs..." -ForegroundColor Yellow
    Stop-Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob, $frontendJob -Force -ErrorAction SilentlyContinue
}
