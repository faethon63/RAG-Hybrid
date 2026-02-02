# RAG-Hybrid Startup Script
# Runs backend in background, frontend in foreground
# Usage: .\start.ps1 or .\start.ps1 -Stop

param(
    [switch]$Stop,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ProjectRoot = "G:\AI-Project\RAG-Hybrid"
$VenvActivate = "$ProjectRoot\.venv\Scripts\Activate.ps1"
$LogFile = "$ProjectRoot\logs\backend.log"

# Ensure logs directory exists
if (-not (Test-Path "$ProjectRoot\logs")) {
    New-Item -ItemType Directory -Path "$ProjectRoot\logs" | Out-Null
}

function Start-Ollama {
    # Check if Ollama is already running
    $ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaRunning) {
        Write-Host "Ollama already running." -ForegroundColor Green
        return
    }

    Write-Host "Starting Ollama..." -ForegroundColor Cyan
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2

    # Verify it started
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
        Write-Host "Ollama started successfully." -ForegroundColor Green
    } catch {
        Write-Host "Ollama may still be starting..." -ForegroundColor Yellow
    }
}

function Stop-Services {
    Write-Host "Stopping RAG services..." -ForegroundColor Yellow
    Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "Services stopped." -ForegroundColor Green
}

function Start-Backend {
    Write-Host "Starting backend..." -ForegroundColor Cyan

    # Start backend as a background job, redirect output to log file
    $backendJob = Start-Job -ScriptBlock {
        param($root, $venv, $log)
        Set-Location $root
        & $venv
        Set-Location "$root\backend"
        python main.py 2>&1 | Tee-Object -FilePath $log
    } -ArgumentList $ProjectRoot, $VenvActivate, $LogFile

    # Wait for backend to start
    Start-Sleep -Seconds 3

    # Check if backend is responding
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" -TimeoutSec 5
        Write-Host "Backend started: http://localhost:8000" -ForegroundColor Green
        Write-Host "  Status: $($response.status)" -ForegroundColor Gray
    } catch {
        Write-Host "Backend starting... (check $LogFile for details)" -ForegroundColor Yellow
    }

    return $backendJob
}

function Start-Frontend {
    Write-Host "Starting frontend..." -ForegroundColor Cyan
    Write-Host "Frontend will run in this window. Press Ctrl+C to stop." -ForegroundColor Gray
    Write-Host ""

    # Check if node_modules exists
    if (-not (Test-Path "$ProjectRoot\frontend-react\node_modules")) {
        Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
        Set-Location "$ProjectRoot\frontend-react"
        npm install
    }

    # Open browser after a short delay
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:5173"
    } | Out-Null

    Set-Location "$ProjectRoot\frontend-react"
    npx vite
}

# Main logic
if ($Stop) {
    Stop-Services
    exit 0
}

# Stop any existing services first
Stop-Services

if ($BackendOnly) {
    Start-Ollama
    Start-Backend | Out-Null
    Write-Host ""
    Write-Host "Backend running in background. Log: $LogFile" -ForegroundColor Green
    Write-Host "To stop: .\start.ps1 -Stop" -ForegroundColor Gray
    exit 0
}

if ($FrontendOnly) {
    Start-Frontend
    exit 0
}

# Start both (default)
Start-Ollama
$job = Start-Backend
Write-Host ""
Start-Frontend

# When frontend stops (Ctrl+C), also stop the backend job
Write-Host ""
Write-Host "Stopping backend..." -ForegroundColor Yellow
Stop-Job $job -ErrorAction SilentlyContinue
Remove-Job $job -ErrorAction SilentlyContinue
Stop-Services
