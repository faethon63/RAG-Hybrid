# RAG-Hybrid Silent Startup Script
# No visible windows - runs completely in background
# Logs output to logs/ directory for debugging

$ProjectRoot = "G:\AI-Project\RAG-Hybrid"
$LogDir = "$ProjectRoot\logs"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

# Ensure logs directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Log startup attempt (use ASCII encoding to avoid UTF-16)
"[$Timestamp] Starting RAG-Hybrid services..." | Out-File "$LogDir\startup.log" -Append -Encoding ASCII

# Start Ollama silently (if not already running)
$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollama) {
    "[$Timestamp] Starting Ollama..." | Out-File "$LogDir\startup.log" -Append -Encoding ASCII
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
} else {
    "[$Timestamp] Ollama already running" | Out-File "$LogDir\startup.log" -Append -Encoding ASCII
}

# Start backend silently using cmd wrapper for proper logging
"[$Timestamp] Starting backend..." | Out-File "$LogDir\startup.log" -Append -Encoding ASCII
$backendCmd = "cd /d `"$ProjectRoot\backend`" && `"$ProjectRoot\.venv\Scripts\python.exe`" main.py >> `"$LogDir\backend.log`" 2>&1"
$backendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $backendCmd -WindowStyle Hidden -PassThru

"[$Timestamp] Backend started (PID: $($backendProcess.Id))" | Out-File "$LogDir\startup.log" -Append -Encoding ASCII

# Wait for backend to be ready
Start-Sleep -Seconds 5

# Start frontend silently using cmd wrapper for proper logging
"[$Timestamp] Starting frontend..." | Out-File "$LogDir\startup.log" -Append -Encoding ASCII
$frontendCmd = "cd /d `"$ProjectRoot\frontend-react`" && npm run dev >> `"$LogDir\frontend.log`" 2>&1"
$frontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $frontendCmd -WindowStyle Hidden -PassThru

"[$Timestamp] Frontend started (PID: $($frontendProcess.Id))" | Out-File "$LogDir\startup.log" -Append -Encoding ASCII
"[$Timestamp] RAG-Hybrid startup complete" | Out-File "$LogDir\startup.log" -Append -Encoding ASCII
