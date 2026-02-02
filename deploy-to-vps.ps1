# Deploy RAG-Hybrid to VPS
# Run from: G:\AI-Project\RAG-Hybrid
# Usage: .\deploy-to-vps.ps1

param(
    [switch]$SkipBuild,  # Skip React build on VPS
    [switch]$BackendOnly # Only restart backend
)

$VPS_HOST = "72.60.27.167"
$VPS_USER = "root"
$REMOTE_PATH = "/opt/rag-hybrid"

Write-Host "=== Deploying RAG-Hybrid to VPS ===" -ForegroundColor Cyan
Write-Host "Target: $VPS_USER@$VPS_HOST:$REMOTE_PATH"
Write-Host ""

# Build deploy command based on flags
if ($BackendOnly) {
    $deployCmd = @"
cd $REMOTE_PATH
source venv/bin/activate
pip install -r requirements.txt --quiet
pm2 restart rag-backend
sleep 3
curl -sf http://localhost:8000/api/v1/health
"@
}
elseif ($SkipBuild) {
    $deployCmd = @"
cd $REMOTE_PATH
git fetch origin main
git reset --hard origin/main
source venv/bin/activate
pip install -r requirements.txt --quiet
pm2 restart rag-backend
sleep 3
curl -sf http://localhost:8000/api/v1/health
"@
}
else {
    $deployCmd = @"
cd $REMOTE_PATH
git fetch origin main
git reset --hard origin/main
source venv/bin/activate
pip install -r requirements.txt --quiet
cd frontend-react
npm ci --silent
npm run build
cd ..
pm2 restart rag-backend
sleep 5
curl -sf http://localhost:8000/api/v1/health
"@
}

Write-Host "Running deployment..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_HOST" $deployCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Deployment Successful ===" -ForegroundColor Green
    Write-Host "Site: https://rag.coopeverything.org"
}
else {
    Write-Host ""
    Write-Host "=== Deployment Failed ===" -ForegroundColor Red
    exit 1
}
