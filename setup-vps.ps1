# Initial VPS Setup for RAG-Hybrid
# Run ONCE from: G:\AI-Project\RAG-Hybrid
# Usage: .\setup-vps.ps1

param(
    [switch]$SkipClone,     # Skip git clone (repo already exists)
    [switch]$SkipSSL        # Skip SSL certificate (do manually)
)

$VPS_HOST = "72.60.27.167"
$VPS_USER = "root"
$REMOTE_PATH = "/opt/rag-hybrid"
$GITHUB_REPO = "git@github.com:faethon63/RAG-Hybrid.git"

Write-Host "=== RAG-Hybrid VPS Initial Setup ===" -ForegroundColor Cyan
Write-Host "Target: $VPS_USER@$VPS_HOST"
Write-Host ""

# Read local .env file to get API keys
$envFile = ".\.env"
if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: .env file not found. Create it first." -ForegroundColor Red
    exit 1
}

# Parse .env file for key values
$envContent = Get-Content $envFile -Raw
$ANTHROPIC_KEY = if ($envContent -match 'ANTHROPIC_API_KEY=(.+)') { $matches[1].Trim() } else { "" }
$PERPLEXITY_KEY = if ($envContent -match 'PERPLEXITY_API_KEY=(.+)') { $matches[1].Trim() } else { "" }
$GROQ_KEY = if ($envContent -match 'GROQ_API_KEY=(.+)') { $matches[1].Trim() } else { "" }
$TAVILY_KEY = if ($envContent -match 'TAVILY_API_KEY=(.+)') { $matches[1].Trim() } else { "" }
$JWT_SECRET = if ($envContent -match 'JWT_SECRET=(.+)') { $matches[1].Trim() } else { "" }

Write-Host "Found API keys in local .env:" -ForegroundColor Yellow
Write-Host "  ANTHROPIC_API_KEY: $($ANTHROPIC_KEY.Substring(0, [Math]::Min(10, $ANTHROPIC_KEY.Length)))..."
Write-Host "  PERPLEXITY_API_KEY: $($PERPLEXITY_KEY.Substring(0, [Math]::Min(10, $PERPLEXITY_KEY.Length)))..."
Write-Host "  GROQ_API_KEY: $($GROQ_KEY.Substring(0, [Math]::Min(10, $GROQ_KEY.Length)))..."
Write-Host ""

# Step 1: Clone repo (if not skipping)
if (-not $SkipClone) {
    Write-Host "Step 1: Cloning repository..." -ForegroundColor Yellow
    $cloneCmd = @"
mkdir -p $REMOTE_PATH
cd $REMOTE_PATH
if [ -d .git ]; then
    echo "Git repo already exists, pulling..."
    git fetch origin main
    git reset --hard origin/main
else
    git clone $GITHUB_REPO .
fi
"@
    ssh "$VPS_USER@$VPS_HOST" $cloneCmd
}
else {
    Write-Host "Step 1: Skipping clone (--SkipClone)" -ForegroundColor Gray
}

# Step 2: Create venv and install dependencies
Write-Host ""
Write-Host "Step 2: Setting up Python environment..." -ForegroundColor Yellow
$pythonCmd = @"
cd $REMOTE_PATH
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
"@
ssh "$VPS_USER@$VPS_HOST" $pythonCmd

# Step 3: Create directories
Write-Host ""
Write-Host "Step 3: Creating data directories..." -ForegroundColor Yellow
$dirsCmd = @"
cd $REMOTE_PATH
mkdir -p data/chromadb data/project-kb data/chats logs
"@
ssh "$VPS_USER@$VPS_HOST" $dirsCmd

# Step 4: Create .env file
Write-Host ""
Write-Host "Step 4: Creating production .env file..." -ForegroundColor Yellow
$envCmd = @"
cat > $REMOTE_PATH/.env << 'ENVEOF'
# RAG-Hybrid Production Environment
ENVIRONMENT=production

# API Keys
ANTHROPIC_API_KEY=$ANTHROPIC_KEY
PERPLEXITY_API_KEY=$PERPLEXITY_KEY
GROQ_API_KEY=$GROQ_KEY
TAVILY_API_KEY=$TAVILY_KEY

# Auth
JWT_SECRET=$JWT_SECRET

# Ollama (disabled on VPS - no GPU)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=disabled

# CORS (production)
CORS_ORIGINS=https://rag.coopeverything.org

# FastAPI
FASTAPI_PORT=8000
ENVEOF
chmod 600 $REMOTE_PATH/.env
"@
ssh "$VPS_USER@$VPS_HOST" $envCmd

# Step 5: Build React frontend
Write-Host ""
Write-Host "Step 5: Building React frontend..." -ForegroundColor Yellow
$buildCmd = @"
cd $REMOTE_PATH/frontend-react
npm install
npm run build
"@
ssh "$VPS_USER@$VPS_HOST" $buildCmd

# Step 6: Setup nginx
Write-Host ""
Write-Host "Step 6: Configuring nginx..." -ForegroundColor Yellow
$nginxCmd = @"
cp $REMOTE_PATH/deploy/nginx-rag.coopeverything.org.conf /etc/nginx/sites-available/rag.coopeverything.org
ln -sf /etc/nginx/sites-available/rag.coopeverything.org /etc/nginx/sites-enabled/
nginx -t
"@
ssh "$VPS_USER@$VPS_HOST" $nginxCmd

# Step 7: SSL certificate (if not skipping)
if (-not $SkipSSL) {
    Write-Host ""
    Write-Host "Step 7: Getting SSL certificate..." -ForegroundColor Yellow
    Write-Host "NOTE: Make sure DNS is already pointing to VPS!" -ForegroundColor Red
    $sslCmd = @"
certbot --nginx -d rag.coopeverything.org --non-interactive --agree-tos --email admin@coopeverything.org
systemctl reload nginx
"@
    ssh "$VPS_USER@$VPS_HOST" $sslCmd
}
else {
    Write-Host ""
    Write-Host "Step 7: Skipping SSL (--SkipSSL)" -ForegroundColor Gray
    Write-Host "Run manually: certbot --nginx -d rag.coopeverything.org"
}

# Step 8: Setup PM2
Write-Host ""
Write-Host "Step 8: Setting up PM2..." -ForegroundColor Yellow
$pm2Cmd = @"
cd $REMOTE_PATH
pm2 start ecosystem.config.js
pm2 save
pm2 startup
"@
ssh "$VPS_USER@$VPS_HOST" $pm2Cmd

# Step 9: Final health check
Write-Host ""
Write-Host "Step 9: Running health check..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
$healthCmd = @"
curl -sf http://localhost:8000/api/v1/health
"@
ssh "$VPS_USER@$VPS_HOST" $healthCmd

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Add DNS A record: rag.coopeverything.org -> $VPS_HOST"
Write-Host "2. If SSL was skipped, run: certbot --nginx -d rag.coopeverything.org"
Write-Host "3. Test: https://rag.coopeverything.org"
Write-Host ""
Write-Host "GitHub Actions setup:" -ForegroundColor Yellow
Write-Host "1. Go to repo Settings -> Secrets and variables -> Actions"
Write-Host "2. Add secret: VPS_SSH_KEY (your SSH private key)"
