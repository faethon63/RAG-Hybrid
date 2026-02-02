#!/bin/bash
# VPS Deployment Script for RAG-Hybrid
# Location on VPS: /opt/rag-hybrid/scripts/deploy.sh
# Called by GitHub Actions or manual deployment

set -e  # Exit on error

cd /opt/rag-hybrid

echo "=== RAG-Hybrid Deployment ==="
echo "Started at $(date)"

# Pull latest code
echo ""
echo "1. Pulling latest code..."
git fetch origin main
git reset --hard origin/main

# Backend: install dependencies
echo ""
echo "2. Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt --quiet

# Frontend: rebuild React app
echo ""
echo "3. Building React frontend..."
cd frontend-react
npm ci --silent
npm run build
cd ..

# Restart backend service
echo ""
echo "4. Restarting backend..."
pm2 restart rag-backend

# Health check
echo ""
echo "5. Running health check..."
sleep 5
HEALTH=$(curl -sf http://localhost:8000/api/v1/health || echo '{"status":"error"}')
echo "Health: $HEALTH"

echo ""
echo "=== Deployment Complete ==="
echo "Finished at $(date)"
