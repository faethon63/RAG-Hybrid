# TogetherOS RAG System - Complete Implementation Guide

## 🎯 System Overview

Your RAG system combines:
- **Claude Pro** (web search, general Q&A) - $20/month
- **Perplexity API** (deep research) - $20/month
- **Local RAG** (private docs, TogetherOS KB) - Free
- **VPS Gateway** (multi-device access) - Your existing VPS
- **Windows PC** (compute power) - Your existing hardware

**Total Cost: $40/month**

---

## 📁 Project Structure

```
G:/Coopeverything/TogetherOS/rag-system/
│
├── backend/                    # API server (runs on VPS + PC)
│   ├── main.py                # FastAPI main app
│   ├── rag_core.py            # Core RAG logic
│   ├── search_integrations.py # Claude/Perplexity/Tavily
│   ├── auth.py                # Authentication & rate limiting
│   ├── models.py              # Data models
│   └── utils.py               # Helper functions
│
├── frontend/                   # Web interface (runs on VPS)
│   ├── app.py                 # Streamlit UI
│   ├── components/            # UI components
│   │   ├── chat.py
│   │   ├── search.py
│   │   └── projects.py
│   └── assets/                # Static files
│
├── scripts/                    # Setup & maintenance
│   ├── setup.py               # System inventory checker
│   ├── install_missing.sh     # Auto-install script
│   ├── initialize_system.py   # First-time setup
│   ├── index_documents.py     # Bulk document indexing
│   ├── test_system.py         # Integration tests
│   └── deploy_vps.sh          # VPS deployment
│
├── data/                       # Data storage (on PC)
│   ├── chromadb/              # Vector database
│   ├── documents/             # Raw documents
│   ├── project-kb/            # Per-project knowledge bases
│   │   ├── togetheros/
│   │   ├── personal/
│   │   └── research/
│   └── cache/                 # Query cache
│
├── config/                     # Configuration
│   ├── .env                   # API keys & settings
│   ├── system_inventory.json  # Auto-generated inventory
│   ├── projects.yaml          # Project definitions
│   └── nginx.conf             # VPS reverse proxy config
│
├── logs/                       # Application logs
│   ├── rag_system.log
│   ├── query.log
│   └── error.log
│
└── docs/                       # Documentation
    ├── setup.md
    ├── api.md
    ├── deployment.md
    └── troubleshooting.md
```

---

## 🚀 Installation Steps

### Step 1: Run System Inventory

Save the `rag-system-setup.py` file I created to your PC, then run:

```bash
# On PC (Windows PowerShell)
cd G:\Coopeverything\TogetherOS
python rag-system-setup.py
```

This will:
- ✅ Check what you have installed
- ❌ Identify what's missing
- 📝 Generate installation script
- 💾 Save system inventory

### Step 2: Install Missing Components

The script will create `install_missing.sh`. Review it, then run:

```bash
# If using UV (recommended)
G:\AI-Project\Python\Scripts\uv.exe pip install langchain chromadb streamlit fastapi sentence-transformers httpx uvicorn python-dotenv bcrypt pyjwt

# Install Ollama (Windows)
winget install Ollama.Ollama

# Pull LLM model
ollama pull qwen3:8b
```

### Step 3: Configure API Keys

Create `.env` file in `rag-system/config/`:

```bash
# Copy template
cp env.template rag-system/config/.env

# Edit with your keys
# - ANTHROPIC_API_KEY from https://console.anthropic.com/settings/keys
# - PERPLEXITY_API_KEY from https://www.perplexity.ai/settings/api
```

### Step 4: Initialize System

```bash
cd rag-system
python scripts/initialize_system.py
```

This will:
- Create ChromaDB collections
- Index TogetherOS documentation
- Set up project knowledge bases
- Test all connections

### Step 5: Start Local Services (PC)

```bash
# Terminal 1: Start backend API
cd backend
python main.py
# Access at: http://localhost:8000

# Terminal 2: Start Streamlit UI (optional - can run on VPS instead)
cd frontend
streamlit run app.py
# Access at: http://localhost:8501
```

---

## 🌐 VPS Deployment (Multi-Device Access)

### Architecture

```
Internet
    ↓
VPS (Public IP)
    ├── Nginx (HTTPS, reverse proxy)
    ├── Streamlit UI (Port 8501)
    └── SSH Tunnel → PC (Port 8002)
            ↓
        PC Backend API
            ├── ChromaDB
            ├── Ollama
            └── Windows MCPs
```

### VPS Setup Steps

#### 1. Create SSH Tunnel (PC → VPS)

On your PC, run:

```bash
# Install autossh (keeps tunnel alive)
winget install autossh

# Create persistent tunnel
autossh -f -N -R 8002:localhost:8000 your_vps_user@your_vps_host
```

This forwards your PC's backend (port 8000) to VPS port 8002.

#### 2. Deploy Frontend to VPS

```bash
# From PC, deploy to VPS
rsync -avz -e "ssh -p 22" \
    rag-system/frontend/ \
    rag-system/config/.env \
    your_vps_user@your_vps_host:/home/your_vps_user/rag-frontend/

# SSH into VPS
ssh your_vps_user@your_vps_host

# Install Python packages on VPS
cd ~/rag-frontend
python3 -m venv venv
source venv/bin/activate
pip install streamlit httpx python-dotenv

# Edit .env to point to PC backend
echo "PC_API_URL=http://localhost:8002" >> .env

# Start Streamlit (use systemd for production)
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

#### 3. Configure Nginx

On VPS, create `/etc/nginx/sites-available/rag-system`:

```nginx
server {
    listen 80;
    server_name your_domain.com;  # or VPS IP

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 4. Add SSL (Optional but Recommended)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

---

## 🔐 Sharing Access with Others

### Option 1: Simple Token Auth

Generate user tokens:

```python
# On PC, run:
python scripts/generate_user_token.py --username "friend_name"
# Outputs: JWT token

# Share token with user
# They add to .env: AUTH_TOKEN=eyJ...
```

### Option 2: Username/Password

Add users in `.env`:

```bash
# Generate password hash
python -c "import bcrypt; print(bcrypt.hashpw(b'their_password', bcrypt.gensalt()).decode())"

# Add to ALLOWED_USERS
ALLOWED_USERS=george:$2b$12$...,friend:$2b$12$...
```

### Option 3: OAuth (Advanced)

Integrate Google/GitHub OAuth in `backend/auth.py` (I can provide code if needed).

---

## 📱 Access from Multiple Devices

### On PC (Direct)
```
http://localhost:8501
```

### On Android/Other Devices (via VPS)
```
http://your_vps_ip:8501
# or with SSL:
https://your_domain.com
```

### On Android Termux (Direct to PC - same network)
```bash
# If on same WiFi as PC
http://192.168.x.x:8501  # Replace with PC's local IP
```

---

## 🧪 Testing the System

### Test 1: Local RAG Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "query": "What is TogetherOS?",
    "mode": "local"
  }'
```

Expected: Answer from your docs + sources.

### Test 2: Web Search Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "query": "Latest news on AI regulation",
    "mode": "web"
  }'
```

Expected: Answer from Claude/Perplexity + web sources.

### Test 3: Hybrid Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "query": "How does TogetherOS compare to similar platforms?",
    "mode": "hybrid"
  }'
```

Expected: Synthesized answer from local + web.

---

## 🎨 UI Features

The Streamlit interface includes:

### 1. Chat Interface
- Type queries naturally
- See answers with citations
- Switch modes (local/web/hybrid)

### 2. Project Switcher
- Select active project (TogetherOS, Personal, Research)
- Each project has separate knowledge base

### 3. Document Manager
- Upload new documents
- View indexed documents
- Re-index on demand

### 4. Search History
- View past queries
- Re-run searches
- Export results

### 5. System Status
- Check API health
- View token usage
- Monitor rate limits

---

## 💰 Cost Optimization Tips

### 1. Use Local RAG First
Set default mode to "local" - only use web search when needed.

### 2. Cache Results
The system caches queries for 1 hour - repeated questions are free.

### 3. Batch Queries
If researching a topic, ask related questions in one session.

### 4. Monitor Usage
Check dashboard monthly to see Claude vs Perplexity usage.

---

## 🐛 Troubleshooting

### Issue: Ollama not responding

```bash
# Check if running
ollama list

# Restart service (Windows)
# Find "Ollama" in Task Manager → Restart

# Or reinstall
winget uninstall Ollama.Ollama
winget install Ollama.Ollama
```

### Issue: ChromaDB errors

```bash
# Clear database (WARNING: deletes all indexed docs)
rm -rf rag-system/data/chromadb/*

# Re-index
python scripts/index_documents.py
```

### Issue: VPS can't reach PC

```bash
# Check SSH tunnel
ps aux | grep autossh

# Restart tunnel
pkill autossh
autossh -f -N -R 8002:localhost:8000 your_vps_user@your_vps_host

# Test connection from VPS
curl http://localhost:8002/api/v1/health
```

### Issue: API rate limits

Check `.env` settings:
```bash
RATE_LIMIT_RPM=30        # Increase if needed
RATE_LIMIT_DAILY=500     # Increase if needed
```

---

## 🔄 Maintenance

### Daily
- Monitor logs: `tail -f logs/rag_system.log`

### Weekly
- Check token usage dashboard
- Review query history for patterns

### Monthly
- Update Ollama model: `ollama pull qwen3:8b`
- Update Python packages: `uv pip install --upgrade -r requirements.txt`
- Backup ChromaDB: `cp -r data/chromadb data/chromadb.backup`

---

## 📚 Next Steps

After setup, explore:

1. **Add More Projects**
   ```bash
   python scripts/create_project.py --name "Research" --docs /path/to/research/docs
   ```

2. **Integrate Notion**
   - Already have Notion MCP
   - Add to search sources in `backend/rag_core.py`

3. **Add More LLMs**
   ```bash
   ollama pull mistral
   ollama pull llama3:70b  # If you have GPU RAM
   ```

4. **Custom Skills**
   - Create TogetherOS-specific RAG skills
   - Add to `.claude/skills/`

---

## 🆘 Getting Help

**System Issues:**
- Check logs in `rag-system/logs/`
- Run `python scripts/test_system.py`

**API Issues:**
- Test endpoints with `curl` commands above
- Check API keys in `.env`

**Deployment Issues:**
- Verify SSH tunnel: `ps aux | grep autossh`
- Check nginx: `sudo nginx -t`

**Questions:**
- Open issue in TogetherOS repo
- Check documentation in `rag-system/docs/`

---

## ✅ Success Checklist

- [ ] System inventory completed
- [ ] Missing components installed
- [ ] API keys configured in `.env`
- [ ] Documents indexed successfully
- [ ] Local backend running (http://localhost:8000)
- [ ] Local UI running (http://localhost:8501)
- [ ] SSH tunnel to VPS established
- [ ] VPS frontend accessible remotely
- [ ] Test queries working (local/web/hybrid)
- [ ] Authentication working
- [ ] Can access from multiple devices

---

**Ready to build? Run the setup script and let's get started! 🚀**
