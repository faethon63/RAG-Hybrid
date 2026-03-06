# Long-term Memory

Facts, decisions, and learnings accumulated from conversations.

## Decisions
- Uses Gemini 2.5 Flash as primary orchestrator (paid, ~$0.15/$0.60 per M tokens) with Groq as free fallback
- Vision/OCR routes to Claude Sonnet 4.5 (local models hallucinate document text)
- Ollama qwen2.5:14b for local text generation

## Local PC Setup
- Windows 11 Home, Python 3.12.10, venv at `G:\AI-Project\RAG-Hybrid\.venv\`
- Services: Backend (FastAPI :8000), Frontend (Vite :5173), Ollama (:11434)
- Ollama model: qwen2.5:14b (9GB, text generation)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local)
- ChromaDB: persistent on-disk at `data/chromadb/`
- Auto-starts on Windows login via Task Scheduler → `start-hidden.vbs` → `start-background.ps1`
- Logs: `logs/backend.log`, `logs/startup.log`

## VPS Setup
- IP: 72.60.27.167, URL: https://rag.coopeverything.org, SSH: root
- Path: /opt/rag-hybrid, Python venv at /opt/rag-hybrid/venv/
- PM2 process: `rag-backend` (FastAPI on :8000)
- nginx: reverse proxy (HTTPS) + serves React build from `frontend-react/dist/`
- PostgreSQL: database `rag_hybrid`, user `rag_app` — stores chats, brain_items, push_subscriptions, system_status
- No Ollama (no GPU) — text generation falls back to Claude (paid)
- Deploy: push to main → GitHub Actions auto-deploys (SSH, git pull, pip install, npm build, pm2 restart)

## Sync Strategy
- **Git syncs:** code, project configs (`config/projects/`), KB documents (`config/project-kb/`)
- **PostgreSQL syncs:** chats, brain_items, push subscriptions (both PC and VPS connect to same DB)
- **Local only:** ChromaDB, .env files, Ollama, allowed_paths, indexed_files

## API Services
- Gemini 2.5 Flash — primary orchestrator + tool routing (paid, cheap)
- Groq Llama 4 Scout 17B — FREE fallback orchestrator + background tasks (session review, heartbeat)
- Perplexity (sonar) — FREE web search
- Claude Sonnet 4.5 — paid, vision/OCR + complex reasoning
- Claude Opus 4.6 — paid, deep reasoning
- Ollama qwen2.5:14b — FREE local text generation (PC only)

## Other Facts
- GitHub: faethon63/RAG-Hybrid (SSH key: `G:/AI-Project/ssh_keys/id_ed25519`)
- Claude Max plan subscriber

## Relocation Plans
- Found an apartment in Gracia neighborhood, Barcelona for 1200 euros, moving May 2026
