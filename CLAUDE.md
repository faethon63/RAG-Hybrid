# RAG-Hybrid Project

## What This Is

A hybrid Retrieval-Augmented Generation system. Single search box that queries 3 sources and blends answers:

- **LOCAL** - ChromaDB vector store + Ollama LLM (free, private, fast)
- **WEB** - Claude API for general queries
- **RESEARCH** - Perplexity API (free tier, no subscription) for deep dives
- **HYBRID** - Local + Web in parallel, Claude synthesizes combined answer (default mode)

## Tech Stack

- **Backend:** FastAPI (localhost:8000) at `backend/main.py`
- **Frontend:** Streamlit (localhost:8501) at `frontend/app.py`
- **Vector DB:** ChromaDB at `data/chromadb/`
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Local LLM:** Ollama with qwen3:8b at `G:\AI-Project\Ollama\`
- **Auth:** JWT + bcrypt, per-user rate limiting
- **Python:** 3.12.10, venv at `.venv/`

## Project Layout

```
RAG-Hybrid/
  backend/
    main.py               # FastAPI server, all endpoints
    auth.py               # JWT verify, bcrypt, rate limiter
    rag_core.py           # ChromaDB, embeddings, Ollama, indexing
    search_integrations.py # Claude, Perplexity (free), Tavily APIs
  frontend/
    app.py                # Streamlit main UI
    components/
      chat.py             # Chat message display, input, history
      search.py           # Source cards, citations
      projects.py         # Project selector, doc manager
  scripts/
    index_documents.py    # Bulk-load docs into ChromaDB
    initialize_system.py  # First-time setup
    test_system.py        # Integration tests
  config/.env.example     # All env var templates
  data/                   # chromadb/, documents/, project-kb/, cache/
```

## Key Decisions

- Perplexity API used on **free tier** (no subscription), rate-limited accordingly
- Ollama runs on Windows side, accessed via localhost:11434
- ChromaDB is persistent (on-disk at data/chromadb/)
- JWT secret + allowed users configured via .env
- GitHub repo: github.com/faethon63/RAG-Hybrid (main + dev branches)
- SSH keys for both `coopeverything` and `faethon63` GitHub accounts

## Running

```bash
# Backend
cd backend && python main.py        # http://localhost:8000

# Frontend
cd frontend && streamlit run app.py  # http://localhost:8501

# Both need .env configured (copy from .env.example)
```
