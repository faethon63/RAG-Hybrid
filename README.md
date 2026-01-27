# RAG-Hybrid

Hybrid Retrieval-Augmented Generation system combining local document search with web-based AI providers.

## Architecture

- **Local RAG** - ChromaDB + sentence-transformers for private document search
- **Ollama** - Local LLM inference (qwen3:8b or similar)
- **Claude API** - Web search and answer synthesis
- **Perplexity API** - Deep research queries

## Structure

```
RAG-Hybrid/
├── backend/          # FastAPI server
├── frontend/         # Streamlit UI
├── scripts/          # Setup & maintenance scripts
├── config/           # Configuration files
├── data/             # ChromaDB, documents, cache (gitignored)
├── docs/             # Documentation
├── tests/            # Test suite
└── logs/             # Application logs (gitignored)
```

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/faethon63/RAG-Hybrid.git
cd RAG-Hybrid

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your API keys

# 5. Run backend
cd backend && python main.py

# 6. Run frontend (separate terminal)
cd frontend && streamlit run app.py
```

## Query Modes

| Mode | Source | Use Case |
|------|--------|----------|
| `local` | ChromaDB docs | Private/fast queries |
| `web` | Claude/Perplexity | Current information |
| `research` | Perplexity deep | Thorough analysis |
| `hybrid` | Local + Web | Best of both (default) |

## API Endpoints

- `GET /` - System info
- `GET /api/v1/health` - Health check
- `POST /api/v1/query` - Main query endpoint
- `POST /api/v1/index` - Index documents
- `GET /api/v1/projects` - List projects

## Requirements

- Python 3.10+
- Ollama (for local LLM)
- API keys for Claude and/or Perplexity
