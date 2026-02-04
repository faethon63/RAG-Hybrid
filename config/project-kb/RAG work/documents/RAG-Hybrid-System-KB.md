# RAG-Hybrid System Knowledge Base

Complete technical documentation for the RAG-Hybrid system - a cost-optimized, multi-provider AI assistant with intelligent query routing.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Agentic Query Flow](#agentic-query-flow)
4. [Backend Components](#backend-components)
5. [Frontend Structure](#frontend-structure)
6. [API Integrations](#api-integrations)
7. [Modes and Models](#modes-and-models)
8. [Tool System](#tool-system)
9. [Project System](#project-system)
10. [Environment Differences (Local vs VPS)](#environment-differences)
11. [Data Storage](#data-storage)
12. [Deployment](#deployment)

---

## System Overview

RAG-Hybrid is a hybrid Retrieval-Augmented Generation system with **smart auto-routing** that optimizes for cost while maintaining quality. The system uses:

- **Groq (Llama 4 Scout)** - FREE orchestration and routing
- **Ollama (qwen2.5:14b)** - FREE local text generation (when available)
- **Perplexity API** - Web search and real-time data
- **Claude (Anthropic)** - Complex reasoning fallback (tiered: Haiku/Sonnet/Opus)

### Design Philosophy

1. **Cost Optimization**: Use free/cheap providers first, expensive providers only when necessary
2. **Environment Adaptation**: Automatically adapts to available services (Ollama on local, Claude fallback on VPS)
3. **Transparency**: Shows which model handled each query and why
4. **Project Context**: Queries are scoped to projects with custom system prompts and file access

---

## Architecture

### High-Level Flow

```
User Query
    │
    ▼
┌─────────────────┐
│   FastAPI       │  ← React Frontend (localhost:5173)
│   Backend       │
│   (port 8000)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query          │  ← Classifies intent (meta, task, automation, etc.)
│  Classifier     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Groq           │  ← FREE orchestrator (Llama 4 Scout)
│  Orchestrator   │     Decides: tools, routing, or direct response
└────────┬────────┘
         │
    ┌────┴────┬────────────┬─────────────┐
    ▼         ▼            ▼             ▼
┌───────┐ ┌────────┐ ┌──────────┐ ┌─────────────┐
│Ollama │ │Perplexity│ │ Claude   │ │ Deep Agent  │
│(local)│ │(web)    │ │(fallback)│ │(smolagents) │
└───────┘ └────────┘ └──────────┘ └─────────────┘
```

### Component Interaction

```
Frontend (React + Zustand)
    │
    │ HTTP/REST
    ▼
FastAPI Backend
    ├── main.py           → API endpoints, request handling
    ├── orchestrator.py   → Query analysis, model selection
    ├── groq_agent.py     → Agentic tool-calling loop
    ├── rag_core.py       → ChromaDB, embeddings, Ollama
    ├── search_integrations.py → Claude, Perplexity, Tavily APIs
    ├── deep_agent.py     → smolagents CodeAgent for complex tasks
    ├── file_tools.py     → Secure file operations
    └── query_classifier.py → Intent classification
```

---

## Agentic Query Flow

The system uses a sophisticated agentic loop powered by Groq's Llama 4 Scout model.

### Step-by-Step Flow

1. **Query Classification** (`query_classifier.py`)
   - Classifies intent: META_QUESTION, TASK_EXECUTION, INFORMATION, AUTOMATION, etc.
   - Detects user corrections
   - Identifies technical/coding questions

2. **Fast-Path Checks** (`groq_agent.py`)
   - Simple greetings → LOCAL (Ollama)
   - Product/supplier queries → Force web_search (bypass Groq decision)

3. **Groq Orchestration Loop**
   ```
   while iterations < max_tool_calls:
       response = groq.chat(messages, tools)

       if response.has_tool_calls:
           for tool_call in response.tool_calls:
               result = execute_tool(tool_call)
               messages.append(tool_result)
           continue  # Let Groq process results
       else:
           return response.content  # Final answer
   ```

4. **Tool Execution**
   - `web_search` → Perplexity API (or Tavily)
   - `search_listings` → Idealista API (real estate)
   - `complex_reasoning` → Claude (Haiku/Sonnet/Opus based on complexity)
   - `github_search` → GitHub API
   - `notion_tool` → Notion API
   - `deep_research` → Perplexity Pro

5. **Response Passthrough**
   - If Perplexity was the only tool used → return Perplexity's answer directly (prevents Groq hallucination)
   - If Claude was called and Groq's response is an excuse → return Claude's answer directly

6. **Truncation Detection**
   - Detects incomplete responses
   - Auto-falls back to Claude Haiku to complete truncated answers

### Routing Decision Table

| Query Type | Routed To | Reason |
|------------|-----------|--------|
| "Hi, hello" | LOCAL (Ollama) | Simple greeting |
| "Find cedar isolate suppliers" | Perplexity (forced) | Product search bypass |
| "What's the weather?" | Perplexity | Current data |
| "Explain quantum mechanics" | LOCAL or Groq | Static knowledge |
| "Write a Python script" | Claude Sonnet | Complex coding |
| "Fill this bankruptcy form" | Deep Agent | Multi-step automation |
| "Apartments in Barcelona under €1400" | Idealista API | Real estate |

---

## Backend Components

### `main.py` - FastAPI Server

Core endpoints:
- `POST /api/v1/query` - Main chat endpoint
- `GET /api/v1/health` - Service health check
- `POST /api/v1/reload` - Reload .env configuration
- `GET /api/v1/projects` - List projects
- `POST /api/v1/projects` - Create project
- `GET /api/v1/chats` - List chat history
- `POST /api/v1/search` - Vector search

### `orchestrator.py` - Query Orchestrator

Uses Groq to analyze queries and route to optimal model:

```python
GROQ_ROUTING_PROMPT = """
Available options:
- LOCAL: Static knowledge, casual chat
- PERPLEXITY_LOW: Current events, prices, news
- PERPLEXITY_HIGH: Deep web research
- SONNET: Complex reasoning without web
- OPUS: Critical/legal/high-stakes

Today's date: {current_date}
Query: {query}
"""
```

### `groq_agent.py` - Agentic Tool Loop

Available tools:
1. **web_search** - Perplexity/Tavily for web data
2. **search_listings** - Real estate (Idealista/Tavily)
3. **deep_research** - Perplexity Pro for comprehensive research
4. **complex_reasoning** - Claude delegation with complexity levels
5. **github_search** - Code search, file reading, issues
6. **notion_tool** - Workspace access (find_info, search, read/write pages)

### `rag_core.py` - RAG Engine

- **ChromaDB**: Vector storage for document embeddings
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Ollama**: Local LLM interface (qwen2.5:14b)
- **Document Indexing**: PDF, text, markdown support

### `search_integrations.py` - External APIs

**ClaudeSearch**:
- Standard Claude API for reasoning
- Model selection: claude-sonnet-4-5, claude-opus-4-5, claude-haiku-3-5

**PerplexitySearch**:
- Models: sonar, sonar-pro, sonar-reasoning-pro, sonar-deep-research
- Recency filters: day, week, month, year
- Academic mode for scholarly sources
- Focused mode for supplier queries

**TavilySearch**:
- Domain-filtered search
- Better for specific URL retrieval
- Real estate domain presets

**IdealistaSearch**:
- Direct real estate API (Spain/Portugal/Italy)
- OAuth authentication
- Returns actual listing URLs with prices

### `deep_agent.py` - smolagents Integration

For complex multi-step tasks:
- Uses smolagents CodeAgent
- Tools: web_search, read_document, search_local_knowledge, summarize_text, compare_items
- PDF tools: read_pdf_fields, fill_pdf_form, download_bankruptcy_form, verify_pdf
- Auto-selects model: Ollama if available, Claude Haiku fallback

### `file_tools.py` - Secure File Operations

- Path validation with allowed_paths whitelist
- Windows-to-WSL path conversion
- Operations: read_file, write_file, list_dir, search_files
- Size limit: 1MB per file

### `query_classifier.py` - Intent Classification

Intent types:
- **META_QUESTION**: "How do I..." (explain process)
- **TASK_EXECUTION**: "Fill this form" (do the task)
- **INFORMATION**: General queries
- **TECHNICAL_CODING**: Programming questions
- **AUTOMATION**: "Automate...", "Batch process..."
- **CONVERSATION**: Greetings, thanks
- **DOCUMENT_LOOKUP**: "According to page 5..."

---

## Frontend Structure

### Tech Stack
- **React** with TypeScript
- **Vite** for bundling
- **Zustand** for state management
- **TailwindCSS** for styling

### Key Components

```
frontend-react/src/
├── components/
│   ├── chat/
│   │   ├── ChatContainer.tsx  - Main chat view
│   │   ├── ChatInput.tsx      - Message input
│   │   └── MessageItem.tsx    - Individual messages
│   ├── sidebar/
│   │   ├── Sidebar.tsx        - Navigation sidebar
│   │   ├── ProjectSelector.tsx- Project dropdown
│   │   ├── ProjectForm.tsx    - Create/edit projects
│   │   └── ChatList.tsx       - Chat history
│   ├── settings/
│   │   └── SettingsPanel.tsx  - Mode/model selection
│   └── common/
│       └── icons.tsx          - SVG icons
├── stores/
│   ├── chatStore.ts           - Chat state
│   ├── projectStore.ts        - Project state
│   └── settingsStore.ts       - Settings + health
├── api/
│   └── client.ts              - API client
└── types/
    └── api.ts                 - TypeScript types
```

### Settings Store

```typescript
// Mode options
MODE_OPTIONS = [
  { value: 'auto', label: 'Smart', requiresOllama: false },
  { value: 'private', label: 'Private', requiresOllama: true },
  { value: 'research', label: 'Research', requiresOllama: false },
  { value: 'deep_agent', label: 'Deep Agent', requiresOllama: false },
]

// Model options
MODEL_OPTIONS = [
  { value: 'auto', label: 'Smart (Groq orchestrates)', requiresOllama: false },
  { value: 'local', label: 'Local (Ollama)', requiresOllama: true },
]
```

Options with `requiresOllama: true` are hidden when Ollama is unavailable (e.g., on VPS).

---

## API Integrations

### Groq API (FREE)
- **Model**: meta-llama/llama-4-scout-17b-16e-instruct
- **Purpose**: Query orchestration, tool routing
- **Endpoint**: https://api.groq.com/openai/v1/chat/completions
- **Cost**: FREE

### Anthropic (Claude) API
- **Models**:
  - claude-haiku-4-5-20251001 (fast, cheap)
  - claude-sonnet-4-5-20250929 (balanced)
  - claude-opus-4-5-20251101 (most capable)
- **Purpose**: Complex reasoning, vision/OCR, fallback
- **Cost**: Paid per token

### Perplexity API
- **Models**:
  - sonar (cheap, fast)
  - sonar-pro (thorough)
  - sonar-reasoning-pro (step-by-step)
  - sonar-deep-research (exhaustive)
- **Purpose**: Real-time web search, current data
- **Features**: Citation tokens are FREE

### Tavily API
- **Purpose**: Domain-filtered search, specific URL retrieval
- **Good for**: Real estate, specific site searches

### Idealista API
- **Purpose**: Direct real estate listings (Spain/Portugal/Italy)
- **Auth**: OAuth client credentials
- **Returns**: Actual listing URLs with prices, photos

### Ollama (Local)
- **Model**: qwen2.5:14b
- **Purpose**: FREE local text generation
- **Host**: localhost:11434
- **Note**: Only available on local (Windows), not on VPS

---

## Modes and Models

### Modes

| Mode | Description | Uses |
|------|-------------|------|
| **Smart (auto)** | Groq orchestrates, routes to best provider | Default, cost-optimized |
| **Private** | Local Ollama only, no external APIs | Offline, privacy-focused |
| **Research** | Deep Perplexity Pro search | Comprehensive web research |
| **Deep Agent** | Multi-step smolagents CodeAgent | Complex automation tasks |

### Model Tiers

| Model | When Used | Cost |
|-------|-----------|------|
| Groq (Llama 4 Scout) | Orchestration, routing | FREE |
| Ollama (qwen2.5:14b) | Local text generation | FREE |
| Perplexity (sonar) | Basic web search | Low |
| Perplexity (sonar-pro) | Thorough search | Medium |
| Claude Haiku | Simple formatting, summaries | Low |
| Claude Sonnet | Code, analysis | Medium |
| Claude Opus | Critical reasoning | High |

---

## Tool System

### Groq Agent Tools

```javascript
// Tool definitions in groq_agent.py
TOOLS = [
  {
    name: "web_search",
    description: "Search web for products, suppliers, current data",
    parameters: { query: string, provider: "perplexity" | "perplexity_pro" | "tavily" }
  },
  {
    name: "search_listings",
    description: "Search real estate listings",
    parameters: { query: string, provider: "tavily" | "idealista", city, max_price, bedrooms, has_terrace }
  },
  {
    name: "deep_research",
    description: "Comprehensive web research with Perplexity Pro",
    parameters: { query: string }
  },
  {
    name: "complex_reasoning",
    description: "Delegate to Claude for complex tasks",
    parameters: { task: string, context: string, complexity: "simple" | "medium" | "critical" }
  },
  {
    name: "github_search",
    description: "Search GitHub repos, read files, list issues",
    parameters: { action: "search_code" | "read_file" | "list_repos" | "list_issues", query, repo }
  },
  {
    name: "notion_tool",
    description: "Notion workspace access",
    parameters: { action: "find_info" | "search" | "read_page" | "create_page" | "update_page", query }
  }
]
```

### File Tools (Project-Scoped)

```python
# Available when project has allowed_paths
read_file(path, allowed_paths)   # Read file contents
write_file(path, content, allowed_paths)  # Write file
list_dir(path, allowed_paths)    # List directory
search_files(path, pattern, allowed_paths)  # Glob search
```

### PDF Tools (Deep Agent)

```python
read_pdf_form_fields(pdf_path)  # Get fillable fields
fill_pdf_form(input, output, field_values)  # Fill form
download_bankruptcy_form(form_id, output_dir)  # Official forms
verify_pdf(pdf_path)  # Check validity
```

---

## Project System

Projects provide context-scoped queries with custom configurations.

### Project Config Structure

```json
// data/project-kb/{project_name}/config.json
{
  "name": "Soap and Cosmetics",
  "description": "Formulation and suppliers for soap/cosmetics",
  "system_prompt": "You are a cosmetic formulation assistant...",
  "allowed_paths": [
    "G:\\Documents\\Recipes",
    "/mnt/g/Documents/Formulas"
  ],
  "providers": ["perplexity", "claude"],
  "default_mode": "auto"
}
```

### Project Features

1. **System Prompts**: Custom instructions for the AI
2. **Allowed Paths**: File system access (read/write)
3. **Knowledge Base**: Project-specific documents in ChromaDB
4. **Provider Preferences**: Which APIs to use

### Creating Projects

Via frontend or API:
```bash
POST /api/v1/projects
{
  "name": "My Project",
  "description": "Description",
  "system_prompt": "Custom instructions...",
  "allowed_paths": ["/path/to/files"]
}
```

---

## Environment Differences

### Local Development (Windows + WSL2)

| Feature | Status | Details |
|---------|--------|---------|
| Ollama | ✅ Available | qwen2.5:14b on GPU |
| Private Mode | ✅ Available | Uses Ollama |
| Local Model | ✅ Available | Dropdown option shown |
| Vision/OCR | Claude Sonnet | Local models hallucinate |
| ChromaDB | ✅ Local | data/chromadb/ |
| URL | localhost:5173 | Vite dev server |

### VPS Production (Linux)

| Feature | Status | Details |
|---------|--------|---------|
| Ollama | ❌ Unavailable | No GPU on VPS |
| Private Mode | ❌ Hidden | Requires Ollama |
| Local Model | ❌ Hidden | Requires Ollama |
| Vision/OCR | Claude Sonnet | Same as local |
| ChromaDB | ✅ Local | /opt/rag-hybrid/data/chromadb/ |
| URL | rag.coopeverything.org | nginx + PM2 |

### Dynamic UI Adaptation

The frontend checks health status and filters options:

```typescript
// Settings panel automatically hides Ollama-dependent options
const ollamaAvailable = health?.services?.ollama ?? false;
const modeOptions = getAvailableModeOptions(ollamaAvailable);
const modelOptions = getAvailableModelOptions(ollamaAvailable);
```

---

## Data Storage

### Directory Structure

```
data/
├── chromadb/           # Vector database (persistent)
│   └── {collection}/   # One collection per project
├── project-kb/         # Project configurations
│   └── {project}/
│       ├── config.json
│       └── documents/  # Indexed files
├── chats/              # Chat history (JSON files)
│   └── {chat_id}.json
├── documents/          # General document storage
└── cache/              # Temporary cache
```

### ChromaDB Collections

- Default: `rag_docs`
- Project-specific: `project_{name}_docs`
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

### Chat Persistence

```json
// data/chats/{uuid}.json
{
  "id": "uuid",
  "title": "Chat title",
  "project": "project_name",
  "created_at": "2026-02-04T12:00:00Z",
  "updated_at": "2026-02-04T12:30:00Z",
  "messages": [
    { "role": "user", "content": "...", "timestamp": "..." },
    { "role": "assistant", "content": "...", "metadata": {...} }
  ]
}
```

---

## Deployment

### Local Development

```powershell
# Start both services
.\start.ps1

# Or manually:
# Terminal 1 - Backend
cd backend
.\.venv\Scripts\python.exe main.py

# Terminal 2 - Frontend
cd frontend-react
npm run dev
```

### VPS Deployment

**Auto-deployment via GitHub Actions:**
```bash
git push origin main
# GitHub Actions automatically:
# 1. SSHs into VPS
# 2. Pulls code
# 3. Installs dependencies
# 4. Rebuilds frontend
# 5. Restarts PM2 backend
```

**Manual VPS commands:**
```bash
# Restart backend
pm2 restart rag-backend

# View logs
pm2 logs rag-backend --lines 50

# Check health
curl http://localhost:8000/api/v1/health

# Full redeploy
cd /opt/rag-hybrid
git pull
source venv/bin/activate
pip install -r requirements.txt
cd frontend-react && npm ci && npm run build && cd ..
pm2 restart rag-backend
```

### VPS Stack

- **Process Manager**: PM2 (rag-backend)
- **Web Server**: nginx (serves React build, proxies /api to FastAPI)
- **SSL**: Let's Encrypt via Certbot
- **Domain**: rag.coopeverything.org

---

## Configuration

### Environment Variables

```bash
# .env file

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
PERPLEXITY_API_KEY=pplx-...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...

# Ollama (local only)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

# ChromaDB
CHROMADB_PATH=./data/chromadb
CHROMADB_COLLECTION=rag_docs

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Server
FASTAPI_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173,https://rag.coopeverything.org

# Model overrides (optional)
CLAUDE_SONNET_MODEL=claude-sonnet-4-5-20250929
CLAUDE_OPUS_MODEL=claude-opus-4-5-20251101
CLAUDE_HAIKU_MODEL=claude-haiku-4-5-20251001
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

### Health Check Response

```json
{
  "status": "healthy",
  "services": {
    "local_rag": true,
    "claude_api": true,
    "perplexity_api": true,
    "ollama": false,  // false on VPS
    "chromadb": true
  },
  "timestamp": "2026-02-04T17:00:00Z"
}
```

---

## Key Design Decisions

1. **Groq for Orchestration**: Free, fast, good at instruction-following
2. **Perplexity Passthrough**: Prevents Groq from hallucinating numbers
3. **Claude Fallback**: Auto-completes truncated Groq responses
4. **Dynamic UI**: Hides unavailable options based on health
5. **Project Scoping**: Queries get relevant context and file access
6. **No Auth**: Currently disabled for personal use

---

## Common Patterns

### Adding a New Tool

1. Define in `groq_agent.py` TOOLS list
2. Register handler in `main.py`:
   ```python
   groq_agent.register_tool_handler("my_tool", my_tool_function)
   ```
3. Implement handler function

### Adding a New Mode

1. Add to `settingsStore.ts` MODE_OPTIONS
2. Handle in `main.py` query endpoint
3. Update routing logic as needed

### Debugging Query Flow

1. Check backend logs for routing decisions
2. Look for `[DEBUG]` prefixed lines
3. Check `tool_calls` in response metadata
4. Verify health endpoint for service availability

---

## Version History

- **2026-01-31**: Switched from Claude Haiku to Groq for orchestration (cost savings)
- **2026-02-04**: Added dynamic settings options based on Ollama availability
- **2026-02-04**: Added Perplexity passthrough to prevent hallucination

---

*Document generated: February 4, 2026*
