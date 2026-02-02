# RAG-Hybrid Build Log

## Project Goal
Build a hybrid RAG system that can:
- Answer questions using local documents (ChromaDB + embeddings)
- Route queries intelligently (FREE orchestration via Groq)
- Execute web searches via Perplexity when current data is needed
- Maintain conversation context across multi-turn dialogs
- Handle complex multi-step research tasks

---

## Executive Summary

### What We Built
A cost-effective RAG system that uses:
- **Groq (FREE)** as the conversational brain with tool calling
- **Perplexity** for real-time web search
- **Ollama** for simple local chat (optional)
- **Claude** for complex reasoning (when needed)

### Total Cost: ~$0/month for typical usage
- Groq: Free tier (generous limits)
- Perplexity: Free tier (~$0.001/query)
- Claude: Only for complex reasoning (rare)
- Ollama: Free (local)

---

## Evolution of the Architecture

### Phase 1: Original Design (Failed)

**Architecture:**
```
User Query
    ↓
Ollama qwen2.5:14b (Orchestrator)
    ↓
    ├── Simple Q&A → Answer directly
    ├── Tool Use → Execute tools
    └── Complex → Delegate to Claude
```

**Why it failed:**
- Ollama qwen2.5:14b cannot reliably use tools
- Model hallucinated file names instead of calling `list_dir`
- Model said "I don't have access" when it DID have folder access
- Model generated fake tool output in prose
- Model didn't know the current year (said 2023 in 2026)

**Lesson:** 14B parameter local models lack the reasoning depth for tool orchestration.

---

### Phase 2: Claude Haiku as Router (Improved but Costly)

**Architecture:**
```
User Query
    ↓
Claude Haiku (Router) ← $0.00025/query
    ↓
    ├── Simple → Ollama (free)
    ├── Web Search → Perplexity
    ├── Complex → Claude Sonnet/Opus
    └── Tools → Haiku executes
```

**Problems:**
1. Haiku only ROUTED queries - didn't maintain conversation context
2. Each query was treated in isolation
3. Follow-up questions lost context (e.g., "Barcelona rentals" → "cheaper areas" → got Charleston results)
4. Still costs money per query

---

### Phase 3: Groq as FREE Router (Wrong Architecture)

**Architecture:**
```
User Query
    ↓
Groq Llama 3.3 70B (Router) ← FREE!
    ↓
    ├── Simple → Ollama
    ├── Web Search → Perplexity (direct)
    └── Complex → Claude
```

**Problems:**
1. Same as Phase 2 - stateless routing only
2. Perplexity received queries without conversation history
3. Follow-up questions had no context
4. User asked about Barcelona → got San Francisco results on refinement

**Lesson:** A router isn't enough. The brain needs to maintain context.

---

### Phase 4: Groq as Conversational Agent (Current - Working)

**Architecture:**
```
User Query + Full Conversation History
    ↓
Groq Agent (Llama 4 Scout) ← FREE, maintains context
    ↓
    Groq decides: respond OR call tools
    ↓
    If tool needed: web_search → Perplexity → results back to Groq
    ↓
    Groq synthesizes final answer with context + tool results + URLs
    ↓
    User gets coherent response
```

**Why this works:**
1. Groq receives FULL conversation history
2. Groq decides when to use tools (not just routing)
3. Tool results (including URLs) are passed back to Groq
4. Groq synthesizes the final response with context
5. Follow-up questions maintain context (Barcelona → cheaper areas near beach → correct results)

---

## Key Technical Decisions

### 1. Model Selection for Tool Use

**Problem:** Not all models support tool calling reliably.

**What we tried:**
| Model | Tool Calling | Result |
|-------|--------------|--------|
| qwen2.5:14b (Ollama) | No native support | Failed - hallucinated tools |
| llama-3.3-70b-versatile (Groq) | Partial | Failed - generated XML-style calls |
| llama-4-scout-17b-16e-instruct (Groq) | Native support | **Works** |

**Solution:** Use `meta-llama/llama-4-scout-17b-16e-instruct` - specifically trained for tool use.

---

### 2. Passing Tool Results with URLs

**Problem:** Perplexity returns citations with URLs, but they weren't appearing in final responses.

**What happened:**
1. Groq called `web_search`
2. Perplexity returned answer + citations with URLs
3. Only the answer text was passed to Groq
4. URLs were stripped out
5. Groq tried to cite sources but didn't have the URLs
6. Model generated incomplete markdown like `[Idealista](` → API error

**Solution:** Append URLs to tool result before sending to Groq:
```python
if citations:
    links_text = "\n\nSources:\n" + "\n".join(
        f"- {c.get('url', '')}"
        for c in citations if c.get('url')
    )
    tool_result += links_text
```

---

### 3. Avoiding Markdown Link Errors

**Problem:** Model kept generating malformed markdown: `[URL](` instead of `[title](URL)`

**Solution:** System prompt instructs model to use plain URLs:
```
IMPORTANT: When citing sources, use PLAIN URLs only. Do NOT use markdown link format.
Good: "Visit https://example.com for more info"
Bad: "[Example](https://example.com)"
```

---

### 4. tool_choice Parameter

**Problem:** Wanted to force tool use for search queries.

**What we tried:**
- `tool_choice: "required"` → Caused API errors when model tried to respond
- `tool_choice: "auto"` → Works, model decides when to use tools

**Solution:** Use `tool_choice: "auto"` and let the model decide. The system prompt guides it to use tools for current data.

---

### 5. Default Request Model

**Problem:** Queries weren't using the orchestrator.

**Root cause:** `QueryRequest.model` defaulted to `"local"` instead of `"auto"`.

**Fix in main.py:**
```python
class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"  # Was "hybrid"
    model: str = "auto"  # Was "local" - bypassed orchestrator!
```

---

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `backend/groq_agent.py` | Conversational agent with tool calling |
| `backend/orchestrator.py` | Query routing (used for simple classification) |
| `backend/query_classifier.py` | Intent classification (info, automation, etc.) |
| `backend/config.py` | Centralized config with lazy env loading |
| `backend/deep_agent.py` | smolagents integration for multi-step tasks |
| `backend/file_tools.py` | Secure file operations for projects |
| `frontend/constants.py` | Shared model options, pricing |
| `frontend/components/settings.py` | Global RAG settings UI |
| `start.ps1` | Windows startup script |

### Modified Files

| File | Changes |
|------|---------|
| `backend/main.py` | Added Groq agent integration, tool handlers |
| `backend/search_integrations.py` | Fixed Perplexity API format, added conversation history |
| `backend/rag_core.py` | Added date injection, Haiku orchestration |
| `frontend/components/chat.py` | Fixed thinking tag regex, added chat persistence |
| `.env` | Added GROQ_API_KEY, updated model names |
| `config/.env.example` | Added model configuration section |

---

## Environment Configuration

### Required API Keys

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...  # For Claude (complex reasoning)
PERPLEXITY_API_KEY=pplx-...   # For web search
GROQ_API_KEY=gsk_...          # For FREE orchestration (get from console.groq.com)
```

### Model Configuration

```bash
# Models (in .env)
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TOOL_MODEL=meta-llama/llama-4-scout-17b-16e-instruct  # For tool calling
CLAUDE_OPUS_MODEL=claude-opus-4-5-20251101
CLAUDE_SONNET_MODEL=claude-sonnet-4-5-20250929
CLAUDE_HAIKU_MODEL=claude-haiku-4-5-20251001
OLLAMA_MODEL=qwen2.5:14b
```

---

## How to Reproduce

### 1. Initial Setup

```powershell
# Clone repo
git clone https://github.com/faethon63/RAG-Hybrid.git
cd RAG-Hybrid

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp config/.env.example .env
# Edit .env and add your API keys
```

### 2. Get API Keys

1. **Groq (FREE):** https://console.groq.com - No credit card required
2. **Perplexity:** https://www.perplexity.ai/settings/api - Free tier available
3. **Anthropic:** https://console.anthropic.com - For Claude (optional, only for complex reasoning)

### 3. Start Ollama (Optional)

```powershell
# Install Ollama from https://ollama.com
ollama pull qwen2.5:14b
ollama serve
```

### 4. Start the App

```powershell
.\start.ps1
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## Testing the System

### Test 1: Web Search with Context

```
Query 1: "best month to rent in Barcelona"
Expected: Groq calls web_search, returns current rental info

Query 2: "what about cheaper areas near the beach?"
Expected: Groq maintains context, searches for Barcelona beach area rentals
```

### Test 2: Direct Knowledge

```
Query: "What is the capital of France?"
Expected: Groq answers directly without tools
```

### Test 3: Current Data

```
Query: "find Barcelona rentals under 1400 euros with balcony"
Expected: Groq searches, returns listings with actual URLs
```

---

## Common Issues and Fixes

### Issue: "Groq error: 400"
**Cause:** Model generating malformed output (incomplete markdown links)
**Fix:** System prompt instructs plain URLs, not markdown

### Issue: Queries going to local instead of orchestrator
**Cause:** `QueryRequest.model` defaulted to `"local"`
**Fix:** Change default to `"auto"` in main.py

### Issue: Follow-up questions lose context
**Cause:** Router-only architecture, no conversation history passed
**Fix:** Use Groq as conversational agent with full history

### Issue: URLs not appearing in responses
**Cause:** Only answer text passed to Groq, not citations
**Fix:** Append URLs to tool results

### Issue: Tool calling fails with llama-3.3-70b
**Cause:** Model uses legacy XML-style function calls
**Fix:** Use `meta-llama/llama-4-scout-17b-16e-instruct`

### Issue: Groq returns generic "visit website" instead of specific URLs
**Cause:**
1. Tool description didn't instruct to include all user filters in query
2. System prompt wasn't explicit enough about requiring actual URLs

**Fix in groq_agent.py:**
1. Updated web_search tool description:
   - "MUST include ALL user criteria: price limits, location, features..."
2. Updated system prompt with explicit rules:
   - "ALWAYS include the actual URLs from search results in your response"
   - Added examples of good vs bad responses

### Issue: User filters (price €1400, balcony, near beach) not passed to Perplexity
**Cause:** Groq was forming search queries from general context, not including specific criteria
**Fix:** Tool description now explicitly instructs to include ALL user criteria in the query

---

## Cost Analysis

### Monthly Usage Estimate (Personal Use)

| Service | Usage | Cost |
|---------|-------|------|
| Groq | 1000 queries | $0 (free tier) |
| Perplexity | 500 searches | $0 (free tier) |
| Claude | 50 complex queries | ~$0.50 |
| Ollama | Unlimited | $0 |
| **Total** | | **~$0.50/month** |

Compared to Claude Max subscription: **$100/month → $0.50/month**

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Streamlit - localhost:8501)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│                     (localhost:8000)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GroqAgent                              │   │
│  │  • Receives query + conversation history                 │   │
│  │  • Maintains context across turns                        │   │
│  │  • Decides: respond directly OR call tools              │   │
│  │  • Synthesizes final response with sources              │   │
│  │  • Model: meta-llama/llama-4-scout-17b-16e-instruct    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                    │                │                │          │
│            ┌───────┴───────┐ ┌──────┴──────┐ ┌──────┴──────┐  │
│            │  web_search   │ │deep_research│ │complex_reason│  │
│            │     Tool      │ │    Tool     │ │    Tool      │  │
│            └───────┬───────┘ └──────┬──────┘ └──────┬──────┘  │
│                    │                │                │          │
│                    ▼                ▼                ▼          │
│            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│            │ Perplexity  │  │ Perplexity  │  │   Claude    │  │
│            │   sonar     │  │  sonar-pro  │  │   Sonnet    │  │
│            └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌─────────────┐         ┌─────────────┐
            │  ChromaDB   │         │   Ollama    │
            │ (local docs)│         │ (optional)  │
            └─────────────┘         └─────────────┘
```

---

## Lessons Learned Summary

1. **Local LLMs cannot orchestrate tools reliably** - Use cloud models for routing/tool decisions
2. **Routing isn't enough** - The brain must maintain conversation context
3. **Tool results must include all data** - Pass URLs, not just text
4. **Model selection matters for tools** - Use models specifically trained for tool calling
5. **Test capabilities before building architecture** - Don't assume models can do something
6. **Free tiers are powerful** - Groq + Perplexity free tiers are sufficient for most use cases
7. **System prompts have limits** - You can't prompt-engineer around capability gaps
8. **Tool descriptions must be explicit** - LLMs follow the description literally; if it doesn't say "include price limits in query", it won't
9. **Examples in prompts are powerful** - Showing good vs bad response examples helps LLMs understand expectations

---

## Version History

- **2026-02-01:** Added multi-provider search: Perplexity (default), Perplexity Pro (deep), Tavily (URLs), Idealista API (direct listings). New `search_listings` tool for real estate. User can request "perplexity pro" for thorough searches.
- **2026-02-01:** Fixed search query formation - Groq now includes all user filters (price, features, location) in web_search queries. Improved system prompt to require actual URLs in responses.
- **2026-02-01:** Implemented Groq conversational agent with tool calling. Fixed URL passing. Updated to Llama 4 Scout model.
- **2026-01-31:** Switched from Haiku to Groq for FREE orchestration. Fixed Perplexity API. Updated Claude model IDs to 4.5 series.
- **2026-01-31:** Implemented Haiku orchestration. Fixed thinking tags. Added model cleanup.
- **2026-01-31:** Initial build log. Documented failure of local model as orchestrator.

---

## Next Steps

1. [ ] Add file tools to Groq agent (read_file, list_dir for projects)
2. [ ] Implement streaming responses
3. [ ] Add cost tracking per query
4. [ ] Create automated tests for tool calling
5. [ ] Document project-specific system prompts
6. [ ] Get Idealista API access (request at developers.idealista.com)

---

## Search Provider Configuration

### Available Providers

| Provider | Use Case | How to Trigger |
|----------|----------|----------------|
| **Perplexity (default)** | General web search, news, events | Default for `web_search` |
| **Perplexity Pro** | Deep/thorough search with more citations | Say "use perplexity pro" or "deep search" |
| **Tavily** | Specific URLs, better for listings | Automatically used for `search_listings` |
| **Idealista** | Direct real estate listings (Spain/PT/IT) | Configure API keys, use `search_listings` with city |

### Tool Selection by Groq

```
User Query                          → Tool + Provider
"Barcelona rentals under €1400"     → search_listings (tavily or idealista)
"Latest news on AI"                 → web_search (perplexity)
"Deep search on climate change"     → web_search (perplexity_pro)
"Use perplexity pro for stocks"     → web_search (perplexity_pro)
"Research renewable energy trends"  → deep_research (perplexity pro)
```

### Idealista API Setup

1. Request access: https://developers.idealista.com/access-request
2. Add credentials to .env:
   ```
   IDEALISTA_API_KEY=your_key
   IDEALISTA_API_SECRET=your_secret
   ```
3. Groq will automatically use Idealista for Spain/Portugal/Italy real estate queries
