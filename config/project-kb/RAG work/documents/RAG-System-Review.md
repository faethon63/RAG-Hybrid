# RAG-Hybrid System Review (Living Document)

*Generated: 2026-02-06*
*Status: Comprehensive Review - v1.0*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Validated Needs Assessment](#2-validated-needs-assessment)
3. [Architecture Review](#3-architecture-review)
4. [Error Catalog](#4-error-catalog)
5. [Code Audit Findings](#5-code-audit-findings)
6. [Research Findings](#6-research-findings)
7. [Per-Project Optimization](#7-per-project-optimization)
8. [Prioritized Action Plan](#8-prioritized-action-plan)
9. [Appendices](#9-appendices)

---

## 1. Executive Summary

### System Health Scorecard

| Area | Rating | Key Finding |
|------|--------|-------------|
| Architecture | C+ | Functional monolith (10k+ LOC backend), multi-provider routing works but has 3 overlapping routing systems |
| Security | D | Auth disabled, CORS open to all origins, public VPS with no rate limiting |
| Reliability | C | 115+ exception handlers, 22 grade-F (dangerous silent failures), race conditions in concurrent requests |
| Cost Efficiency | B- | Groq (free) orchestration is smart, but duplicate API calls and unnecessary Claude fallbacks waste money |
| UX | B | Clean React frontend, good routing transparency, but inconsistent response quality across code paths |
| Maintainability | C- | 2796-line main.py monolith, significant code duplication, no automated tests |

### Top 5 Critical Issues

1. **Race condition in GroqAgent singleton** (`backend/groq_agent.py`): Mutable per-request state stored on a singleton object. Concurrent requests on VPS cross-contaminate project configs, conversation history, and KB searches. This is a **data leak vulnerability**.

2. **Three overlapping routing systems** (`query_classifier.py`, `orchestrator.py`, `groq_agent.py`): Each has its own classification logic, leading to misrouted queries, wasted API calls, and inconsistent behavior. Hardcoded product keywords (soap/cosmetics) in routing code don't generalize.

3. **22 dangerous silent exception handlers**: Critical operations (answer synthesis, KB context injection, health checks, ChromaDB operations, auth) silently swallow errors. Users get degraded answers with no indication of what went wrong.

4. **Public VPS with no authentication or rate limiting**: `rag.coopeverything.org` is accessible to anyone. CORS set to `*`. No rate limits. Any website can make API calls, potentially running up API costs or exfiltrating Notion/GitHub data.

5. **Deep Agent broken on VPS**: `AgentMemory` iteration error crashes step extraction (`deep_agent.py:387`), and the `/api/v1/search` endpoint doesn't exist, so local knowledge search always returns 404.

### Top 5 Recommended Improvements

1. **Fix GroqAgent concurrency** -- Use request-scoped state instead of singleton attributes. Pass context through function parameters, not mutable instance variables.

2. **Consolidate routing into a single pipeline** -- Merge `QueryClassifier`, `QueryOrchestrator`, and `GroqAgent._should_force_web_search()` into one clear decision tree with project-configurable rules instead of hardcoded keywords.

3. **Add structured error handling** -- Replace bare `except:` and `except Exception: pass` with specific exceptions and logging. Return user-friendly error messages with actionable information.

4. **Re-enable basic authentication** -- At minimum, add API key authentication for the VPS. Restrict CORS to known origins. Add rate limiting.

5. **Break up main.py** -- Extract tool handlers (Notion, GitHub, suppliers, file tools) into separate modules. Target: main.py under 500 lines, focused on endpoint definitions and request routing.

---

## 2. Validated Needs Assessment

### 2.1 System-Level Needs

#### Routing Reliability (CRITICAL)

The query routing pipeline has multiple layers of heuristic bypass logic that conflict:

- **Three overlapping systems**: `QueryClassifier` (regex in `backend/query_classifier.py`), `QueryOrchestrator._ask_groq()` (Groq LLM in `backend/orchestrator.py`), and `GroqAgent._should_force_web_search()` (hardcoded keyword triggers in `backend/groq_agent.py:497-547`).
- `_should_force_web_search()` has product-specific keywords like "isolate", "absolute", "terpene" -- tightly coupled to the Soap project, won't generalize.
- `groq_agent.py:572-742` contains a **170-line bypass path** that duplicates logic from the normal agent flow.
- `_detect_incomplete_response()` uses fragile heuristics (checking punctuation, counting pipe-delimited lines) causing false positives that trigger unnecessary Claude fallback calls.
- Follow-up detection (`groq_agent.py:510-518`) uses keyword matching ("did you", "is it") which misclassifies genuine new questions.

**Impact**: Misrouted queries cost money, produce hallucinated answers, or fail silently.

#### Error Handling & Resilience (CRITICAL)

- Error messages exposed to users: `"[Groq error: 400] {body}"` and `"[Claude error {status}: {body}]"` show raw API error text.
- `main.py:848` references undefined variable `tool_results`.
- No circuit breaker pattern: if Perplexity is down, every routed query times out at 90 seconds.
- No retry logic with backoff for transient API failures (429 rate limits, 503 service unavailable).

#### Cost Optimization (HIGH)

- **Duplicate Perplexity calls**: Forced web search bypass makes a Perplexity call, then Groq may call web_search again as a tool.
- **Claude fallback triggers too easily**: Short answers (query >100 chars, answer <150 chars) trigger Claude Haiku fallback even when the short answer is correct.
- **Auto-bump complexity**: Any project mentioning "financial/legal/medical" auto-bumps to Sonnet, even for simple formatting.
- **VPS cost**: Every query that would use free Ollama locally hits paid Claude on VPS.
- **Static max_tokens**: 2000 tokens for ALL calls regardless of query complexity.

#### Deployment Stability (HIGH)

- `git reset --hard origin/main` in deploy destroys any manual VPS-side changes.
- Health check is a single curl with `sleep 15`, no retry.
- No `__pycache__` clearing in deploy script.
- No rollback mechanism.

#### Security (HIGH)

- Auth fully disabled; VPS publicly accessible.
- CORS set to `*` -- any origin can call the API.
- `_tool_github_search` passes user input directly to subprocess arguments.
- No rate limiting on public VPS.
- Notion API key, GitHub tokens accessible via any cross-origin request.

### 2.2 Per-Project Needs

#### Chapter 7 Bankruptcy Assistant (CRITICAL)

- **34 PDF documents** in KB including tax returns, bankruptcy forms, bank statements.
- KB uses 500-character chunks with 50-char overlap -- too small for legal documents where provisions span paragraphs.
- PDF extraction via pypdf is text-only; form fields (checkboxes, tables) don't extract.
- System prompt has strict rules but no enforcement mechanism -- Groq decides which mode to use.
- Auto-bump detects "bankruptcy" keyword and bumps to Sonnet even for simple questions.

**Needs**: Larger chunks (1000-1500 chars), form-field-aware PDF extraction, strict hallucination guards for legal citations.

#### George's Barcelona Move (MEDIUM)

- No KB documents; relies entirely on web search.
- Idealista API may not be configured.
- Perplexity free tier rate limits during heavy research sessions with no backoff handling.

**Needs**: Graceful rate-limit handling, verified Idealista integration, project-specific KB for accumulated research.

#### Soap and Cosmetics (MEDIUM)

- Detailed naturalness analysis instructions in project config.
- Hardcoded cosmetic keywords in routing code.
- Procurement agent depends on Playwright browser binaries.
- No KB documents uploaded.

**Needs**: Reliable supplier tool on VPS, price tracking across sessions, ingredient/recipe KB.

#### RAG Work (LOW)

- One KB document (system documentation).
- Minimal system prompt.

**Needs**: Keep KB current; this review document will be added here.

### 2.3 UX Needs

#### Response Quality Consistency (CRITICAL)

Same query can go through 4+ different paths (bypass web search, Groq direct, Groq with tools, Claude fallback), each producing different formatting and quality. Groq passthrough strips citations; other paths don't. Claude passthrough only activates when Groq uses excuse phrases -- bad answers without excuses reach the user.

#### Transparency & User Control (HIGH)

- `RoutingInfo` only computed for auto mode; other modes don't populate it.
- No query cost display or cumulative session cost.
- No way to retry a query with a different model/mode.
- Mode selector UX is unclear.

#### Chat Persistence (MEDIUM)

- PostgreSQL sync works but chats without proper `project` field disappear from project view.
- Conversation history limited to 6 messages (hardcoded, undocumented).

### 2.4 Technical Debt

| Debt Item | Priority | Impact |
|-----------|----------|--------|
| Monolithic main.py (2796 lines) | HIGH | Hard to maintain, test, debug |
| Duplicated code (Notion blocks read 2x, Claude fallback 2x, citation stripping 2x) | HIGH | Bug fixes don't propagate |
| Hardcoded domain knowledge in routing | MEDIUM | Can't add projects without code changes |
| No automated test coverage | MEDIUM | Regressions from every change |
| Stale docstrings (reference old mode names, model IDs) | LOW | Developer confusion |
| Dead code (disabled auth, unused model constants) | LOW | Potential for accidental re-enablement |

---

## 3. Architecture Review

### 3.1 Component Map

```
RAG-Hybrid System Architecture (10,191 lines backend Python)
=============================================================

                    ┌──────────────────────────────────┐
                    │    React Frontend (Vite + TS)     │
                    │    - ChatContainer, ChatInput     │
                    │    - Sidebar + ProjectSelector    │
                    │    - SettingsPanel                │
                    │    State: Zustand (3 stores)      │
                    │    Styling: TailwindCSS           │
                    └──────────────┬───────────────────┘
                                   │ HTTP REST (no auth)
                                   │ CORS: * (open)
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (main.py)                     │
│                  2,796 lines -- MONOLITH                      │
│                                                                │
│  Endpoints:   /query, /health, /reload, /projects, /chats,    │
│               /search, /vision, /index, /settings, ...         │
│                                                                │
│  Tool Handlers: _tool_web_search, _tool_notion (~300 lines),  │
│                 _tool_github_search, _tool_search_listings,    │
│                 _tool_read_file, _tool_list_directory,         │
│                 _tool_deep_research, _tool_complex_reasoning,  │
│                 _tool_find_suppliers                            │
│                                                                │
│  Business Logic: combine_answers, query_vision,                │
│                  auto-bump complexity, project config loading   │
└──────────────────────┬───────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────────────────┐
       ▼               ▼                           ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐
│orchestrator │ │groq_agent   │ │ query_classifier          │
│ 219 lines   │ │ 1,094 lines │ │ 245 lines                │
│ Groq LLM    │ │ Agentic     │ │ Regex-based              │
│ routing     │ │ tool loop   │ │ intent detection          │
│ decisions   │ │ + bypass    │ │ (META, TASK, INFO, etc.) │
└──────┬──────┘ └──────┬──────┘ └──────────────────────────┘
       │               │
       └───────┬───────┘
               ▼
┌──────────────────────────────────────────────────┐
│         search_integrations.py (1,029 lines)      │
│  ClaudeSearch | PerplexitySearch | TavilySearch   │
│  IdealistaSearch                                   │
└──────────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┬────────────┐
    ▼          ▼          ▼            ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
│rag_core│ │deep_   │ │file_     │ │procurement_  │
│1,991 ln│ │agent   │ │tools     │ │agent         │
│ChromaDB│ │482 ln  │ │418 ln    │ │682 ln        │
│Ollama  │ │smol-   │ │Secure    │ │Playwright    │
│Embeds  │ │agents  │ │file ops  │ │browser auto  │
│Indexing│ │Code-   │ │Path val  │ │Supplier      │
│        │ │Agent   │ │          │ │scraping      │
└────────┘ └────────┘ └──────────┘ └──────────────┘

Supporting:
  config.py (190 ln) -- Central env var config
  auth.py (133 ln) -- JWT/bcrypt (DISABLED)
  pdf_tools.py (534 ln) -- PDF form operations
  supplier_db.py (378 ln) -- Supplier database
```

### 3.2 Data Flow

```
Query Flow (auto mode):
========================

1. Frontend POST /api/v1/query
   └─> main.py receives QueryRequest

2. Project Config Loading
   └─> Load system_prompt, instructions, allowed_paths
   └─> Auto-bump complexity for financial/legal/medical keywords

3. Query Classification (query_classifier.py)
   └─> Regex patterns → intent type (META, TASK, INFO, AUTOMATION...)

4. Groq Routing Decision (orchestrator.py)
   └─> Groq LLM analyzes query → route (LOCAL, PERPLEXITY, SONNET, OPUS)

5. BYPASS CHECK (groq_agent.py:497-547)
   ├─> If supplier keywords → forced web_search (skip Groq agent)
   ├─> If URL in query → forced web_search
   └─> If follow-up → check if supplier follow-up

6. GROQ AGENT LOOP (if not bypassed)
   ├─> Groq decides: use tool or respond
   ├─> Tool execution: web_search, complex_reasoning, etc.
   ├─> Passthrough checks:
   │   ├─> Perplexity-only → return Perplexity answer directly
   │   └─> Claude answer + Groq excuses → return Claude answer directly
   └─> Truncation detection → Claude Haiku completion

7. Response Assembly
   ├─> combine_answers() if both local + web results
   ├─> Add routing_info, tokens, cost estimate
   └─> Return QueryResponse to frontend

8. Chat Save
   └─> PostgreSQL (primary) or JSON file (fallback)
```

### 3.3 Dead Code & Stale References

| Item | Location | Status |
|------|----------|--------|
| Auth module (JWT, bcrypt, rate limiting) | `backend/auth.py` | Imported but disabled |
| Login endpoint | `main.py:1148-1157` | Exists but auth disabled |
| `cleanup_orphaned_collections()` | `rag_core.py` | Disabled per CLAUDE.md warning |
| Mode docstring "local/web/hybrid" | `main.py:1064-1068` | Documents modes that don't exist |
| "Llama 3.3 70B" references | `groq_agent.py:25`, `orchestrator.py:25` | Actually uses Llama 4 Scout 17B |
| `MODEL_DEEP_AGENT` constant | `orchestrator.py:33` | Defined but deep_agent routing is in main.py |
| IdealistaSearch initialization | `main.py:30,62` | Imported/initialized but API keys may not be configured |

### 3.4 Monolith Decomposition Proposal

**Current**: `main.py` (2,796 lines) contains endpoints, tool handlers, business logic, and integration code.

**Proposed Structure**:

```
backend/
  main.py              (~400 lines) -- FastAPI app, endpoint definitions, middleware
  tools/
    __init__.py
    web_search.py       -- _tool_web_search handler
    notion.py           -- _tool_notion (~300 lines of Notion block reading)
    github.py           -- _tool_github_search
    listings.py         -- _tool_search_listings (Idealista/Tavily)
    suppliers.py        -- _tool_find_suppliers
    reasoning.py        -- _tool_complex_reasoning, _tool_deep_research
  services/
    query_service.py    -- Query processing pipeline (combine_answers, query_vision)
    chat_service.py     -- Chat CRUD operations
    project_service.py  -- Project config management
    index_service.py    -- Document indexing
  middleware/
    auth.py             -- Authentication (when re-enabled)
    rate_limit.py       -- Rate limiting
    cors.py             -- CORS configuration
```

---

## 4. Error Catalog

### 4.1 Critical Errors (Crashes, Data Loss Risks)

#### Active Runtime: `AgentMemory` iteration crash
- **Location**: `backend/deep_agent.py:387` -- `for step in agent.memory`
- **Seen**: 2 times in VPS logs
- **Cause**: smolagents `CodeAgent.memory` is an `AgentMemory` object, NOT iterable
- **Impact**: Deep agent mode crashes after producing an answer. Step extraction fails.
- **Fix**: Use `agent.memory.steps` or equivalent smolagents API

#### Active: `/api/v1/search` returns 404
- **Location**: `backend/deep_agent.py:101-103`
- **Cause**: The `search_local_knowledge` tool calls `/api/v1/search` which was never implemented
- **Impact**: Deep agent local knowledge search ALWAYS fails on VPS. Agent produces plausible but ungrounded answers.

#### Bare `except:` catches SystemExit
- **Location**: `backend/main.py:335`
- **Issue**: Catches ALL exceptions including SystemExit and KeyboardInterrupt in base64 decode path
- **Fix**: Replace with `except (ValueError, UnicodeDecodeError)`

#### Undefined variable reference
- **Location**: `backend/main.py:848`
- **Issue**: References `tool_results` in `if not choices` block -- variable may not be defined at that point
- **Fix**: Ensure variable is initialized before the conditional

### 4.2 Silent Failures

| Location | What fails silently | User impact |
|----------|-------------------|-------------|
| `main.py:1934` | Claude synthesis in `combine_answers()` | Falls to Ollama without logging; answer quality degrades invisibly |
| `main.py:262` | KB context injection for complex_reasoning | Answer lacks KB data; user gets worse answer with no indication |
| `rag_core.py:102` | Ollama health check | Network issues appear as "Ollama not installed" |
| `rag_core.py:111` | ChromaDB health check | Connection errors swallowed; dashboard shows healthy |
| `rag_core.py:867` | Collection count during project listing | Corrupted collections silently masked |
| `rag_core.py:905` | Collection deletion | Caller thinks collection deleted when it wasn't |
| `rag_core.py:983` | Collection delete during overwrite | Could cause duplicate data on re-import |
| `auth.py:61` | Password verification | bcrypt bug would lock ALL users out with no logging |
| `procurement_agent.py` (x12) | Various browser automation steps | Price extraction, JSON-LD parsing, selector interaction |

### 4.3 Routing Errors

1. **Supplier query detection** (`groq_agent.py:570-640`): Complex regex-based detection. Follow-up questions misrouted. Pattern `"find " + noun` triggers supplier search. Still has false-positive-prone patterns.

2. **Research mode key inconsistency** (`main.py:1580-1610`): Claude path uses `result["sources"]`, Perplexity path uses `result["citations"]` -- inconsistent keys could cause KeyError.

3. **Groq bypass triple-condition** (`groq_agent.py:563-640`): Three separate bypass conditions (forced supplier web_search, forced URL web_search, supplier-follow-up check) with complex interactions.

### 4.4 Hallucination Pathways

1. **Groq paraphrasing tool results** (`groq_agent.py:928-1016`): After tools return data, Groq can fabricate data in its summary. Only web_search-only and complex_reasoning have passthrough protections.

2. **`combine_answers()` synthesis** (`main.py:1920-1928`): LLM can introduce fabricated connections ("both sources agree") or misrepresent sources.

3. **Deep agent `final_answer`** (`deep_agent.py:382`): Agent can fabricate final answer when tools return errors. Seen in VPS logs -- search returned 404 but agent produced plausible answer.

4. **Error messages as answers** (`groq_agent.py:1070`): Groq API error body returned as user-facing "answer".

### 4.5 Race Conditions

**CRITICAL: GroqAgent Singleton State**

`groq_agent.py` creates a global `groq_agent = GroqAgent()` with mutable request state:
- `_current_project`
- `_current_project_config`
- `_current_conversation_history`

These are set at request start (`groq_agent.py:563-565`). If concurrent requests arrive:
- Request A's `complex_reasoning` reads Request B's project config
- Request A's KB search queries Request B's project
- Request A's conversation history shows Request B's messages

**Impact**: Cross-contamination between concurrent VPS users. One user could see another's project data.

**Additional race conditions**:
- `httpx.AsyncClient` lazy init has no lock -- concurrent requests could create two clients
- ChromaDB concurrent writes have no locking -- parallel indexing could create duplicates

---

## 5. Code Audit Findings

### 5.1 Frontend-Backend Contract Audit

#### Perfect Matches
- QueryRequest contract: Frontend and backend perfectly aligned
- QueryResponse contract: All fields match
- AttachedFile contract: Field names and types match
- Chat save/load: Project field properly included
- Routing transparency: Full tool/model details exposed

#### Critical Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **Vision model override**: Frontend forces `model='local'` for images, preventing Claude's superior vision | `chatStore.ts:206` | Worse image analysis quality |
| 2 | **Backend docs outdated**: Docstring documents "local/web/hybrid" modes but code uses "auto/private/research/deep_agent" | `main.py:1064-1068` | Developer confusion |
| 3 | **Error details hidden**: `ApiError.body` contains backend detail but frontend shows generic "Query failed" | `chatStore.ts:262` | Users can't diagnose issues |

#### Medium Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 4 | Settings default sync: Only saves on explicit "Save" click | `SettingsPanel.tsx` | Unsaved changes lost |
| 5 | Conversation history limit undocumented: Hardcoded 6 messages | `chatStore.ts:199` | No explanation for context truncation |
| 6 | Project validation missing: Backend doesn't verify project exists | `main.py:1162` | 500 errors instead of 404 |
| 7 | Model options gap: Backend accepts Claude model IDs directly but frontend only offers auto/local | `settingsStore.ts` | Power users can't select specific models |

### 5.2 Error Handling Quality

**Exception Handler Statistics (115+ handlers analyzed)**:

| Grade | Count | % | Criteria |
|-------|-------|---|----------|
| A (Proper) | 8 | 7% | Catches specific exceptions, logs appropriately |
| B (Acceptable) | 15 | 13% | Catches broad but handles reasonably |
| C (Mediocre) | 18 | 15% | Catches broad, logs but doesn't handle well |
| D (Poor) | 12 | 10% | Catches broad, minimal logging |
| F (Dangerous) | 22 | 19% | Silently swallows errors |

**Best pattern in codebase**: `file_tools.py:82-97` -- Three-level exception hierarchy (PathSecurityError vs ValueError vs Exception) with appropriate handling per type.

**Worst patterns**: 42 exception handlers in `procurement_agent.py` alone, mostly grade D-F.

### 5.3 Security Issues

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| No authentication | HIGH | System-wide | VPS publicly accessible, anyone can use it |
| CORS wildcard | HIGH | `main.py:47-54` | Any origin can call the API |
| No rate limiting | HIGH | System-wide | Cost abuse possible |
| Subprocess with user input | MEDIUM | `main.py:291-368` | GitHub search passes query to `gh` CLI args |
| API keys accessible | MEDIUM | System-wide | CORS `*` + no auth = any website can proxy through the API |
| Raw error exposure | LOW | `groq_agent.py:1070` | Internal error text reaches users |

### 5.4 Code Duplication

| Duplicated Code | Location A | Location B | Lines |
|----------------|------------|------------|-------|
| `read_blocks_recursive` (Notion) | `main.py:~605` (find_info) | `main.py:~699` (read_page) | ~40 lines each |
| Claude Haiku fallback | `groq_agent.py:699-729` (bypass) | `groq_agent.py:991-1006` (main) | ~30 lines each |
| Raw content detection | `groq_agent.py:654-658` (bypass) | `groq_agent.py:950-956` (main) | ~10 lines each |
| Citation stripping | `groq_agent.py:18-21` | Multiple inline regex uses | ~5 lines each |
| Ollama health + generation | `rag_core.py:95-103` | `groq_agent.py:670-694` | ~25 lines each |

### 5.5 Model ID Consistency

Model IDs are hardcoded across multiple files:

| Model | Locations | Risk |
|-------|-----------|------|
| `claude-haiku-4-5-20251001` | `groq_agent.py:713`, `deep_agent.py:183`, `search_integrations.py`, `main.py` | Next rename breaks all |
| `claude-sonnet-4-5-20250929` | `search_integrations.py`, `main.py` (vision) | Same |
| `claude-opus-4-5-20251101` | `search_integrations.py` | Same |
| `meta-llama/llama-4-scout-17b-16e-instruct` | `groq_agent.py:100`, `orchestrator.py` | Same |

**Recommendation**: Define model constants in `config.py` and import everywhere. Already have `CLAUDE_SONNET_MODEL` etc. in env vars but not all code uses them.

---

## 6. Research Findings

### 6.1 Framework Comparisons

The current system is a **custom-built orchestration framework**. Industry alternatives include:

| Framework | Pros | Cons | Relevance |
|-----------|------|------|-----------|
| **LangChain** | Mature ecosystem, many integrations, structured chains | Heavy dependency, over-abstracted for simple cases, vendor lock-in | Could replace tool-calling loop |
| **LlamaIndex** | Strong RAG focus, advanced indexing strategies | Less flexible for multi-model routing | Could improve KB/chunking |
| **Semantic Kernel** | Microsoft-backed, good multi-model support | .NET-centric, Python support newer | Not ideal fit |
| **Custom (current)** | Full control, no dependencies, tailored to needs | Maintenance burden, no community testing | Already in place |

**Recommendation**: Keep custom approach but adopt specific patterns from frameworks:
- LlamaIndex's hierarchical chunking for legal documents (Chapter 7)
- LangChain's tool-calling protocol for standardized tool interfaces
- Circuit breaker pattern from resilience libraries (tenacity, circuitbreaker)

### 6.2 Routing Best Practices

**Current problem**: Three overlapping routing systems with hardcoded heuristics.

**Industry best practices**:
1. **Single routing decision point**: One classifier that produces a structured routing plan
2. **Configurable routing rules**: Per-project routing preferences in config, not code
3. **Fallback chain with clear priority**: PRIMARY -> SECONDARY -> FALLBACK with explicit conditions
4. **Routing telemetry**: Log every routing decision with reasoning for debugging

**Recommended architecture**:
```
Query → Single Router (config-driven)
  ├─> Rules engine (project config defines preferred providers)
  ├─> LLM classifier (Groq) for ambiguous queries
  └─> Fallback chain: Groq → Ollama → Claude Haiku
```

### 6.3 KB Management Recommendations

**Current issues**:
- Fixed 500-char chunks too small for legal docs
- No per-project chunk size configuration
- No hybrid search (semantic + keyword)

**Recommendations**:
1. **Per-project chunking strategy**: Legal docs (1000-1500 chars), general docs (500 chars), code (function-level)
2. **Hybrid search**: Combine ChromaDB vector search with BM25 keyword search for better recall
3. **Metadata enrichment**: Store document type, date, section headings in chunk metadata
4. **Re-ranking**: Add a lightweight re-ranker (cross-encoder) before returning top results

### 6.4 Hallucination Prevention

**Current mitigations**: Perplexity passthrough, Claude passthrough on excuses.

**Additional techniques**:
1. **Grounding verification**: After Groq summarizes tool results, verify key facts (numbers, names, dates) appear verbatim in tool output
2. **Citation enforcement**: Require inline citations [1], [2] in responses; validate they reference actual sources
3. **Confidence scoring**: Have the model self-assess confidence; flag low-confidence answers for user review
4. **Fact extraction + comparison**: Extract claims from response, compare against tool results

### 6.5 Resilience Patterns

**Current gaps**: No circuit breakers, no retry with backoff, no health-aware routing.

**Recommended patterns**:

1. **Circuit Breaker** (for each external API):
   ```
   States: CLOSED (normal) → OPEN (failing, skip calls) → HALF-OPEN (test one call)
   Track: failure count, last failure time, consecutive successes
   ```

2. **Retry with Exponential Backoff** (for transient failures):
   ```
   Attempt 1: immediate
   Attempt 2: 1s delay
   Attempt 3: 4s delay
   Max: 3 attempts
   Only retry: 429, 503, 502
   ```

3. **Health-Aware Routing**:
   ```
   Before routing to a provider, check circuit breaker state
   If OPEN, skip to next provider in fallback chain
   ```

4. **Timeout Budgets**:
   ```
   Total query timeout: 30s
   Per-tool timeout: 15s
   If tool exceeds budget, cancel and use fallback
   ```

---

## 7. Per-Project Optimization

### 7.1 Chapter 7 Bankruptcy Assistant

**Current state**: 34 KB documents, strict system prompt, PDF tools available.

**Issues identified**:
- 500-char chunks split legal provisions across chunks (needs assessment)
- pypdf text-only extraction misses form fields (needs assessment)
- Auto-bump on "bankruptcy" keyword wastes Sonnet tokens on simple questions (needs assessment)
- No test coverage for KB search accuracy against legal questions (needs assessment)

**Recommended optimizations**:
1. **Increase chunk size to 1500 chars** with 200-char overlap for legal documents
2. **Add pypdf form field extraction** alongside text extraction
3. **Smart auto-bump**: Only bump to Sonnet when query actually requires complex reasoning, not just keyword presence
4. **Hallucination guard**: For any response citing form instructions, verify the citation exists in indexed KB
5. **Test suite**: Create 10-20 known-answer bankruptcy questions to regression-test KB search quality

### 7.2 George's Barcelona Move

**Current state**: No KB, relies on web search, Idealista integration.

**Issues identified**:
- No rate-limit handling for Perplexity (needs assessment)
- Idealista API configuration unknown (needs assessment)
- No accumulated research storage (needs assessment)

**Recommended optimizations**:
1. **Add Perplexity rate-limit backoff**: Detect 429 responses, wait and retry
2. **Verify Idealista API keys**: Check config, test a sample query
3. **Auto-KB accumulation**: Save key research findings (visa rules, banking requirements) to project KB for future reference
4. **Location-aware search**: Default Perplexity searches to Barcelona/Spain context

### 7.3 Soap and Cosmetics

**Current state**: Detailed instructions for naturalness analysis, hardcoded product keywords in routing.

**Issues identified**:
- Product keywords in routing code won't generalize (needs assessment)
- Procurement agent needs Playwright on VPS (needs assessment)
- No ingredient/recipe KB (needs assessment)

**Recommended optimizations**:
1. **Move product keywords to project config**: Let each project define its domain-specific search triggers
2. **Verify Playwright on VPS**: Check if browser binaries are installed, add to deploy if not
3. **Create ingredient KB**: Index common ingredients, their properties, and suppliers
4. **Price tracking**: Store supplier prices with timestamps for comparison across sessions

### 7.4 RAG Work (Meta/System)

**Current state**: One KB document, minimal config.

**Recommended optimizations**:
1. **Add this review document** to KB
2. **Add CLAUDE.md** content to KB for self-referential system queries
3. **Development log**: Track significant changes for future reviews

---

## 8. Prioritized Action Plan

### Immediate (This Week)

| # | Action | Priority | Effort | Impact |
|---|--------|----------|--------|--------|
| 1 | **Fix GroqAgent singleton race condition**: Pass project/config/history as function parameters instead of mutable instance state | CRITICAL | Medium | Prevents data leaks between concurrent users |
| 2 | **Fix deep_agent.py:387**: Use `agent.memory.steps` instead of iterating `agent.memory` | CRITICAL | Low | Unbreaks Deep Agent mode on VPS |
| 3 | **Fix or remove `/api/v1/search` reference**: Either implement the endpoint or remove it from deep_agent | CRITICAL | Low | Fixes local knowledge search in Deep Agent |
| 4 | **Replace bare `except:` at main.py:335**: Use `except (ValueError, UnicodeDecodeError)` | CRITICAL | Low | Prevents catching SystemExit/KeyboardInterrupt |
| 5 | **Fix vision model override**: Remove `chatStore.ts:206` that forces `model='local'` for images | HIGH | Low | Enables Claude's superior vision on VPS |
| 6 | **Fix undefined `tool_results` at main.py:848** | HIGH | Low | Prevents potential NameError |

### Short-Term (This Month)

| # | Action | Priority | Effort | Impact |
|---|--------|----------|--------|--------|
| 7 | **Add basic auth to VPS**: API key authentication, restrict CORS to known origins | HIGH | Medium | Prevents public abuse |
| 8 | **Add logging to silent exception handlers**: Especially `main.py:1934`, `rag_core.py:102,111`, `auth.py:61` | HIGH | Medium | Makes failures diagnosable |
| 9 | **Centralize model IDs in config.py**: Define constants, import everywhere, use env var overrides | HIGH | Medium | Prevents model ID drift |
| 10 | **Add circuit breaker for external APIs**: At minimum for Perplexity and Groq | HIGH | Medium | Prevents cascading timeouts |
| 11 | **Update stale docstrings**: Mode names, model references, architecture comments | MEDIUM | Low | Reduces developer confusion |
| 12 | **Add __pycache__ clearing to deploy script** | MEDIUM | Low | Prevents stale bytecode issues |

### Medium-Term (Next Quarter)

| # | Action | Priority | Effort | Impact |
|---|--------|----------|--------|--------|
| 13 | **Break up main.py**: Extract tool handlers into `tools/` module, services into `services/` | HIGH | High | Maintainability, testability |
| 14 | **Consolidate routing**: Merge three routing systems into single config-driven pipeline | HIGH | High | Eliminates misrouting, removes hardcoded keywords |
| 15 | **Add retry with backoff for transient API failures** | HIGH | Medium | Better reliability under API instability |
| 16 | **Per-project chunk size for KB**: Legal docs 1500 chars, general 500 chars | MEDIUM | Medium | Better Chapter 7 search quality |
| 17 | **Deduplicate code**: Notion block reading, Claude fallback, citation stripping | MEDIUM | Medium | Bug fixes propagate correctly |
| 18 | **Move domain keywords to project config** | MEDIUM | Medium | Projects configurable without code changes |
| 19 | **Add user-friendly error messages**: Replace raw API errors with structured, actionable messages | MEDIUM | Medium | Better user experience |
| 20 | **Frontend: Show error details from ApiError.body** | MEDIUM | Low | Users can diagnose issues |

### Long-Term (Future)

| # | Action | Priority | Effort | Impact |
|---|--------|----------|--------|--------|
| 21 | **Add automated test suite**: Unit tests for routing, tool handlers, error paths | MEDIUM | High | Confidence in changes, catch regressions |
| 22 | **Hybrid search (vector + keyword)** for KB | MEDIUM | High | Better recall for factual queries |
| 23 | **Hallucination verification**: Post-response fact checking against tool outputs | MEDIUM | High | Higher answer accuracy |
| 24 | **Cost tracking dashboard**: Aggregate per-query costs, daily/monthly totals | LOW | Medium | Budget awareness |
| 25 | **Migrate `on_event` to lifespan context manager** | LOW | Low | Future FastAPI compatibility |
| 26 | **Re-ranker for KB search results** | LOW | Medium | Better precision for top results |
| 27 | **Deploy rollback mechanism** | LOW | Medium | Quick recovery from bad deploys |
| 28 | **Health-aware routing**: Skip providers with open circuit breakers | LOW | Medium | Smarter failover |

---

## 9. Appendices

### Appendix A: Full Error Handler Inventory

#### Grade A (Proper -- 8 handlers)

| Location | Type | Description |
|----------|------|-------------|
| `main.py:167-172` | `ValueError` (x2) | Type coercion for search_listings parameters |
| `main.py:1805-1818` | `httpx.HTTPStatusError` | Vision API HTTP errors, then general re-raise |
| `file_tools.py:82-97` | `PathSecurityError/ValueError/Exception` | Three-level hierarchy, best pattern |
| `groq_agent.py:1018` | `httpx.HTTPStatusError` | Specific HTTP error with body parsing |
| `search_integrations.py:959,1012` | `ImportError` | Optional dependency checks (Crawl4AI) |

#### Grade B (Acceptable -- 15 handlers)

| Location | Type | Description |
|----------|------|-------------|
| `main.py:366` | `Exception` | GitHub tool error with user-readable message |
| `main.py:856` | `Exception` | Notion tool error with context |
| `main.py:895-903` | `ImportError/Exception` | Procurement fallback to web_search |
| `groq_agent.py:918` | `Exception` | Tool execution failure with clear fallback |
| `groq_agent.py:1018-1067` | `HTTPStatusError` | Recovery from `failed_generation` errors |
| `file_tools.py:141-178` | `PathSecurityError/Exception` | Security exceptions separated from general |
| `search_integrations.py:694-699` | `HTTPStatusError/Exception` | Tavily extract structured error |
| `pdf_tools.py:85-87` | `ImportError` | PDF import graceful handling |

#### Grade C (Mediocre -- 18 handlers)

Key locations: `main.py:116,262`, `groq_agent.py:695,728`, `deep_agent.py:53,82,118,165,192`, `rag_core.py:1069-1092`

#### Grade D (Poor -- 12 handlers)

Key locations: `main.py:1466,1989,2076`, `groq_agent.py:1075`, `supplier_db.py:131`

#### Grade F (Dangerous -- 22 handlers)

Key locations: `main.py:335,1934`, `rag_core.py:102,111,867,905,983`, `auth.py:61`, `procurement_agent.py` (12 handlers)

### Appendix B: Git History Analysis

**Recurring bug patterns found in git history:**

| Pattern | Times Fixed | Root Cause | Current Risk |
|---------|------------|------------|-------------|
| Hallucination | 3 | Groq paraphrasing tool results | MEDIUM-HIGH |
| Routing errors | 4 | Regex matching too broadly/narrowly | MEDIUM |
| Missing VPS fallback | 2 | Code assumed Ollama available | LOW (fixed for known paths) |
| Model ID errors | 2 | Anthropic renamed models, code had old IDs | LOW (but will recur) |
| Content validation | 1 | `all()` vs `any()` logic error | LOW |

**Key commits analyzed:**
- `c3701ad` -- Hallucination fix: strengthened system prompt
- `515b467` -- Perplexity passthrough bypass
- `db800f3` -- Fixed `combine_answers()` fabrication
- `1f34a39` -- Follow-up question misrouting fix
- `3fb9746` -- Model-specific bypass fix
- `83186a1` -- Indentation scope error
- `212435d` -- Multiple critical bug fixes (model IDs, content validation, VPS fallback)
- `d9b3511` -- VPS fallback additions

### Appendix C: Key File Reference

| File | Lines | Purpose | Health |
|------|-------|---------|--------|
| `backend/main.py` | 2,796 | API endpoints, tool handlers, business logic | Needs decomposition |
| `backend/groq_agent.py` | 1,094 | Agentic tool-calling loop with bypass paths | Complex, has race condition |
| `backend/rag_core.py` | 1,991 | ChromaDB, embeddings, Ollama, indexing | Silent failures in health checks |
| `backend/search_integrations.py` | 1,029 | Claude, Perplexity, Tavily, Idealista APIs | Generally good error handling |
| `backend/procurement_agent.py` | 682 | Playwright browser automation | 12 silent exception handlers |
| `backend/pdf_tools.py` | 534 | PDF form operations | Good error handling |
| `backend/deep_agent.py` | 482 | smolagents CodeAgent | Broken: memory iteration + missing endpoint |
| `backend/file_tools.py` | 418 | Secure file operations | Best error handling pattern |
| `backend/supplier_db.py` | 378 | Supplier database | Silent timestamp parsing failure |
| `backend/query_classifier.py` | 245 | Regex-based intent classification | Functional but overlaps with other classifiers |
| `backend/orchestrator.py` | 219 | Groq LLM routing decisions | Stale comments, overlaps with groq_agent |
| `backend/config.py` | 190 | Central env var configuration | Good, lazy loading pattern |
| `backend/auth.py` | 133 | JWT, bcrypt, rate limiting | Disabled, silent password check failure |
| `frontend-react/src/stores/chatStore.ts` | -- | Chat state management | Vision model override bug |
| `frontend-react/src/stores/settingsStore.ts` | -- | Settings + health polling | Good |
| `frontend-react/src/stores/projectStore.ts` | -- | Project state | Good |
| `frontend-react/src/api/client.ts` | -- | API client | Error body not shown to user |

---

*This is a living document. It will be updated as issues are resolved and new findings emerge.*

*Last updated: 2026-02-06*
*Generated by: RAG-Hybrid System Review Team*
