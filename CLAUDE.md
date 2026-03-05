# RAG-Hybrid Project

## What This Is

A hybrid Retrieval-Augmented Generation system with **smart auto-routing**:

- **GEMINI ORCHESTRATOR** - Gemini 2.5 Flash routes queries (FREE, Groq as fallback)
- **LOCAL (Ollama)** - Free, private, fast. Used for text generation
- **PERPLEXITY** - Real-time web search. Used for current events, prices, news
- **CLAUDE (Sonnet/Opus)** - Paid. Only used for complex reasoning

## Second Brain Features (Added 2026-03-03)

- **User Memory** (`config/project-kb/_user_memory/`): Persistent personal memory files (soul.md, user.md, interests.md, memory.md) auto-injected into all conversations. Groq has `update_user_memory` tool to auto-update when user shares personal info.
- **PWA + Push Notifications**: Frontend is a PWA (installable on phone). Backend manages VAPID keys and push subscriptions via `backend/notifications.py`.
- **Heartbeat** (`backend/heartbeat.py`): Background APScheduler service — periodic check (every 30min) + daily briefing (9AM). Uses Groq (free) for decisions, Perplexity for news, sends push notifications.
- **API Endpoints**: `/api/v1/memory/*`, `/api/v1/notifications/*`, `/api/v1/heartbeat/*`

## Architecture (Updated 2026-01-31)

When a project has `allowed_paths` (file access), Groq orchestrates tool use:
1. Groq receives query + available tools
2. Groq decides: use tool OR respond OR delegate
3. If tool: execute, send result back to Groq
4. Loop until Groq says "respond"
5. Ollama generates final text response (free)

Switched from Claude Haiku to Groq for cost savings (Groq is FREE) and better tool routing.

## Modes (Simplified)

- **auto** (default) - Orchestrator routes to cheapest capable model
- **private** - Local only, no external APIs (offline-safe)
- **research** - Deep Perplexity search

## Tech Stack

- **Backend:** FastAPI (localhost:8000) at `backend/main.py`
- **Frontend:** React + Vite (localhost:5173) at `frontend-react/`
- **Vector DB:** ChromaDB at `data/chromadb/`
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Local LLM:** Ollama at `G:\AI-Project\Ollama\` with qwen2.5:14b (text generation)
- **Vision/OCR:** Claude Sonnet 4.5 (local models hallucinate document text)
- **Auth:** JWT login disabled. API key middleware active on VPS (requires `X-API-Key` header when `API_KEY` env var is set)
- **Python:** 3.12.10, venv at `.venv/`
- **PDF Support:** Requires `pypdf` package

## Project Layout

```
RAG-Hybrid/
  backend/
    main.py               # FastAPI server, all endpoints
    config.py             # Central config with lazy env var loading
    auth.py               # JWT verify, bcrypt, rate limiter
    rag_core.py           # ChromaDB, embeddings, Ollama, indexing
    search_integrations.py # Claude, Perplexity (free), Tavily APIs
    orchestrator.py       # Query analysis, model selection
    file_tools.py         # Secure file operations for projects
    notifications.py      # Push notification service (VAPID, subscriptions)
    heartbeat.py          # Proactive heartbeat (APScheduler, daily briefing)
  frontend-react/          # React frontend
    src/
      components/         # React components (chat, sidebar, settings)
      stores/             # Zustand state management
      api/                # API client
    package.json          # npm dependencies
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

## Local vs VPS Sync

**What DOES sync (via git push to main):**
- Project configs: name, description, system_prompt, flexible_prompt, instructions
  - Stored in `config/projects/{name}.json` (tracked in git)
- Uploaded KB documents (via the file upload UI)
  - Stored in `config/project-kb/{name}/documents/` (tracked in git)

**What does NOT sync (environment-specific):**
- `allowed_paths` - External file paths differ between Windows and Linux
- `indexed_files` - Tracks what's indexed locally (ChromaDB is local)
- Ollama - Only available locally (VPS has no GPU)

**Chat Storage (DOES sync via PostgreSQL):**
- Chats are stored in PostgreSQL database on VPS (`DATABASE_URL` in .env)
- Both local and VPS connect to the same database, so chats sync automatically
- JSON files in `data/chats/` are fallback only (if DATABASE_URL not set)
- To query chats: `ssh root@72.60.27.167 "sudo -u postgres psql -d rag_hybrid -c 'SELECT id, title FROM chats ORDER BY updated_at DESC LIMIT 10'"`

## WARNINGS - DO NOT DO THESE

1. **NEVER auto-delete ChromaDB collections on startup** - This caused data loss. The `cleanup_orphaned_collections()` function is disabled for this reason.
2. **NEVER add cleanup/purge logic that runs automatically** - User data in `data/chromadb/` and `data/project-kb/` must persist.
3. **Project configs are split:** synced in `config/projects/{name}.json`, local in `data/project-kb/{name}/local.json` - Don't delete either.
4. **JWT auth is disabled, but API key middleware is active on VPS** - VPS `.env` has `API_KEY` set, so all API requests (except `/health`, `/docs`) require `X-API-Key` header. Local dev without `API_KEY` env var skips auth entirely.
5. **NEVER use Write tool to replace entire files** - Use Edit tool with targeted changes instead. Write tool destroys uncommitted user modifications.
6. **ALWAYS Read a file before modifying it** - Even for "simple" changes. The file may have local modifications not in git.
7. **When making large changes, use multiple small Edits** - Not one big Write. This preserves existing code the user may have customized.
8. **ALWAYS push changes to deploy to VPS** - After making frontend or backend changes, commit and push to main. GitHub Actions auto-deploys to VPS. Local-only changes are useless if the user expects them on VPS too.
9. **NEVER implement a feature without first verifying it doesn't already exist** - This project has had numerous sessions. Features like push notifications, heartbeat, user memory, file editor, and form filling are ALREADY BUILT. Before writing any code, use Glob/Read to check if the relevant files exist and contain the feature logic. Read MEMORY.md's "Completed Features Registry" first. Re-implementing existing features wastes time and may overwrite working code.
10. **This system is BUILT — you are in MAINTENANCE mode.** Your job is bug fixes and incremental improvements. When you see a feature description (in plans, summaries, or messages), your FIRST response must be "does this exist?" not "let me build this." If you find yourself creating new files for features listed in the Completed Features Registry, STOP — you are re-implementing.

## User Profile

**The user does not code.** They "vibe-code" with Claude - describing what they want and having Claude implement it. This means:

1. **Automate everything possible** - Don't ask the user to run commands if you can run them yourself
2. **When automation isn't possible, provide COMPLETE step-by-step instructions:**
   - State what application to open (e.g., "Open PowerShell as Administrator")
   - State what directory to be in FIRST (e.g., "Make sure you're in G:\AI-Project\RAG-Hybrid")
   - Provide the exact command to paste
   - Explain what success looks like
   - Explain what to do if it fails
3. **Never assume technical knowledge** - Explain what things mean
4. **Test everything yourself** before saying it's done
5. **Check logs and errors yourself** instead of asking the user to debug
6. **Terminal limitations:** Cannot control Windows Terminal tabs or close windows programmatically. Prefer background processes to avoid window clutter.
7. **ALWAYS start/restart services yourself** - If the backend or frontend needs to be started or restarted, DO IT. Never say "start the backend" and wait for the user. Use PowerShell via Bash tool to start services in background.
8. **NEVER ask permission for obvious next steps** - If you just changed backend code, restart the backend. If you changed frontend code, the frontend auto-reloads. If a task has an obvious continuation, just do it. Asking "want me to restart?" or "shall I start the frontend?" wastes time and adds no value.

## Claude Instructions

When telling user to run PowerShell commands:
1. Always specify to run from `G:\AI-Project\RAG-Hybrid`
2. **For pip: this venv needs `ensurepip` first** (see Installing Python Packages section)
3. For python: use `.\.venv\Scripts\python.exe` or activate venv first
4. Include full command sequence if venv activation is needed
5. **NEVER chain assumptions** - if first command fails, don't assume the "fix" will work either
6. **When uncertain, ask user to test** a verification command first before giving the full solution
7. **DO NOT ask user to restart services** - Start/restart backend and frontend yourself using Bash tool
8. **Check chat history for errors** - When something goes wrong, check the last chat via API instead of asking user to debug
9. **ALWAYS test after making changes** - Verify the change works before reporting completion
10. **NEVER spam PowerShell windows** - Each `powershell.exe -Command` from bash opens a new window. Combine multiple checks into single commands, or use background processes. Don't run dozens of separate PS commands that leave windows open.
11. **Delete pycache when changes don't work** - Python caches bytecode in `__pycache__` folders. If code changes aren't taking effect despite restarts, delete `backend/__pycache__` first.

## Development Workflow (CRITICAL)

When making backend changes, follow this exact sequence:

1. **Make the code change** using Edit tool
2. **Wait for watchfiles reload** - Check health endpoint passes
3. **Delete pycache** - `rm -rf backend/__pycache__`
4. **Test via API** - Run test query, save results to user's active project (so tests are VISIBLE in UI)
5. **CHECK LOGS** - Verify new code path executed (look for expected log messages)
   - If logs confirm new code ran → proceed
   - If logs show old behavior or no new messages → restart cleanly, retest
6. **User reviews visible test results** - Don't ask user to "test it yourself", they see the test chats you created

**Why log verification matters:** Watchfiles auto-reload can fail silently or have race conditions. Always check logs to confirm the new code path actually executed. If expected log messages are missing, the fix didn't deploy.

**When to force restart:** Only if logs show the new code didn't run:
- Delete `backend/__pycache__/`
- Kill Python completely: `Stop-Process -Name python -Force`
- Start fresh
- Retest

**Test chat visibility:** When saving test chats via API, ALWAYS include the `project` field matching the user's current view. Chats with `project: null` won't appear if user has a project filter active. Verify chats appear via API before telling user to look.

**Testing from WSL:** Use PowerShell for API calls since WSL curl can't reach Windows localhost:
```bash
powershell.exe -Command 'Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query" -Method POST -Body $body -ContentType "application/json"'
```

## Mandatory Verification Protocol (NEVER SKIP)

**NEVER declare a change "done" until it is verified working on VPS.** Pushing code is not delivery. A passing deploy is not delivery. Only a verified end-to-end test is delivery.

### After Every Push

1. **Wait for deploy** - `gh run list --repo faethon63/RAG-Hybrid --limit 1` must show `success`
2. **Verify code landed** - SSH to VPS and confirm the change is present (grep for new code, check git log)
3. **Run end-to-end test on VPS** - not locally, ON THE VPS where the user will see it
4. **Check VPS logs** - `ssh root@72.60.27.167 "pm2 logs rag-backend --lines 30 --nostream"` to confirm new code path executed
5. **Only then report to user**

### Verification by Change Type

| Change Type | How to Verify |
|-------------|--------------|
| **Frontend display** | Check built JS contains new code + test the API that feeds data to it |
| **Backend routing/logic** | Send test query via VPS API, check response content AND logs |
| **Data persistence** | Write via API, read back from DB (`sudo -u postgres psql`) |
| **Tool/agent behavior** | Send query that triggers the tool, verify tool appears in logs and response metadata |

### What "Verify" Means (Examples)

- Timestamp display: Check that `message.timestamp` exists in DB after saving a chat, not just that the UI code renders it
- Groq tool routing: Send a Ch7 query via VPS, check logs show `get_data_profile` was called, check response contains actual numbers
- Any bug fix: Reproduce the original bug scenario on VPS and confirm it no longer occurs

### When Verification Fails

Do NOT report partial success. Fix the issue, push again, and re-verify. The user should only hear about a change when it is confirmed working.

### Agent Teams for Complex Changes

For multi-step changes that touch frontend + backend + database, use the Task tool to spawn an agent team:
- **Coordinator** (main agent): Plans the change, holds full conversation context, assigns tasks
- **Coder**: Makes targeted file edits
- **Tester**: After deploy, runs VPS verification (API calls, log checks, DB queries)

The coordinator bridges context between agents. Use teams when a change requires 3+ files or crosses frontend/backend/DB boundaries. For simple single-file edits, self-test is sufficient.

## Running (Windows)

### Quick Start (Recommended)

Run from PowerShell:
```powershell
.\start.ps1
```
Or double-click `restart-all.bat`.

This runs everything in a **single window**:
- Backend starts in background (logs to `logs/backend.log`)
- Frontend runs in foreground (shows output)
- Press `Ctrl+C` to stop both

**Options:**
```powershell
.\start.ps1              # Start both (default)
.\start.ps1 -Stop        # Stop all services
.\start.ps1 -BackendOnly # Start just backend
.\start.ps1 -FrontendOnly # Start just frontend
```

### Manual Start - Step by Step

Run each command, verify the expected output, then proceed to next step.

---

**STEP 1:** Open PowerShell, navigate to project:
```powershell
cd G:\AI-Project\RAG-Hybrid
```
**Verify:** Prompt shows `PS G:\AI-Project\RAG-Hybrid>`

---

**STEP 2:** Activate virtual environment:
```powershell
.\.venv\Scripts\activate
```
**Verify:** Prompt now shows `(.venv) PS G:\AI-Project\RAG-Hybrid>`

---

**STEP 3:** Start backend:
```powershell
cd backend
python main.py
```
**Verify:** You see `INFO: Application startup complete.` (ignore deprecation warnings)

---

**STEP 4:** Open a NEW PowerShell window (leave backend running), then:
```powershell
cd G:\AI-Project\RAG-Hybrid
```
**Verify:** Prompt shows `PS G:\AI-Project\RAG-Hybrid>`

---

**STEP 5:** Activate venv in new window:
```powershell
.\.venv\Scripts\activate
```
**Verify:** Prompt shows `(.venv) PS G:\AI-Project\RAG-Hybrid>`

---

**STEP 6:** Start frontend:
```powershell
cd frontend-react
npm run dev
```
**Verify:** You see `VITE ready` and `http://localhost:5173`

---

**STEP 7:** Open browser to:
```
http://localhost:5173
```
**Verify:** You see the RAG chat interface

### Verification Commands

**Check if backend is running:**
```powershell
curl http://localhost:8000/api/v1/health
```
**Expected:** `{"status":"healthy",...}` or `{"status":"degraded",...}`

**Check if frontend is running:**
Open http://localhost:5173 in browser. You should see the chat interface.

### Stopping the App

Press `Ctrl+C` in each PowerShell window to stop that service.

### Full Restart (Kill Everything First)

If things aren't working, kill all Python processes and start fresh:
```powershell
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Start-Sleep 2
```
Then start backend and frontend as shown above.

### When to Restart

- **Backend**: After changing backend `.py` files (usually auto-reloads, but restart if issues)
- **Frontend**: Vite hot-reloads automatically for most changes
- **Both**: After system reboot, changing `.env` files, or if UI shows stale data
- **Full restart**: If you see old UI elements or "nothing changed" after code updates

Both services need `.env` configured (copy from `config/.env.example`).

### Auto-Start on Windows Login

RAG-Hybrid automatically starts when you log into Windows - no manual startup needed.

**How it works:**
- Windows Task Scheduler runs `start-hidden.vbs` at login
- This launches `start-background.ps1` completely hidden (no windows)
- Ollama, backend, and frontend all start in the background
- Logs are written to `logs/` directory for debugging

**To disable auto-start:**
1. Open Task Scheduler (search "Task Scheduler" in Start menu)
2. Find "RAG-Hybrid-Startup" in the list
3. Right-click > Disable

**To re-enable:**
1. Open Task Scheduler
2. Find "RAG-Hybrid-Startup"
3. Right-click > Enable

**To manually trigger (without reboot):**
Double-click `start-hidden.vbs` - services will start with no visible windows.

**Debug logs location:**
- `logs/startup.log` - Startup sequence log
- `logs/backend-stdout.log` / `logs/backend-stderr.log` - Backend output
- `logs/frontend-stdout.log` / `logs/frontend-stderr.log` - Frontend output

## Troubleshooting

### "Nothing changed" after code updates
1. Kill all Python processes: `Stop-Process -Name python -Force`
2. Clear Python cache: Delete `__pycache__` folders in `backend/`
3. Restart both services
4. Hard refresh browser: `Ctrl+Shift+R`

### Backend won't start
- Check `.env` file exists and has required keys
- Check port 8000 isn't in use: `netstat -an | findstr 8000`
- Look for error messages in PowerShell output

### Frontend shows login but can't connect
- Make sure backend is running first (check http://localhost:8000/api/v1/health)
- Check both services are using same `.env` file

### API returns 400 errors
- Click "Reload Config" button in UI after changing `.env`
- Verify API keys are valid (not placeholder values like `your_key_here`)

## Installing Python Packages

**This venv was created without pip. First bootstrap pip:**
```powershell
cd G:\AI-Project\RAG-Hybrid
.\.venv\Scripts\python.exe -m ensurepip --upgrade
```

**Then install packages:**
```powershell
.\.venv\Scripts\python.exe -m pip install PACKAGE_NAME
```

**Required packages not in requirements.txt:**
- `pypdf` - For PDF text extraction during indexing

## Form Filling Skills (AUTO-INVOKE)

When the user asks to fill, verify, audit, or fix bankruptcy forms, AUTOMATICALLY invoke the corresponding skill. Do NOT ask the user to type the slash command -- just invoke it.

| User says | Invoke |
|-----------|--------|
| "fill the forms" / "generate the package" / "fill form 101" / "regenerate forms" | `/fill-forms` |
| "smart fill" / "AI fill" / "fill the gaps" / "fill unmapped fields" | `/smart-fill` |
| "verify the forms" / "check the forms" / "run verification" | `/verify-forms` |
| "audit form 101" / "review form 101" / "check form 101 for errors" | `/audit-form 101` |
| "fix form 101" / "correct form 101" | `/fix-form 101` |
| "approve form 101" / "form 101 looks good" / "101 is approved" | `/approve-form 101` |

### Key files

| File | Purpose |
|------|---------|
| `backend/form_utils.py` | Shared logic: path resolution, value formatting, data loading |
| `backend/form_context.py` | Form context builder: extracts text, fields, profile slices for AI agents |
| `backend/rules_engine.py` | SaaS rules engine: evaluates rule-based mappings against any profile |
| `backend/form_mappings/form_*.json` | Static field mappings (auto-discovered by glob) |
| `backend/form_rules/form_*_rules.json` | Rule-based mappings (SaaS format, from /approve-form) |
| `scripts/generate_filing_package.py` | Mechanical fill: all forms from data_profile.json |
| `scripts/ai_fill_form.py` | AI fill: heuristics + AI reasoning for unmapped fields |
| `scripts/capture_rules.py` | Convert AI reasoning into reusable SaaS rules |
| `scripts/verify_filled_forms.py` | Mechanical verification (expected vs actual PDF values) |
| `backend/form_filler_engine.py` | API-based form fill engine (used by tools) |
| `data/project-kb/Chapter_7_Assistant/filing_package/` | Output: filled forms, verification results, AI reasoning |
| `docs/form_playbooks/form_*_playbook.md` | Decision playbooks: how/why each field was filled |

### Verification is mandatory

After ANY form fill or mapping change, ALWAYS run `verify_filled_forms.py`. Never declare forms "done" without verification passing.

## Model Storage

### Ollama Models (G:\AI-Project\Ollama\models\)
Active models managed by Ollama. Currently installed:
- `qwen2.5:14b` - Text generation model (9GB)

Note: Local vision models (moondream, llava, minicpm-v, qwen2.5vl) all hallucinate document text.
Vision/OCR is routed to Claude Sonnet 4.5 for accuracy. These local vision models can be deleted to save space.

### Standalone GGUF Files (G:\AI-Project\models\)
These are for LM Studio or other tools, NOT used by RAG-Hybrid:
- `granite-embedding-107m-multilingual-f16.gguf` (211MB) - Embeddings (can be deleted if not using LM Studio)

## VPS Deployment

### VPS Info
- **URL:** https://rag.coopeverything.org
- **IP:** 72.60.27.167
- **User:** root
- **Path:** /opt/rag-hybrid
- **Process Manager:** PM2 (backend runs as `rag-backend`)
- **Web Server:** nginx (serves React build + proxies API)

### SSH Access
Claude HAS SSH access to the VPS from this machine. Use it to:
- Update .env files on VPS
- Restart services (pm2 restart rag-backend)
- Check logs (pm2 logs rag-backend)
- Pull code manually if needed

**DO NOT ask user to SSH** - do it yourself:
```bash
ssh root@72.60.27.167 "command here"
```

Example commands:
```bash
ssh root@72.60.27.167 "pm2 restart rag-backend"
ssh root@72.60.27.167 "pm2 logs rag-backend --lines 50"
ssh root@72.60.27.167 "cd /opt/rag-hybrid && git pull"
ssh root@72.60.27.167 "curl -s http://localhost:8000/api/v1/health"
```

### Auto-Deployment (Recommended)
Pushing to `main` branch triggers GitHub Actions which:
1. SSHs into VPS using secret key
2. Runs `git pull`
3. Installs Python dependencies
4. Rebuilds React frontend
5. Restarts PM2 backend
6. Runs health check

**To deploy:** Just push to main. GitHub Actions handles everything.
```powershell
# From G:\AI-Project\RAG-Hybrid
git add -A
git commit -m "Your message"
git push origin main
```

**Check deployment status:**
```powershell
gh run list --repo faethon63/RAG-Hybrid --limit 3
```

### Manual VPS Commands (via SSH)
If you need to run commands on the VPS manually:

**Restart backend:**
```bash
pm2 restart rag-backend
```

**View backend logs:**
```bash
pm2 logs rag-backend --lines 50
```

**Check health:**
```bash
curl http://localhost:8000/api/v1/health
```

**Full redeploy:**
```bash
cd /opt/rag-hybrid
git fetch origin main
git reset --hard origin/main
source venv/bin/activate
pip install -r requirements.txt
cd frontend-react && npm ci && npm run build && cd ..
pm2 restart rag-backend
```

### VPS vs Local Differences
| Feature | Local (Windows) | VPS |
|---------|----------------|-----|
| Ollama | ✅ Running (GPU) | ❌ Disabled (no GPU) |
| Vision/OCR | Claude Sonnet 4.5 | Claude Sonnet 4.5 |
| Text generation | Ollama qwen2.5:14b | Claude (fallback) |
| URL | localhost:5173 | rag.coopeverything.org |

