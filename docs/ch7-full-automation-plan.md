# Chapter 7 Assistant — Full Automation Plan

## Vision

The Ch7 Assistant should be fully autonomous through the chat UI. A user should be able to:
1. Upload documents (bank statements, tax returns) via the UI
2. Have data automatically extracted into the truth file (data_profile.json)
3. Fill out any bankruptcy form through chat ("fill form 101" → approve → done)
4. Get filled forms automatically saved to their Windows folder AND synced to VPS
5. Never need Claude Code for any of the above

## Current State (Feb 2026)

### Already Working
- Form filling via chat: plan → approve → fill → download link
- Groq chat interceptor auto-detects "fill form X" and "approve" patterns
- Blank forms auto-downloaded from uscourts.gov if missing
- Data profile auto-injected into Groq for financial queries
- File upload + auto-indexing via UI

### Implemented (this session)
- Filled forms sync between local and VPS via PostgreSQL
- Filled forms auto-saved to Windows `Filled Forms` subfolder
- data_profile.json synced via PostgreSQL (no more manual SCP)

## Remaining Gaps (Future Work)

### 1. Auto-Extract Data Profile After Upload

**Problem:** When user uploads bank statements or tax returns via the UI, files are indexed into ChromaDB but financial data is NOT extracted into data_profile.json. User must manually chat "build data profile" to trigger extraction.

**Solution:** In `main.py:upload_project_files()` (line 2652), after auto-index succeeds:
- Detect if uploaded files match bank/tax patterns (filename contains "tax", "federal", "bank", "statement", or starts with "Bus "/"Per ")
- If matches found AND Claude API key is configured, auto-call `build_data_profile()` from `data_extractor.py:491`
- Use `use_dual_verification=False` for speed
- Skip silently if no API key

**Dependency:** Requires Claude API credits for OCR/extraction.

### 2. Form Inventory UI

**Problem:** No UI to show available blank forms, filled forms, or form status. Users must know form IDs and type them in chat.

**Solution:**
- Add sidebar panel or modal listing all Ch7 forms with status (blank available / filled / not downloaded)
- Use existing `GET /api/v1/projects/{name}/forms/list` endpoint
- Add "Fill This Form" button per row that sends the chat command
- Show filled form download links inline

### 3. Auto-Seed Blank Forms

**Problem:** Blank forms directory starts empty. User must request each form download individually.

**Solution:** On first form fill attempt for a project, or via a "seed forms" chat command:
- Download all standard Ch7 forms from uscourts.gov in one batch
- Forms: 101, 106A/B, 106C, 106D, 106E/F, 106G, 106H, 106I, 106J, 106Sum, 107, 108, 119, 121, 122A-1, 122A-2

### 4. Guided Workflow

**Problem:** User must know what to do next. No proactive guidance.

**Solution:** Update Ch7 system prompt to include a guided workflow:
- When user first opens project: "Welcome! To get started, upload your bank statements and tax returns."
- After upload: "I've extracted your financial data. Ready to fill forms? Start with Form 101 (Voluntary Petition)."
- After each form: "Form 101 complete! Next recommended: Form 106A/B (Property)."
- Track progress: which forms are done, which are pending

### 5. Auto-Consistency Check

**Problem:** `check_data_consistency` tool exists but only runs on explicit request.

**Solution:** Auto-run after data profile is built. Show warnings in chat:
- "Warning: Bank deposits ($X) don't match reported income ($Y)"
- "Note: 2 months of business statements are missing"

### 6. Batch Form Filling

**Problem:** Can only fill one form at a time through chat.

**Solution:** Add "fill all forms" or "fill forms 101, 106A, 122A-1" batch command that:
- Generates plans for all requested forms
- Shows combined plan for review
- Fills all on approval
- Reports results for each

## Priority Order

1. Auto-extract after upload (biggest friction point)
2. Guided workflow (system prompt update, low effort)
3. Form inventory UI (nice to have)
4. Auto-seed blank forms (convenience)
5. Auto-consistency check (quality improvement)
6. Batch form filling (power user feature)
