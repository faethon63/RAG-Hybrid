"""
Session Review Pipeline — Second Brain
Async post-response pipeline that triages, extracts, and stores learnings.

Architecture:
  Stage 1: TRIAGE (Groq, free) — classifies interaction into flags
  Stage 2: EXTRACT & STORE (Groq, free) — saves to memory files / brain_items DB
  Stage 3: DEEP REVIEW (Claude Haiku, paid) — only for factual concerns (~10%)

Runs as fire-and-forget asyncio.create_task() after user gets their response.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Paths
_SESSION_LEARNINGS_DIR = Path(__file__).parent.parent / "data" / "session_learnings"

# Ensure directories exist
_SESSION_LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)


def _get_db_connection():
    """Get PostgreSQL connection. Returns None if unavailable."""
    try:
        import psycopg2
        from config import get_database_url
        db_url = get_database_url()
        if not db_url:
            return None
        return psycopg2.connect(db_url, connect_timeout=5)
    except Exception as e:
        logger.warning(f"Session review: DB connection failed: {e}")
        return None


def _append_jsonl(filename: str, data: dict):
    """Append a JSON line to a learnings file."""
    filepath = _SESSION_LEARNINGS_DIR / filename
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write to {filename}: {e}")


# ======================================================================
# Tool Handlers for Stages 2 & 3
# ======================================================================

async def _handle_save_brain_item(
    type: str,
    title: str,
    content: str = "",
    priority: int = 3,
    nudge_hours: int = None,
    **kwargs,
) -> Dict:
    """Save an item to the brain_items PostgreSQL table."""
    conn = _get_db_connection()
    if not conn:
        logger.warning("save_brain_item: No DB connection, skipping")
        return {"status": "skipped", "reason": "no database"}

    try:
        next_action_at = None
        if nudge_hours and nudge_hours > 0:
            next_action_at = datetime.utcnow() + timedelta(hours=nudge_hours)

        content_json = json.dumps({"notes": [content]} if content else {})

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO brain_items (type, title, content, priority, next_action_at, created_at, updated_at)
                   VALUES (%s, %s, %s::jsonb, %s, %s, NOW(), NOW())
                   RETURNING id""",
                (type, title, content_json, priority, next_action_at),
            )
            item_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"Brain item saved: [{type}] {title} (id={item_id})")
        return {"status": "saved", "id": str(item_id)}
    except Exception as e:
        logger.error(f"save_brain_item failed: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        conn.close()


async def _handle_log_interaction_learning(
    learning_type: str,
    description: str,
    **kwargs,
) -> Dict:
    """Log a learning to the appropriate JSONL file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": learning_type,
        "description": description,
        **{k: v for k, v in kwargs.items() if v is not None},
    }

    if learning_type == "correction":
        _append_jsonl("corrections.jsonl", entry)
    elif learning_type == "error_pattern":
        _append_jsonl("error_patterns.jsonl", entry)
    else:
        # Catch-all for any learning type (interest, insight, habit, preference, etc.)
        _append_jsonl("corrections.jsonl", entry)

    logger.info(f"Interaction learning logged: [{learning_type}] {description[:80]}")
    return {"status": "logged", "type": learning_type}


async def _handle_log_review_finding(
    severity: str,
    finding: str,
    suggestion: str = "",
    **kwargs,
) -> Dict:
    """Log a deep review finding (from Claude Haiku)."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": severity,
        "finding": finding,
        "suggestion": suggestion,
    }
    _append_jsonl("review_findings.jsonl", entry)
    logger.info(f"Review finding: [{severity}] {finding[:80]}")
    return {"status": "logged", "severity": severity}


# ======================================================================
# Stage 1: Triage (Groq — free)
# ======================================================================

TRIAGE_SYSTEM_PROMPT = """You are a triage classifier. Given a user query and the system's response, classify the interaction. Output ONLY valid JSON with no extra text:

{
  "memory_update": {"detected": false, "info": ""},
  "correction": {"detected": false, "original": "", "corrected": ""},
  "idea": {"detected": false, "description": ""},
  "interest": {"detected": false, "topic": ""},
  "factual_concern": {"detected": false, "concern": ""},
  "learning": {"detected": false, "pattern": ""}
}

Rules:
- memory_update: User shares personal info (plans, location, preferences, goals, health, relationships, decisions)
- correction: User corrects something ("actually it's X not Y", "that's wrong, it should be...")
- idea: User describes a feature, project, or creative idea they want to explore
- interest: User shows sustained interest in a topic (research requests, repeated questions)
- factual_concern: The response contains something that seems questionable, unverified, or potentially hallucinated
- learning: There's a pattern worth remembering (what worked, what didn't, user preference for how things are done)

Set detected=true only when clearly present. Most interactions will have all flags false."""


async def _stage_triage(query: str, answer: str) -> Optional[Dict]:
    """Stage 1: Classify the interaction using Groq (free)."""
    try:
        from groq_agent import get_groq_api_key
        from config import get_groq_model

        api_key = get_groq_api_key()
        if not api_key:
            return None

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"USER QUERY:\n{query}\n\nSYSTEM RESPONSE:\n{answer[:2000]}"},
        ]

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": get_groq_model(),
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON — handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            triage = json.loads(content)
            flags = [k for k, v in triage.items() if isinstance(v, dict) and v.get("detected")]
            if flags:
                logger.info(f"Session review triage flags: {flags}")
            return triage

    except json.JSONDecodeError:
        logger.warning("Session review: Triage JSON parse failed, skipping")
        return None
    except Exception as e:
        logger.warning(f"Session review: Triage failed: {e}")
        return None


# ======================================================================
# Stage 2: Extract & Store (Groq — free, with tools)
# ======================================================================

STAGE2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_user_memory",
            "description": "Update persistent user memory when personal info is detected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["user", "interests", "memory", "soul", "notes"],
                        "description": "Which memory file: 'user' (identity, location, goals), 'interests' (topics, projects), 'memory' (facts, decisions), 'soul' (communication preferences), 'notes' (ONLY for explicit user requests to save notes — never use this automatically)",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["add", "update", "remove"],
                        "description": "add=new info, update=correct existing, remove=delete outdated",
                    },
                    "content": {
                        "type": "string",
                        "description": "The information to save (as a bullet point)",
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional: which section header to place this under",
                    },
                },
                "required": ["category", "action", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_brain_item",
            "description": "Save a tracked item (idea, interest, discussion topic) to the brain database for follow-up nudges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["idea", "interest", "discussion", "fact", "decision", "error_pattern"],
                        "description": "Category of brain item",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short summary (1 line)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Detailed notes",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "1-5, where 5 is most important",
                    },
                    "nudge_hours": {
                        "type": "integer",
                        "description": "Hours from now to send a follow-up nudge. Use 24 for next day.",
                    },
                },
                "required": ["type", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_interaction_learning",
            "description": "Log a correction or pattern for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "learning_type": {
                        "type": "string",
                        "enum": ["correction", "error_pattern", "preference", "interest", "insight", "habit"],
                        "description": "Type of learning",
                    },
                    "description": {
                        "type": "string",
                        "description": "What was learned",
                    },
                },
                "required": ["learning_type", "description"],
            },
        },
    },
]

STAGE2_SYSTEM_PROMPT = """You are a Second Brain assistant. Based on the triage classification below, use the available tools to save relevant information.

Rules:
- If memory_update detected: call update_user_memory with the personal info. NEVER use category="notes" — notes are only for explicit user requests.
- If correction detected: call log_interaction_learning with type "correction"
- If idea detected: call save_brain_item with type "idea", priority >= 3, and nudge_hours=24. Only save genuinely novel ideas the user described, not routine questions.
- If interest detected: call save_brain_item with type "interest" and priority >= 3. Only save sustained/deep interests, NOT routine queries like "user asked about X".
- If learning detected: call log_interaction_learning with the pattern
- You may call multiple tools if multiple flags are set
- Only call tools for flags that are actually detected (detected=true)
- If no flags are detected, respond with "no_action_needed"
- QUALITY OVER QUANTITY: Only save items worth revisiting. "User asked about the weather" is NOT worth saving. "User is planning to open a massage practice in Barcelona" IS worth saving.

Be concise in your tool arguments. Use clear, short descriptions."""


async def _stage_extract_store(
    query: str,
    answer: str,
    triage: Dict,
    memory_handler=None,
) -> None:
    """Stage 2: Extract info and store using Groq + tools."""
    # Check if any flags are set
    any_flag = any(
        isinstance(v, dict) and v.get("detected")
        for v in triage.values()
    )
    if not any_flag:
        return

    logger.info(f"Session review Stage 2: starting extract & store")

    try:
        from groq_agent import get_groq_api_key
        from config import get_groq_model

        api_key = get_groq_api_key()
        if not api_key:
            return

        # Build tool handlers map
        tool_handlers = {
            "save_brain_item": _handle_save_brain_item,
            "log_interaction_learning": _handle_log_interaction_learning,
        }
        if memory_handler:
            tool_handlers["update_user_memory"] = memory_handler
        logger.info(f"Stage 2 tools available: {list(tool_handlers.keys())}")

        triage_summary = json.dumps(
            {k: v for k, v in triage.items() if isinstance(v, dict) and v.get("detected")},
            indent=2,
        )

        messages = [
            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"TRIAGE RESULTS:\n{triage_summary}\n\n"
                f"USER QUERY:\n{query}\n\n"
                f"SYSTEM RESPONSE:\n{answer[:1500]}"
            )},
        ]

        async with httpx.AsyncClient(timeout=20.0) as client:
            # Single Groq call — execute returned tools, then stop
            active_tools = STAGE2_TOOLS if memory_handler else [t for t in STAGE2_TOOLS if t["function"]["name"] != "update_user_memory"]
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": get_groq_model(),
                    "messages": messages,
                    "tools": active_tools,
                    "tool_choice": "auto",
                    "max_tokens": 800,
                    "temperature": 0.1,
                },
            )

            tool_calls_to_execute = []

            if resp.status_code != 200:
                # Try to recover tool calls from failed_generation (Groq quirk)
                error_text = resp.text
                logger.warning(f"Stage 2 Groq error {resp.status_code}: {error_text[:300]}")
                try:
                    error_data = json.loads(error_text)
                    failed_gen = error_data.get("error", {}).get("failed_generation", "")
                    if failed_gen:
                        recovered = json.loads(failed_gen)
                        if isinstance(recovered, list):
                            for tc in recovered:
                                tool_calls_to_execute.append((tc.get("name", ""), tc.get("parameters", {})))
                except Exception as recover_err:
                    logger.debug(f"Stage 2 recovery failed: {recover_err}")
            else:
                data = resp.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls_to_execute.append((name, args))

            # Execute all tool calls once
            for name, args in tool_calls_to_execute:
                handler = tool_handlers.get(name)
                if handler:
                    result = await handler(**args)
                    logger.info(f"Session review Stage 2: {name}({args}) -> {result}")

    except Exception as e:
        logger.warning(f"Session review Stage 2 failed: {e}")


# ======================================================================
# Stage 3: Deep Review (Claude Haiku — paid, only for factual concerns)
# ======================================================================

STAGE3_TOOLS = [
    {
        "type": "auto",
        "name": "log_review_finding",
        "description": "Log a quality review finding.",
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How serious the issue is",
                },
                "finding": {
                    "type": "string",
                    "description": "What was found (hallucination, contradiction, etc.)",
                },
                "suggestion": {
                    "type": "string",
                    "description": "How to improve",
                },
            },
            "required": ["severity", "finding"],
        },
    },
    {
        "type": "auto",
        "name": "save_brain_item",
        "description": "Save a tracked item for follow-up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["idea", "interest", "discussion", "fact", "decision", "error_pattern"],
                },
                "title": {"type": "string", "description": "Short summary"},
                "content": {"type": "string", "description": "Detailed notes"},
                "priority": {"type": "integer", "description": "1-5"},
                "nudge_hours": {"type": "integer", "description": "Hours until nudge"},
            },
            "required": ["type", "title"],
        },
    },
]

STAGE3_SYSTEM_PROMPT = """You are a quality reviewer. Analyze the system's response for:
1. Hallucinations — claims presented as fact without basis
2. Contradictions — response contradicts the user's stated info
3. Missing caveats — response is overly confident about uncertain things
4. Incorrect reasoning — logical errors

If you find issues, call log_review_finding for each one.
If the response quality is acceptable, just say "review_passed".
Be specific and concise."""


async def _stage_deep_review(query: str, answer: str, concern: str) -> None:
    """Stage 3: Claude Haiku reviews answer quality (paid, ~$0.002/call)."""
    try:
        from config import get_claude_haiku_model, get_anthropic_api_key

        api_key = get_anthropic_api_key()
        if not api_key:
            return

        model = get_claude_haiku_model()

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 800,
                    "temperature": 0.1,
                    "system": STAGE3_SYSTEM_PROMPT,
                    "tools": STAGE3_TOOLS,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"FLAGGED CONCERN: {concern}\n\n"
                                f"USER QUERY:\n{query}\n\n"
                                f"SYSTEM RESPONSE:\n{answer[:2000]}"
                            ),
                        }
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Process tool use blocks from Claude's response
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "")
                    args = block.get("input", {})

                    if name == "log_review_finding":
                        await _handle_log_review_finding(**args)
                    elif name == "save_brain_item":
                        await _handle_save_brain_item(**args)

    except Exception as e:
        logger.warning(f"Session review Stage 3 (deep review) failed: {e}")


# ======================================================================
# Main Entry Point
# ======================================================================

async def run_review_pipeline(
    query: str,
    answer: str,
    memory_handler=None,
    chat_id: str = None,
) -> None:
    """
    Main entry point — run the full review pipeline.
    Called as fire-and-forget: asyncio.create_task(run_review_pipeline(...))

    Args:
        query: The user's original query
        answer: The system's response text
        memory_handler: The update_user_memory handler from main.py (async callable)
        chat_id: Optional chat ID for linking brain items to conversations
    """
    try:
        # Stage 1: Triage
        triage = await _stage_triage(query, answer)
        if not triage:
            return  # Parse failed or no API key — skip silently

        # Check if any flags were set
        any_flag = any(
            isinstance(v, dict) and v.get("detected")
            for v in triage.values()
        )
        if not any_flag:
            logger.debug("Session review: No flags detected, skipping stages 2-3")
            return

        # Stage 2: Extract & Store (always runs if any flag set)
        await _stage_extract_store(query, answer, triage, memory_handler)

        # Stage 3: Deep Review (only if factual_concern flagged)
        factual = triage.get("factual_concern", {})
        if isinstance(factual, dict) and factual.get("detected"):
            concern = factual.get("concern", "Unspecified concern")
            logger.info(f"Session review: Triggering deep review for concern: {concern}")
            await _stage_deep_review(query, answer, concern)

        logger.info("Session review pipeline completed")

    except Exception as e:
        logger.error(f"Session review pipeline error: {e}")
        # Never crash — this is fire-and-forget
