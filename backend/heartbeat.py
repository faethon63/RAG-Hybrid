"""
Heartbeat Service
Proactive background process that checks on the user and sends notifications.
Uses APScheduler running inside the FastAPI process.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_LOG_FILE = _DATA_DIR / "heartbeat_log.json"
_CONFIG_FILE = _DATA_DIR / "heartbeat_config.json"
_USER_MEMORY_DIR = Path(__file__).parent.parent / "config" / "project-kb" / "_user_memory"


class HeartbeatService:
    def __init__(self):
        self.scheduler = None
        self._config = self._load_config()
        self._last_check: Optional[str] = None
        self._recent_notifications: List[str] = []  # dedup cache

    def _load_config(self) -> Dict:
        defaults = {
            "enabled": True,
            "check_interval_minutes": 30,
            "briefing_hour": 9,  # 9 AM
            "briefing_minute": 0,
            "quiet_hours_start": 23,  # 11 PM
            "quiet_hours_end": 7,     # 7 AM
        }
        try:
            if _CONFIG_FILE.exists():
                with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    defaults.update(saved)
        except Exception as e:
            logger.warning(f"Failed to load heartbeat config: {e}")
        return defaults

    def _save_config(self):
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)

    def _is_quiet_hours(self) -> bool:
        now = datetime.now().hour
        start = self._config.get("quiet_hours_start", 23)
        end = self._config.get("quiet_hours_end", 7)
        if start > end:
            return now >= start or now < end
        return start <= now < end

    def _load_user_memory(self) -> str:
        parts = []
        for fname in ("user.md", "interests.md", "memory.md", "soul.md"):
            fpath = _USER_MEMORY_DIR / fname
            try:
                parts.append(fpath.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
        return "\n\n".join(parts) if parts else ""

    def _load_heartbeat_log(self) -> List[Dict]:
        try:
            if _LOG_FILE.exists():
                with open(_LOG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_heartbeat_log(self, log: List[Dict]):
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Keep only last 7 days
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        log = [entry for entry in log if entry.get("timestamp", "") > cutoff]
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    async def _get_recent_chats_summary(self) -> str:
        """Get summary of recent chat activity (last 24h)."""
        try:
            import rag_core
            chats = await rag_core.list_chats(limit=10)
            if not chats:
                return "No recent chats."

            summaries = []
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            for chat_summary in chats[:5]:
                updated = chat_summary.get("updated_at", "")
                if updated and updated < cutoff:
                    continue
                title = chat_summary.get("title", chat_summary.get("name", "Untitled"))
                project = chat_summary.get("project", "General")
                summaries.append(f"- [{project}] {title}")

            if not summaries:
                return "No chats in the last 24 hours."
            return "Recent conversations:\n" + "\n".join(summaries)
        except Exception as e:
            logger.warning(f"Failed to get recent chats: {e}")
            return "Could not retrieve recent chats."

    async def check_and_notify(self):
        """Periodic check: should we notify the user about anything?"""
        if not self._config.get("enabled", True):
            return
        if self._is_quiet_hours():
            logger.debug("Heartbeat: quiet hours, skipping")
            return

        logger.info("Heartbeat: running periodic check")
        self._last_check = datetime.utcnow().isoformat()

        user_memory = self._load_user_memory()
        recent_chats = await self._get_recent_chats_summary()

        prompt = f"""You are a proactive AI assistant analyzing whether to send a notification to the user.

USER PROFILE:
{user_memory}

RECENT ACTIVITY:
{recent_chats}

CURRENT TIME: {datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")}

Should you send a notification? Consider:
- Upcoming deadlines or tasks the user mentioned
- Follow-ups on ongoing projects
- Reminders for things they said they'd do
- Encouragement if they've been working hard
- Do NOT notify just to say "no updates" — only if there's something genuinely useful

Respond with ONLY valid JSON:
{{"notify": true/false, "title": "short title", "message": "notification body (1-2 sentences)", "reason": "why you decided this"}}"""

        try:
            from groq_agent import get_groq_api_key
            api_key = get_groq_api_key()
            if not api_key:
                logger.warning("Heartbeat: no Groq API key")
                return

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "You are a proactive AI assistant. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 300,
                    },
                )
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"].strip()

                # Parse JSON
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                decision = json.loads(content)

                # Log the decision
                log = self._load_heartbeat_log()
                log.append({
                    "timestamp": self._last_check,
                    "type": "check",
                    "notify": decision.get("notify", False),
                    "reason": decision.get("reason", ""),
                    "message": decision.get("message", ""),
                })
                self._save_heartbeat_log(log)

                if decision.get("notify"):
                    title = decision.get("title", "RAG-Hybrid")
                    message = decision.get("message", "")

                    # Dedup: don't send same message within 24h
                    msg_key = message[:50].lower()
                    if msg_key in self._recent_notifications:
                        logger.info(f"Heartbeat: skipping duplicate notification: {msg_key}")
                        return

                    self._recent_notifications.append(msg_key)
                    # Keep dedup cache manageable
                    if len(self._recent_notifications) > 100:
                        self._recent_notifications = self._recent_notifications[-50:]

                    # Send push notification
                    from notifications import send_push_notification
                    await send_push_notification(title=title, body=message, tag="heartbeat")
                    logger.info(f"Heartbeat notification sent: {title}")
                else:
                    logger.info(f"Heartbeat: no notification needed. Reason: {decision.get('reason', 'N/A')}")

        except Exception as e:
            logger.error(f"Heartbeat check failed: {e}")

    async def daily_briefing(self):
        """Generate and send daily briefing based on user interests."""
        if not self._config.get("enabled", True):
            return

        logger.info("Heartbeat: generating daily briefing")
        user_memory = self._load_user_memory()

        # Extract interests for search queries
        interests_text = ""
        interests_path = _USER_MEMORY_DIR / "interests.md"
        try:
            interests_text = interests_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            interests_text = "AI, technology, personal productivity"

        try:
            from groq_agent import get_groq_api_key
            api_key = get_groq_api_key()
            if not api_key:
                return

            # Step 1: Generate search queries from interests
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "Generate 2-3 web search queries for a daily news briefing. Return ONLY a JSON array of search query strings."},
                            {"role": "user", "content": f"User interests:\n{interests_text}\n\nDate: {datetime.now().strftime('%B %d, %Y')}"},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                queries = json.loads(content)

            # Step 2: Search using Perplexity (if available)
            search_results = []
            perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")
            if perplexity_key:
                for query in queries[:3]:
                    try:
                        async with httpx.AsyncClient(timeout=20.0) as client:
                            resp = await client.post(
                                "https://api.perplexity.ai/chat/completions",
                                headers={"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"},
                                json={
                                    "model": "sonar",
                                    "messages": [{"role": "user", "content": f"Latest news: {query}"}],
                                    "max_tokens": 300,
                                },
                            )
                            resp.raise_for_status()
                            answer = resp.json()["choices"][0]["message"]["content"]
                            search_results.append(f"**{query}**\n{answer}")
                    except Exception as e:
                        logger.warning(f"Briefing search failed for '{query}': {e}")

            # Step 3: Summarize into briefing
            search_text = "\n\n".join(search_results) if search_results else "No search results available."

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "Create a concise daily briefing (3-5 bullet points). Keep it personal and relevant."},
                            {"role": "user", "content": f"User profile:\n{user_memory}\n\nSearch results:\n{search_text}\n\nCreate a daily briefing for {datetime.now().strftime('%B %d, %Y')}."},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                briefing = resp.json()["choices"][0]["message"]["content"].strip()

            # Step 4: Send notification
            from notifications import send_push_notification
            await send_push_notification(
                title=f"Daily Briefing - {datetime.now().strftime('%b %d')}",
                body=briefing[:200] + ("..." if len(briefing) > 200 else ""),
                tag="daily-briefing",
            )

            # Step 5: Save as chat message (visible in UI)
            try:
                import rag_core
                chat_data = {
                    "name": f"Daily Briefing - {datetime.now().strftime('%B %d, %Y')}",
                    "project": None,
                    "messages": [
                        {"role": "assistant", "content": f"# Daily Briefing - {datetime.now().strftime('%B %d, %Y')}\n\n{briefing}"},
                    ],
                }
                await rag_core.save_chat(chat_data)
                logger.info("Daily briefing saved as chat")
            except Exception as e:
                logger.warning(f"Failed to save briefing as chat: {e}")

            # Log it
            log = self._load_heartbeat_log()
            log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": "daily_briefing",
                "queries": queries,
                "briefing_length": len(briefing),
            })
            self._save_heartbeat_log(log)

            logger.info("Daily briefing sent successfully")

        except Exception as e:
            logger.error(f"Daily briefing failed: {e}")

    def start(self):
        """Start the heartbeat scheduler."""
        if not self._config.get("enabled", True):
            logger.info("Heartbeat is disabled, not starting scheduler")
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
            from apscheduler.triggers.cron import CronTrigger

            self.scheduler = AsyncIOScheduler()

            # Periodic check every N minutes
            interval = self._config.get("check_interval_minutes", 30)
            self.scheduler.add_job(
                self.check_and_notify,
                IntervalTrigger(minutes=interval),
                id="heartbeat_check",
                replace_existing=True,
            )

            # Daily briefing at configured hour
            hour = self._config.get("briefing_hour", 9)
            minute = self._config.get("briefing_minute", 0)
            self.scheduler.add_job(
                self.daily_briefing,
                CronTrigger(hour=hour, minute=minute),
                id="daily_briefing",
                replace_existing=True,
            )

            # PC status reporting every 5 minutes (for Remote Control UI)
            self.scheduler.add_job(
                self._report_pc_status,
                IntervalTrigger(minutes=5),
                id="pc_status_report",
                replace_existing=True,
            )
            # Report immediately on startup
            self.scheduler.add_job(
                self._report_pc_status,
                id="pc_status_initial",
                replace_existing=True,
            )

            self.scheduler.start()
            logger.info(f"Heartbeat started: check every {interval}min, briefing at {hour:02d}:{minute:02d}, PC status every 5min")
        except Exception as e:
            logger.error(f"Failed to start heartbeat scheduler: {e}")

    def stop(self):
        """Stop the heartbeat scheduler."""
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None
            logger.info("Heartbeat stopped")

    def get_status(self) -> Dict:
        """Get current heartbeat status."""
        next_check = None
        next_briefing = None
        if self.scheduler:
            check_job = self.scheduler.get_job("heartbeat_check")
            briefing_job = self.scheduler.get_job("daily_briefing")
            if check_job and check_job.next_run_time:
                next_check = check_job.next_run_time.isoformat()
            if briefing_job and briefing_job.next_run_time:
                next_briefing = briefing_job.next_run_time.isoformat()

        recent_log = self._load_heartbeat_log()[-5:]

        return {
            "enabled": self._config.get("enabled", True),
            "running": self.scheduler is not None and self.scheduler.running if self.scheduler else False,
            "last_check": self._last_check,
            "next_check": next_check,
            "next_briefing": next_briefing,
            "interval_minutes": self._config.get("check_interval_minutes", 30),
            "quiet_hours": f"{self._config.get('quiet_hours_start', 23)}:00 - {self._config.get('quiet_hours_end', 7)}:00",
            "recent_log": recent_log,
        }

    async def _report_pc_status(self):
        """Report PC status to shared PostgreSQL (runs every 5 min)."""
        try:
            from pc_status import report_pc_status
            await report_pc_status()
        except Exception as e:
            logger.debug(f"PC status report failed: {e}")

    def update_config(self, updates: Dict):
        """Update heartbeat configuration."""
        self._config.update(updates)
        self._save_config()

        # Restart scheduler if running
        if self.scheduler:
            self.stop()
            if self._config.get("enabled", True):
                self.start()


# Singleton instance
heartbeat_service = HeartbeatService()
