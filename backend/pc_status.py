"""
PC Status Reporter
Reports local PC status (online, Claude Code running) to shared PostgreSQL
so the VPS frontend can display Remote Control availability.
"""

import logging
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from config import get_database_url

logger = logging.getLogger(__name__)


def _get_db_connection():
    """Get PostgreSQL connection if DATABASE_URL is configured."""
    import psycopg2
    db_url = get_database_url()
    if not db_url:
        return None
    try:
        return psycopg2.connect(db_url)
    except Exception as e:
        logger.warning(f"PC status DB connection failed: {e}")
        return None


def ensure_status_table():
    """Create system_status table if it doesn't exist."""
    conn = _get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_status (
                    key VARCHAR(100) PRIMARY KEY,
                    value JSONB NOT NULL DEFAULT '{}',
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
        conn.commit()
        logger.info("system_status table verified/created")
    except Exception as e:
        logger.warning(f"Failed to ensure system_status table: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def _is_claude_code_running() -> bool:
    """Check if Claude Code process is running on this machine."""
    system = platform.system()
    try:
        if system == "Windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq claude.exe", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return "claude.exe" in result.stdout.lower()
        else:
            # Linux/Mac
            result = subprocess.run(
                ["pgrep", "-x", "claude"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
    except Exception as e:
        logger.debug(f"Claude Code process check failed: {e}")
        return False


def report_pc_status():
    """Write current PC status to PostgreSQL. Called periodically by heartbeat."""
    import json
    conn = _get_db_connection()
    if not conn:
        return

    claude_running = _is_claude_code_running()
    hostname = platform.node()

    status = {
        "online": True,
        "claude_code_running": claude_running,
        "hostname": hostname,
        "platform": platform.system(),
    }

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO system_status (key, value, updated_at)
                VALUES ('pc_status', %s, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
            """, (json.dumps(status),))
        conn.commit()
        logger.debug(f"PC status reported: claude_running={claude_running}")
    except Exception as e:
        logger.warning(f"Failed to report PC status: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def get_pc_status() -> Dict:
    """Read PC status from PostgreSQL. Used by VPS endpoint."""
    import json
    conn = _get_db_connection()
    if not conn:
        return {
            "pc_online": False,
            "claude_code_running": False,
            "reason": "Database not configured",
        }

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT value, updated_at FROM system_status
                WHERE key = 'pc_status'
            """)
            row = cur.fetchone()

        if not row:
            return {
                "pc_online": False,
                "claude_code_running": False,
                "reason": "No status reported yet",
            }

        value, updated_at = row
        if isinstance(value, str):
            value = json.loads(value)

        # Consider PC offline if last report > 10 minutes ago
        now = datetime.now(timezone.utc)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = now - updated_at
        is_stale = age > timedelta(minutes=10)

        return {
            "pc_online": not is_stale,
            "claude_code_running": value.get("claude_code_running", False) and not is_stale,
            "hostname": value.get("hostname", ""),
            "last_seen": updated_at.isoformat(),
            "stale": is_stale,
            "age_seconds": int(age.total_seconds()),
        }

    except Exception as e:
        logger.warning(f"Failed to read PC status: {e}")
        return {
            "pc_online": False,
            "claude_code_running": False,
            "reason": f"Database error: {e}",
        }
    finally:
        conn.close()
