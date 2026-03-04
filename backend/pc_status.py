"""
PC Status Reporter
Local PC reports status via HTTPS to VPS API.
VPS stores in PostgreSQL and serves to frontend.
"""

import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict

import httpx

from config import get_database_url

logger = logging.getLogger(__name__)

_VPS_API_URL = os.getenv("VPS_API_URL", "https://rag.coopeverything.org")


def _get_db_connection():
    """Get PostgreSQL connection (only works on VPS where DB is localhost)."""
    import psycopg2
    db_url = get_database_url()
    if not db_url:
        return None
    try:
        return psycopg2.connect(db_url, connect_timeout=5)
    except Exception as e:
        logger.debug(f"PC status DB connection failed: {e}")
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
            result = subprocess.run(
                ["pgrep", "-x", "claude"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
    except Exception as e:
        logger.debug(f"Claude Code process check failed: {e}")
        return False


def _is_local_environment() -> bool:
    """Check if running on local PC (not VPS)."""
    return platform.system() == "Windows"


async def report_pc_status():
    """Report PC status. Local PC → POST to VPS API. VPS → write to DB directly."""
    claude_running = _is_claude_code_running()
    hostname = platform.node()

    status = {
        "online": True,
        "claude_code_running": claude_running,
        "hostname": hostname,
        "platform": platform.system(),
    }

    if _is_local_environment():
        # Local PC: POST status to VPS API
        api_key = os.getenv("VPS_API_KEY", os.getenv("API_KEY", ""))
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{_VPS_API_URL}/api/v1/remote-control/report",
                    json=status,
                    headers={"X-API-Key": api_key} if api_key else {},
                )
                if resp.status_code == 200:
                    logger.debug(f"PC status reported to VPS: claude_running={claude_running}")
                else:
                    logger.warning(f"VPS status report failed: {resp.status_code}")
        except Exception as e:
            logger.debug(f"Failed to report status to VPS: {e}")
    else:
        # VPS: write directly to local PostgreSQL
        _write_status_to_db(status)


def _write_status_to_db(status: Dict):
    """Write PC status to PostgreSQL (VPS only, localhost DB)."""
    conn = _get_db_connection()
    if not conn:
        return
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
        logger.debug(f"PC status written to DB")
    except Exception as e:
        logger.warning(f"Failed to write PC status: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def save_reported_status(status: Dict):
    """Called by VPS API endpoint when local PC reports in."""
    _write_status_to_db(status)


def get_pc_status() -> Dict:
    """Read PC status from PostgreSQL. Used by VPS endpoint."""
    conn = _get_db_connection()
    if not conn:
        # If no DB (local dev), check locally
        if _is_local_environment():
            return {
                "pc_online": True,
                "claude_code_running": _is_claude_code_running(),
                "hostname": platform.node(),
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "local_mode": True,
            }
        return {
            "pc_online": False,
            "claude_code_running": False,
            "reason": "Database not available",
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
