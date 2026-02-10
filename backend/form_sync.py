"""
Form Sync Module
Syncs filled forms and data profiles between local and VPS via PostgreSQL.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List

from config import get_database_url, get_project_kb_path

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
        logger.warning(f"Form sync DB connection failed: {e}")
        return None


def ensure_sync_tables():
    """Create sync tables if they don't exist."""
    conn = _get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS filled_forms (
                    id VARCHAR(255) PRIMARY KEY,
                    project_name VARCHAR(255) NOT NULL,
                    form_id VARCHAR(50) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    pdf_data BYTEA NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_filled_forms_project
                    ON filled_forms(project_name);
                CREATE TABLE IF NOT EXISTS data_profiles (
                    project_name VARCHAR(255) PRIMARY KEY,
                    profile_data JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()
        logger.info("Sync tables verified/created")
    except Exception as e:
        logger.warning(f"Failed to ensure sync tables: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Filled Forms
# ---------------------------------------------------------------------------

def save_filled_form_to_db(
    project_name: str, form_id: str, filename: str, pdf_path: str,
    metadata: dict = None,
) -> bool:
    """Save filled PDF binary to Postgres."""
    import psycopg2
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        form_db_id = f"{project_name}_{filename}"
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO filled_forms (id, project_name, form_id, filename, pdf_data, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    pdf_data = EXCLUDED.pdf_data,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (
                form_db_id, project_name, form_id, filename,
                psycopg2.Binary(pdf_bytes),
                json.dumps(metadata or {}),
            ))
        conn.commit()
        logger.info(f"Saved filled form to DB: {filename} ({len(pdf_bytes)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to save filled form to DB: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()


def copy_to_windows_folder(pdf_path: str, allowed_paths: List[str]) -> Optional[str]:
    """Copy a filled PDF to {allowed_paths[0]}/Filled Forms/ subfolder."""
    if not allowed_paths:
        return None
    from file_tools import convert_windows_to_wsl
    for ap in allowed_paths:
        wsl_path = convert_windows_to_wsl(ap)
        target_dir = Path(wsl_path) / "Filled Forms"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(pdf_path).name
            shutil.copy2(pdf_path, target_path)
            logger.info(f"Copied filled form to: {target_path}")
            # Return the Windows-style path for display
            win_target = f"{ap}\\Filled Forms\\{Path(pdf_path).name}"
            return win_target
        except Exception as e:
            logger.warning(f"Failed to copy to {target_dir}: {e}")
            continue
    return None


def sync_filled_forms_from_db(project_name: str, allowed_paths: List[str] = None) -> int:
    """Pull missing filled forms from Postgres to local filesystem + Windows folder."""
    conn = _get_db_connection()
    if not conn:
        return 0

    filled_dir = Path(get_project_kb_path()) / project_name / "filled_forms"
    filled_dir.mkdir(parents=True, exist_ok=True)

    synced = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename, pdf_data FROM filled_forms WHERE project_name = %s",
                (project_name,),
            )
            for filename, pdf_data in cur.fetchall():
                local_path = filled_dir / filename
                if not local_path.exists():
                    with open(local_path, "wb") as f:
                        f.write(bytes(pdf_data))
                    logger.info(f"Synced from DB: {filename}")
                    synced += 1
                    if allowed_paths:
                        copy_to_windows_folder(str(local_path), allowed_paths)
    except Exception as e:
        logger.error(f"Failed to sync filled forms: {e}")
    finally:
        conn.close()
    return synced


def list_filled_forms_in_db(project_name: str) -> list:
    """List filled forms in Postgres for a project."""
    conn = _get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename, form_id, created_at, length(pdf_data) FROM filled_forms WHERE project_name = %s ORDER BY created_at DESC",
                (project_name,),
            )
            return [
                {"filename": r[0], "form_id": r[1], "created_at": r[2].isoformat() if r[2] else None, "size": r[3]}
                for r in cur.fetchall()
            ]
    except Exception as e:
        logger.error(f"Failed to list filled forms: {e}")
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data Profile
# ---------------------------------------------------------------------------

def save_data_profile_to_db(project_name: str, profile_data: dict) -> bool:
    """Save data_profile.json to Postgres."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data_profiles (project_name, profile_data, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (project_name) DO UPDATE SET
                    profile_data = EXCLUDED.profile_data,
                    updated_at = NOW()
            """, (project_name, json.dumps(profile_data)))
        conn.commit()
        logger.info(f"Saved data profile to DB: {project_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save data profile to DB: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()


def load_data_profile_from_db(project_name: str) -> Optional[dict]:
    """Load data profile from Postgres. Returns None if not found."""
    conn = _get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT profile_data FROM data_profiles WHERE project_name = %s",
                (project_name,),
            )
            row = cur.fetchone()
            if row:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    except Exception as e:
        logger.error(f"Failed to load data profile from DB: {e}")
    finally:
        conn.close()
    return None


def sync_data_profile_from_db(project_name: str) -> bool:
    """Pull data profile from Postgres if local file is missing or older."""
    db_profile = load_data_profile_from_db(project_name)
    if not db_profile:
        return False

    local_path = Path(get_project_kb_path()) / project_name / "data_profile.json"

    if local_path.exists():
        try:
            with open(local_path) as f:
                local_data = json.load(f)
            local_updated = local_data.get("metadata", {}).get("updated_at", "")
            db_updated = db_profile.get("metadata", {}).get("updated_at", "")
            if local_updated and db_updated and local_updated >= db_updated:
                return False  # Local is newer or same
        except Exception:
            pass  # Can't read local — overwrite it

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(db_profile, f, indent=2)
    logger.info(f"Synced data profile from DB: {project_name}")
    return True
