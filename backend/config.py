"""
Central Configuration Module
All environment variables are loaded lazily and can be reloaded at runtime.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Track if we've loaded .env files
_env_loaded = False


def _ensure_env_loaded():
    """Load .env files if not already loaded."""
    global _env_loaded
    if not _env_loaded:
        base_dir = Path(__file__).parent.parent
        load_dotenv(base_dir / "config" / ".env")
        load_dotenv(base_dir / ".env")
        _env_loaded = True


def reload_env():
    """Reload environment variables from .env files."""
    global _env_loaded
    base_dir = Path(__file__).parent.parent
    load_dotenv(base_dir / "config" / ".env", override=True)
    load_dotenv(base_dir / ".env", override=True)
    _env_loaded = True


# --- API Keys (lazy getters) ---

def get_anthropic_api_key() -> str:
    _ensure_env_loaded()
    return os.getenv("ANTHROPIC_API_KEY", "")


def get_perplexity_api_key() -> str:
    _ensure_env_loaded()
    return os.getenv("PERPLEXITY_API_KEY", "")


def get_tavily_api_key() -> str:
    _ensure_env_loaded()
    return os.getenv("TAVILY_API_KEY", "")


# --- Ollama Settings ---

def get_ollama_host() -> str:
    _ensure_env_loaded()
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


def get_ollama_model() -> str:
    _ensure_env_loaded()
    return os.getenv("OLLAMA_MODEL", "qwen2.5:14b")


# --- ChromaDB Settings ---

def get_chromadb_path() -> str:
    _ensure_env_loaded()
    default = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
    return os.getenv("CHROMADB_PATH", default)


def get_chromadb_collection() -> str:
    _ensure_env_loaded()
    return os.getenv("CHROMADB_COLLECTION", "rag_docs")


# --- Embedding Settings ---

def get_embedding_model() -> str:
    _ensure_env_loaded()
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# --- RAG Settings ---

def get_top_k() -> int:
    _ensure_env_loaded()
    return int(os.getenv("TOP_K_RESULTS", "5"))


def get_similarity_threshold() -> float:
    _ensure_env_loaded()
    return float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))


def get_max_tokens() -> int:
    _ensure_env_loaded()
    return int(os.getenv("MAX_TOKENS", "2000"))


def get_temperature() -> float:
    _ensure_env_loaded()
    return float(os.getenv("TEMPERATURE", "0.7"))


# --- Project Paths ---

def get_project_kb_path() -> str:
    """Local project data (documents, indexed_files) - gitignored."""
    _ensure_env_loaded()
    default = os.path.join(os.path.dirname(__file__), "..", "data", "project-kb")
    return os.getenv("PROJECT_KB_PATH", default)


def get_synced_projects_path() -> str:
    """Synced project configs (name, description, prompts) - tracked in git."""
    _ensure_env_loaded()
    default = os.path.join(os.path.dirname(__file__), "..", "config", "projects")
    return os.getenv("SYNCED_PROJECTS_PATH", default)


def get_rag_config_path() -> str:
    _ensure_env_loaded()
    default = os.path.join(os.path.dirname(__file__), "..", "data", "rag_config.json")
    return os.getenv("RAG_CONFIG_PATH", default)


# --- Auth Settings ---

def get_jwt_secret() -> str:
    _ensure_env_loaded()
    return os.getenv("JWT_SECRET", "change_me_to_random_hex_string")


def get_rate_limit_rpm() -> int:
    _ensure_env_loaded()
    return int(os.getenv("RATE_LIMIT_RPM", "30"))


def get_rate_limit_daily() -> int:
    _ensure_env_loaded()
    return int(os.getenv("RATE_LIMIT_DAILY", "500"))


def get_allowed_users() -> dict:
    """Parse ALLOWED_USERS env var into {username: hashed_password} dict."""
    _ensure_env_loaded()
    raw = os.getenv("ALLOWED_USERS", "")
    users = {}
    if not raw:
        return users
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        username, hashed = entry.split(":", 1)
        users[username.strip()] = hashed.strip()
    return users


# --- Server Settings ---

def get_log_level() -> str:
    _ensure_env_loaded()
    return os.getenv("LOG_LEVEL", "INFO")


def get_fastapi_port() -> int:
    _ensure_env_loaded()
    return int(os.getenv("FASTAPI_PORT", "8000"))


# --- Chat Storage ---

def get_chats_path() -> str:
    """Get the path for storing chat history JSON files (fallback if no DB)."""
    _ensure_env_loaded()
    default = os.path.join(os.path.dirname(__file__), "..", "data", "chats")
    return os.getenv("CHATS_PATH", default)


def get_database_url() -> str:
    """Get PostgreSQL connection URL for chat sync across local/VPS."""
    _ensure_env_loaded()
    return os.getenv("DATABASE_URL", "")
