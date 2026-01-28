"""
Authentication & Rate Limiting Module
JWT token verification, bcrypt password checking, per-user rate limiting.
"""

import os
import time
import jwt
import bcrypt
from typing import Optional, Dict
from collections import defaultdict
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- Configuration ---

JWT_SECRET = os.getenv("JWT_SECRET", "change_me_to_random_hex_string")
JWT_ALGORITHM = "HS256"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))
RATE_LIMIT_DAILY = int(os.getenv("RATE_LIMIT_DAILY", "500"))


def _load_allowed_users() -> Dict[str, str]:
    """Parse ALLOWED_USERS env var into {username: hashed_password} dict."""
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


ALLOWED_USERS = _load_allowed_users()


# --- JWT Token Functions ---

def create_token(username: str) -> str:
    """Create a JWT token for a user."""
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400 * 30,  # 30 days
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """
    Verify a JWT token and return the username, or None if invalid.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username:
            return username
    except jwt.ExpiredSignatureError:
        pass
    except jwt.InvalidTokenError:
        pass
    return None


# --- Password Functions ---

def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Authenticate with username/password, return JWT token or None.
    """
    hashed = ALLOWED_USERS.get(username)
    if hashed and check_password(password, hashed):
        return create_token(username)
    return None


# --- Rate Limiter ---

class RateLimiter:
    """Per-user rate limiting with RPM and daily caps."""

    def __init__(
        self,
        rpm: int = RATE_LIMIT_RPM,
        daily: int = RATE_LIMIT_DAILY,
    ):
        self.rpm = rpm
        self.daily = daily
        # {username: [timestamp, ...]}
        self._minute_log: Dict[str, list] = defaultdict(list)
        self._daily_log: Dict[str, list] = defaultdict(list)

    def check_limit(self, username: str) -> bool:
        """Return True if the user is within rate limits."""
        now = time.time()

        # Clean old entries
        minute_ago = now - 60
        day_ago = now - 86400
        self._minute_log[username] = [
            t for t in self._minute_log[username] if t > minute_ago
        ]
        self._daily_log[username] = [
            t for t in self._daily_log[username] if t > day_ago
        ]

        # Check limits
        if len(self._minute_log[username]) >= self.rpm:
            return False
        if len(self._daily_log[username]) >= self.daily:
            return False

        # Record this request
        self._minute_log[username].append(now)
        self._daily_log[username].append(now)
        return True

    def get_usage(self, username: str) -> Dict:
        """Return current usage stats for a user."""
        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400
        minute_count = len([t for t in self._minute_log.get(username, []) if t > minute_ago])
        daily_count = len([t for t in self._daily_log.get(username, []) if t > day_ago])
        return {
            "rpm_used": minute_count,
            "rpm_limit": self.rpm,
            "daily_used": daily_count,
            "daily_limit": self.daily,
        }
