"""
Authentication & Rate Limiting Module
JWT token verification, bcrypt password checking, per-user rate limiting.
"""

import time
import jwt
import bcrypt
from typing import Optional, Dict
from collections import defaultdict

from config import (
    get_jwt_secret,
    get_rate_limit_rpm,
    get_rate_limit_daily,
    get_allowed_users,
)

JWT_ALGORITHM = "HS256"


# --- JWT Token Functions ---

def create_token(username: str) -> str:
    """Create a JWT token for a user."""
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400 * 30,  # 30 days
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """
    Verify a JWT token and return the username, or None if invalid.
    """
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
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
    allowed_users = get_allowed_users()
    hashed = allowed_users.get(username)
    if hashed and check_password(password, hashed):
        return create_token(username)
    return None


# --- Rate Limiter ---

class RateLimiter:
    """Per-user rate limiting with RPM and daily caps."""

    def __init__(
        self,
        rpm: Optional[int] = None,
        daily: Optional[int] = None,
    ):
        if rpm is None:
            rpm = get_rate_limit_rpm()
        if daily is None:
            daily = get_rate_limit_daily()
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
