"""
Push Notification Service
Manages VAPID keys, push subscriptions, and sending notifications.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Storage paths
_DATA_DIR = Path(__file__).parent.parent / "data"
_SUBSCRIPTIONS_FILE = _DATA_DIR / "push_subscriptions.json"
_NOTIFICATION_QUEUE_FILE = _DATA_DIR / "notification_queue.json"


def _get_vapid_keys() -> Dict[str, str]:
    """Get or generate VAPID key pair. Stored in .env."""
    from config import _ensure_env_loaded
    _ensure_env_loaded()

    public_key = os.getenv("VAPID_PUBLIC_KEY", "")
    private_key = os.getenv("VAPID_PRIVATE_KEY", "")

    if public_key and private_key:
        return {"public_key": public_key, "private_key": private_key}

    # Generate new VAPID keys
    try:
        from py_vapid import Vapid
        vapid = Vapid()
        vapid.generate_keys()
        public_key = vapid.public_key_urlsafe_base64
        private_key = vapid.private_key_urlsafe_base64

        # Write to .env
        env_path = Path(__file__).parent.parent / "config" / ".env"
        if env_path.exists():
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Push notification VAPID keys (auto-generated)\n")
                f.write(f"VAPID_PUBLIC_KEY={public_key}\n")
                f.write(f"VAPID_PRIVATE_KEY={private_key}\n")
            logger.info("Generated and saved VAPID keys to .env")
        else:
            logger.warning(f"Cannot save VAPID keys: {env_path} not found. Set VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY manually.")

        return {"public_key": public_key, "private_key": private_key}
    except Exception as e:
        logger.error(f"Failed to generate VAPID keys: {e}")
        return {"public_key": "", "private_key": ""}


def get_vapid_public_key() -> str:
    """Get the VAPID public key for frontend subscription."""
    return _get_vapid_keys()["public_key"]


def _load_subscriptions() -> List[Dict]:
    """Load push subscriptions from disk."""
    try:
        if _SUBSCRIPTIONS_FILE.exists():
            with open(_SUBSCRIPTIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load subscriptions: {e}")
    return []


def _save_subscriptions(subs: List[Dict]):
    """Save push subscriptions to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SUBSCRIPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(subs, f, indent=2)


def add_subscription(subscription: Dict) -> bool:
    """Add a new push subscription. Returns True if new, False if duplicate."""
    subs = _load_subscriptions()
    endpoint = subscription.get("endpoint", "")

    # Check for duplicate
    for s in subs:
        if s.get("endpoint") == endpoint:
            # Update existing
            s.update(subscription)
            _save_subscriptions(subs)
            return False

    subs.append(subscription)
    _save_subscriptions(subs)
    logger.info(f"New push subscription registered (total: {len(subs)})")
    return True


def remove_subscription(endpoint: str) -> bool:
    """Remove a push subscription by endpoint."""
    subs = _load_subscriptions()
    original_len = len(subs)
    subs = [s for s in subs if s.get("endpoint") != endpoint]
    if len(subs) < original_len:
        _save_subscriptions(subs)
        return True
    return False


def get_subscription_count() -> int:
    """Get number of active push subscriptions."""
    return len(_load_subscriptions())


async def send_push_notification(title: str, body: str, url: str = "/", tag: str = None) -> Dict:
    """Send push notification to all subscribers."""
    keys = _get_vapid_keys()
    if not keys["public_key"] or not keys["private_key"]:
        return {"sent": 0, "failed": 0, "error": "VAPID keys not configured"}

    subs = _load_subscriptions()
    if not subs:
        # Queue the notification for later (non-PWA fallback)
        _queue_notification(title, body, url, tag)
        return {"sent": 0, "failed": 0, "queued": True, "message": "No subscribers, notification queued"}

    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        return {"sent": 0, "failed": 0, "error": "pywebpush not installed"}

    payload = json.dumps({
        "title": title,
        "body": body,
        "url": url,
        "tag": tag or "rag-notification",
    })

    vapid_claims = {
        "sub": "mailto:notifications@rag-hybrid.local",
    }

    sent = 0
    failed = 0
    dead_endpoints = []

    for sub in subs:
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=keys["private_key"],
                vapid_claims=vapid_claims,
            )
            sent += 1
        except WebPushException as e:
            logger.warning(f"Push failed for {sub.get('endpoint', '')[:50]}...: {e}")
            # Check for expired subscription - try response object first, fallback to string
            is_dead = False
            if hasattr(e, 'response') and e.response:
                try:
                    is_dead = e.response.status_code in (404, 410)
                except AttributeError:
                    pass
            if not is_dead:
                err_str = str(e)
                is_dead = "410" in err_str or "404" in err_str
            if is_dead:
                dead_endpoints.append(sub.get("endpoint"))
            failed += 1
        except Exception as e:
            logger.error(f"Push error: {e}")
            failed += 1

    # Clean up dead subscriptions
    if dead_endpoints:
        for ep in dead_endpoints:
            remove_subscription(ep)
        logger.info(f"Removed {len(dead_endpoints)} expired push subscriptions")

    return {"sent": sent, "failed": failed}


def _queue_notification(title: str, body: str, url: str = "/", tag: str = None):
    """Queue a notification for non-PWA clients to poll."""
    from datetime import datetime
    queue = _load_notification_queue()
    queue.append({
        "title": title,
        "body": body,
        "url": url,
        "tag": tag,
        "timestamp": datetime.utcnow().isoformat(),
        "read": False,
    })
    # Keep only last 50 notifications
    queue = queue[-50:]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_NOTIFICATION_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def _load_notification_queue() -> List[Dict]:
    """Load notification queue from disk."""
    try:
        if _NOTIFICATION_QUEUE_FILE.exists():
            with open(_NOTIFICATION_QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def get_pending_notifications(mark_read: bool = True) -> List[Dict]:
    """Get unread notifications (fallback for non-PWA clients)."""
    queue = _load_notification_queue()
    pending = [n for n in queue if not n.get("read")]
    if mark_read and pending:
        for n in queue:
            n["read"] = True
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_NOTIFICATION_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)
    return pending
