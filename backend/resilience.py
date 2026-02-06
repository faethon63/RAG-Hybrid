"""Resilience patterns for external API calls."""
import asyncio
import logging
import random

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 502, 503}


async def retry_with_backoff(
    coro_func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retryable_statuses: set = None,
    **kwargs
):
    """
    Retry an async function with exponential backoff + jitter.
    Only retries on specific HTTP status codes.
    """
    retryable = retryable_statuses or RETRYABLE_STATUS_CODES
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            # Check if this is a retryable HTTP error
            status_code = None
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code

            if status_code not in retryable or attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            total_delay = delay + jitter

            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after {status_code} error. "
                f"Waiting {total_delay:.1f}s. Error: {e}"
            )
            await asyncio.sleep(total_delay)

    raise last_exception
