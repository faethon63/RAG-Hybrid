"""
Search Integrations Module
Claude API, Perplexity API (free tier), and Tavily search.
Each provider has a search method and a health check.
"""

import os
import logging
from typing import Dict, Any, List, Optional

import httpx
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

# --- Configuration ---

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


# ======================================================================
# Claude Search (Anthropic API)
# ======================================================================

class ClaudeSearch:
    """Web search and answer generation via Claude API."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def is_healthy(self) -> bool:
        """Check if Claude API key is configured and reachable."""
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("your_"):
            return False
        try:
            client = self._get_client()
            # Light check: send a tiny message
            resp = await client.post(
                self.API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Claude health check failed: {e}")
            return False

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Ask Claude to answer a question.
        Returns {"answer": str, "sources": list[dict]}.
        """
        client = self._get_client()
        prompt = f"""Answer the following question accurately and concisely.
If you cite any sources, list them at the end.

Question: {query}"""

        resp = await client.post(
            self.API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()

        answer = ""
        if data.get("content"):
            answer = data["content"][0].get("text", "")

        return {
            "answer": answer,
            "sources": [
                {
                    "title": "Claude AI",
                    "url": "https://claude.ai",
                    "snippet": answer[:200] if answer else "",
                }
            ],
        }

    async def generate(self, prompt: str) -> str:
        """
        General-purpose generation with Claude.
        Used for synthesizing hybrid answers.
        """
        client = self._get_client()
        resp = await client.post(
            self.API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("content"):
            return data["content"][0].get("text", "")
        return ""


# ======================================================================
# Perplexity Search (Free Tier — OpenAI-compatible API)
# ======================================================================

class PerplexitySearch:
    """
    Search and research via Perplexity API.
    Uses the OpenAI-compatible endpoint. Free tier has rate limits
    but does not require a subscription.
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def is_healthy(self) -> bool:
        """Check if Perplexity API key is configured."""
        if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY.startswith("your_"):
            return False
        try:
            client = self._get_client()
            resp = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Perplexity health check failed: {e}")
            return False

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Quick search via Perplexity sonar model.
        Returns {"answer": str, "citations": list[dict]}.
        """
        client = self._get_client()
        resp = await client.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer concisely with citations where possible.",
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.2,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        answer = ""
        if data.get("choices"):
            answer = data["choices"][0].get("message", {}).get("content", "")

        # Perplexity returns citations in the response
        raw_citations = data.get("citations", [])
        citations = []
        for i, url in enumerate(raw_citations):
            citations.append({
                "title": f"Source {i + 1}",
                "url": url if isinstance(url, str) else str(url),
                "snippet": "",
            })

        if not citations:
            citations.append({
                "title": "Perplexity AI",
                "url": "https://perplexity.ai",
                "snippet": answer[:200] if answer else "",
            })

        return {"answer": answer, "citations": citations}

    async def research(self, query: str, depth: str = "deep") -> Dict[str, Any]:
        """
        Deep research query via Perplexity sonar-pro model.
        Falls back to sonar if sonar-pro is not available on free tier.
        """
        # Try sonar-pro first (deeper analysis), fall back to sonar
        model = "sonar-pro" if depth == "deep" else "sonar"

        client = self._get_client()
        try:
            resp = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Provide a thorough, well-researched answer. "
                                "Include specific details, data, and cite sources."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.1,
                },
            )

            # If sonar-pro fails (not available on free tier), retry with sonar
            if resp.status_code != 200 and model == "sonar-pro":
                logger.info("sonar-pro unavailable, falling back to sonar")
                resp = await client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Provide a thorough, well-researched answer. "
                                    "Include specific details, data, and cite sources."
                                ),
                            },
                            {"role": "user", "content": query},
                        ],
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0.1,
                    },
                )

            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return {
                "answer": f"[Perplexity API error: {e.response.status_code}]",
                "citations": [],
            }

        answer = ""
        if data.get("choices"):
            answer = data["choices"][0].get("message", {}).get("content", "")

        raw_citations = data.get("citations", [])
        citations = []
        for i, url in enumerate(raw_citations):
            citations.append({
                "title": f"Source {i + 1}",
                "url": url if isinstance(url, str) else str(url),
                "snippet": "",
            })

        if not citations:
            citations.append({
                "title": "Perplexity AI",
                "url": "https://perplexity.ai",
                "snippet": answer[:200] if answer else "",
            })

        return {"answer": answer, "citations": citations}


# ======================================================================
# Tavily Search (Backup)
# ======================================================================

class TavilySearch:
    """Backup web search via Tavily API."""

    API_URL = "https://api.tavily.com/search"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def is_healthy(self) -> bool:
        """Check if Tavily API key is configured."""
        return bool(TAVILY_API_KEY) and not TAVILY_API_KEY.startswith("your_")

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search via Tavily.
        Returns {"answer": str, "sources": list[dict]}.
        """
        client = self._get_client()
        resp = await client.post(
            self.API_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "search_depth": "basic",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "")
        sources = []
        for result in data.get("results", []):
            sources.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:200],
            })

        return {"answer": answer, "sources": sources}
