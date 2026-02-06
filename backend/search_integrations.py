"""
Search Integrations Module
Claude API, Perplexity API (free tier), and Tavily search.
Each provider has a search method and a health check.
"""

import logging
from typing import Dict, Any, List, Optional

import httpx

import base64

from config import (
    get_anthropic_api_key,
    get_perplexity_api_key,
    get_tavily_api_key,
    get_max_tokens,
    get_temperature,
    get_claude_haiku_model,
)
from resilience import retry_with_backoff


def get_idealista_api_key() -> str:
    """Get Idealista API key from environment."""
    import os
    return os.getenv("IDEALISTA_API_KEY", "")


def get_idealista_api_secret() -> str:
    """Get Idealista API secret from environment."""
    import os
    return os.getenv("IDEALISTA_API_SECRET", "")

logger = logging.getLogger(__name__)


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
        """Check if Claude API key is configured (no API call, saves quota)."""
        api_key = get_anthropic_api_key()
        if not api_key or api_key.startswith("your_"):
            return False
        # Key is configured - assume healthy (actual errors caught on query)
        return True

    async def search(self, query: str, model: str = None) -> Dict[str, Any]:
        """
        Ask Claude to answer a question.
        Returns {"answer": str, "sources": list[dict], "usage": dict}.
        """
        client = self._get_client()
        prompt = f"""Answer the following question accurately and concisely.
If you cite any sources, list them at the end.

Question: {query}"""

        # Use local model fallback if specified
        if model is None or model == "local":
            model = get_claude_haiku_model()
        actual_model = model
        api_key = get_anthropic_api_key()

        # System message to prevent unhelpful "I can't browse" responses
        system_msg = """You are a helpful assistant integrated with web search tools.
CRITICAL RULES:
- NEVER say "I cannot visit links", "I cannot browse", or "I don't have web access". Web data has been fetched FOR you.
- NEVER tell the user to "search directly" or "copy and paste". Answer their question using available data.
- If data is limited, give the best answer possible. Do not refuse or redirect.
- Never include citation markers like [1], [2] in your response."""

        try:
            async def _do_post():
                r = await client.post(
                    self.API_URL,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": actual_model,
                        "max_tokens": get_max_tokens(),
                        "temperature": get_temperature(),
                        "system": system_msg,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                r.raise_for_status()
                return r

            resp = await retry_with_backoff(_do_post)
            data = resp.json()

            answer = ""
            if data.get("content"):
                answer = data["content"][0].get("text", "")

            usage = data.get("usage", {})

            return {
                "answer": answer,
                "sources": [
                    {
                        "title": "Claude AI",
                        "url": "https://claude.ai",
                        "snippet": answer[:200] if answer else "",
                    }
                ],
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            }
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Claude API error {e.response.status_code}: {error_body}")
            return {
                "answer": f"[Claude error {e.response.status_code}: {error_body}]",
                "sources": [],
                "usage": {},
            }

    async def generate(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """
        General-purpose generation with Claude.
        Used for synthesizing hybrid answers.
        Returns {"text": str, "usage": dict}.
        """
        client = self._get_client()
        if model is None or model == "local":
            model = get_claude_haiku_model()
        actual_model = model
        api_key = get_anthropic_api_key()
        try:
            async def _do_post():
                r = await client.post(
                    self.API_URL,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": actual_model,
                        "max_tokens": get_max_tokens(),
                        "temperature": get_temperature(),
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                r.raise_for_status()
                return r

            resp = await retry_with_backoff(_do_post)
            data = resp.json()
            text = ""
            if data.get("content"):
                text = data["content"][0].get("text", "")
            usage = data.get("usage", {})
            return {
                "text": text,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            }
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Claude generate error {e.response.status_code}: {error_body}")
            return {
                "text": f"[Claude error {e.response.status_code}: {error_body}]",
                "usage": {},
            }


# ======================================================================
# Perplexity Search (Free Tier — OpenAI-compatible API)
# ======================================================================

class PerplexitySearch:
    """
    Search and research via Perplexity API (2026 Sonar models).

    Models available:
    - sonar: Standard web-grounded search (cheapest)
    - sonar-pro: More thorough search
    - sonar-reasoning-pro: Step-by-step reasoning across sources
    - sonar-deep-research: Exhaustive research (hundreds of sources)

    Parameters:
    - search_recency_filter: "day", "week", "month", "year"
    - depth: "medium" (default) or "high" (for deep research)
    - web_search_options: {"search_mode": "academic"} for scholarly sources

    Citation tokens are free (not charged).
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    # Available models
    MODEL_SONAR = "sonar"                      # Fast, cheap
    MODEL_SONAR_PRO = "sonar-pro"              # Thorough
    MODEL_REASONING = "sonar-reasoning-pro"    # Multi-step reasoning
    MODEL_DEEP_RESEARCH = "sonar-deep-research"  # Exhaustive research

    # Depth levels for deep research model
    DEPTH_MEDIUM = "medium"  # Default, balanced
    DEPTH_HIGH = "high"      # More thorough, more tokens

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=90.0)
        return self._client

    async def is_healthy(self) -> bool:
        """Check if Perplexity API key is configured."""
        api_key = get_perplexity_api_key()
        if not api_key or api_key.startswith("your_"):
            return False
        return True

    def _parse_response(self, data: Dict) -> Dict[str, Any]:
        """Parse Perplexity response and extract answer + citations."""
        answer = ""
        if data.get("choices"):
            answer = data["choices"][0].get("message", {}).get("content", "")

        # Citations are now free (no token charge)
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

        # Include cost info if available (new API feature)
        usage = data.get("usage", {})

        return {"answer": answer, "citations": citations, "usage": usage}

    async def search(
        self,
        query: str,
        search_mode: str = "low",
        recency: Optional[str] = None,
        academic: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Search via Perplexity Sonar model.

        Args:
            query: Search query
            search_mode: "low" (cheap), "medium" (balanced), "high" (thorough)
            recency: Filter by time - "day", "week", "month", "year", or None
            academic: If True, prioritize academic/scholarly sources
            conversation_history: Previous messages for context

        Returns: {"answer": str, "citations": list[dict], "usage": dict}
        """
        client = self._get_client()
        api_key = get_perplexity_api_key()

        # Use sonar-pro for high mode, sonar for low/medium
        model = self.MODEL_SONAR_PRO if search_mode == "high" else self.MODEL_SONAR

        # Build messages with conversation history for context
        messages = [
            {
                "role": "system",
                "content": """You are a web-enabled research assistant with FULL access to browse any URL.

CRITICAL RULES:
- NEVER say "I cannot access", "I cannot browse", "I'm unable to visit", or similar disclaimers
- When given a URL, you CAN and DO access it. Report what you find on that page.
- If a specific page doesn't contain certain information, say "The page doesn't mention X" not "I can't access the page"
- Be direct and factual. Answer the user's specific question about the URL content.
- Include direct links to sources.""",
            },
        ]

        # Add conversation history (last 4 exchanges max to avoid token limits)
        if conversation_history:
            recent_history = conversation_history[-8:]  # Last 4 user+assistant pairs
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current query
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": get_max_tokens(),
            "temperature": 0.2,
        }

        # Build web_search_options
        web_opts = {}
        if recency:
            web_opts["search_recency_filter"] = recency
        if academic:
            web_opts["search_mode"] = "academic"
        if web_opts:
            payload["web_search_options"] = web_opts

        try:
            async def _do_post():
                r = await client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                r.raise_for_status()
                return r

            resp = await retry_with_backoff(_do_post)
            return self._parse_response(resp.json())
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Perplexity search error {e.response.status_code}: {error_body}")
            return {"answer": f"[Perplexity error {e.response.status_code}: {error_body}]", "citations": [], "usage": {}}

    async def focused_search(
        self,
        query: str,
        num_results: int = 10,
        recency: str = "week",
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Focused search optimized for supplier/product queries.
        Returns direct product page URLs with prices.

        Args:
            query: Search query
            num_results: Number of results (5-10 recommended)
            recency: Time filter - "day", "week", "month", "year"
            exclude_domains: Domains to exclude (SEO spam, etc.)

        Returns: {"answer": str, "citations": list[dict], "usage": dict}
        """
        client = self._get_client()
        api_key = get_perplexity_api_key()

        # Default domains to exclude (SEO spam, AI content farms)
        if exclude_domains is None:
            exclude_domains = ["quora.com", "pinterest.com", "reddit.com"]

        # Format domains for API: prefix with "-" to exclude
        domain_filter = [f"-{d}" for d in exclude_domains]

        # Simple, direct prompt that works well with Perplexity
        user_prompt = f"""{query}

Find direct product page URLs with current prices where available.
Focus on US-based suppliers with active product listings.

FORMAT YOUR RESPONSE AS A MARKDOWN TABLE with these columns:
| Supplier | Product | Price | URL |

Example:
| Supplier | Product | Price | URL |
|----------|---------|-------|-----|
| Eden Botanicals | Rose Absolute | $45/5ml | https://edenbotanicals.com/rose |

List at least 5 suppliers. Include the full clickable URL in the URL column."""

        payload = {
            "model": self.MODEL_SONAR_PRO,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": get_max_tokens(),
            "temperature": 0.1,
            "web_search_options": {
                "search_recency_filter": recency,
                "search_domain_filter": domain_filter,
            },
        }

        try:
            resp = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            # Parse response
            answer = ""
            if data.get("choices"):
                answer = data["choices"][0].get("message", {}).get("content", "")

            # Extract citations
            raw_citations = data.get("citations", [])
            citations = []
            for i, url in enumerate(raw_citations):
                citations.append({
                    "title": f"Source {i + 1}",
                    "url": url if isinstance(url, str) else str(url),
                    "snippet": "",
                })

            usage = data.get("usage", {})

            logger.info(f"Focused search returned {len(citations)} citations")

            return {
                "answer": answer,
                "citations": citations,
                "table": answer,  # The answer IS the table
                "usage": usage,
            }

        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Perplexity focused search error {e.response.status_code}: {error_body}")
            return {
                "answer": f"[Perplexity error {e.response.status_code}: {error_body}]",
                "citations": [],
                "table": "",
                "usage": {},
            }

    async def research(
        self,
        query: str,
        depth: str = "high",
        recency: Optional[str] = None,
        academic: bool = False,
        conversation_history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Deep research via Perplexity Sonar Deep Research model.

        Args:
            query: Research query
            depth: "high" (exhaustive) or "medium" (balanced) - passed to sonar-deep-research
            recency: Filter by time - "day", "week", "month", "year"
            academic: If True, prioritize academic sources
            conversation_history: Previous messages for context in follow-ups

        Returns: {"answer": str, "citations": list[dict], "usage": dict}
        """
        # Use sonar-deep-research for exhaustive research
        model = self.MODEL_DEEP_RESEARCH
        api_key = get_perplexity_api_key()

        # Build messages with conversation history for follow-ups
        messages = [
            {
                "role": "system",
                "content": (
                    "Provide a thorough, well-researched answer. "
                    "Include specific details, data, and cite all sources."
                ),
            }
        ]
        # Add recent conversation history (last 6 messages)
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": get_max_tokens(),
            "temperature": 0.1,
            "depth": depth,  # "high" or "medium" for deep research model
        }

        # Build web_search_options
        web_opts = {}
        if recency:
            web_opts["search_recency_filter"] = recency
        if academic:
            web_opts["search_mode"] = "academic"
        if web_opts:
            payload["web_search_options"] = web_opts

        client = self._get_client()
        try:
            resp = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            # Fallback to sonar-pro if sonar-deep-research unavailable
            if resp.status_code != 200 and model == self.MODEL_DEEP_RESEARCH:
                logger.info("sonar-deep-research unavailable, falling back to sonar-pro")
                payload["model"] = self.MODEL_SONAR_PRO
                payload.pop("depth", None)  # Remove depth param (not valid for sonar-pro)
                resp = await client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            resp.raise_for_status()
            return self._parse_response(resp.json())
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Perplexity research error {e.response.status_code}: {error_body}")
            return {"answer": f"[Perplexity error {e.response.status_code}: {error_body}]", "citations": [], "usage": {}}


# ======================================================================
# Tavily Search (Backup)
# ======================================================================

class TavilySearch:
    """
    Web search via Tavily API.
    Better for specific URL retrieval than Perplexity.
    """

    API_URL = "https://api.tavily.com/search"

    # Real estate domains for focused searches
    REAL_ESTATE_DOMAINS = [
        "idealista.com",
        "atemporal.com",
        "spotahome.com",
        "habitaclia.com",
        "fotocasa.es",
        "pisos.com",
    ]

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def is_healthy(self) -> bool:
        """Check if Tavily API key is configured."""
        api_key = get_tavily_api_key()
        return bool(api_key) and not api_key.startswith("your_")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search via Tavily with domain filtering.

        Args:
            query: Search query
            max_results: Max results to return (default 10)
            search_depth: "basic" or "advanced" (advanced gets more specific URLs)
            include_domains: Only search these domains
            exclude_domains: Exclude these domains

        Returns: {"answer": str, "citations": list[dict]} - citations format for compatibility
        """
        client = self._get_client()
        api_key = get_tavily_api_key()

        if not api_key or api_key.startswith("your_"):
            return {
                "answer": "Tavily API key not configured",
                "citations": [],
            }

        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": True,
            "search_depth": search_depth,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            resp = await client.post(self.API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "")

            # Return in citations format for compatibility with Groq agent
            citations = []
            for result in data.get("results", []):
                citations.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:300],
                })

            logger.info(f"Tavily returned {len(citations)} results with URLs")
            for c in citations[:5]:
                logger.info(f"  - {c['url']}")

            return {"answer": answer, "citations": citations}

        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily error: {e.response.status_code}")
            return {"answer": f"[Tavily error: {e.response.status_code}]", "citations": []}

    async def extract(
        self,
        urls: List[str],
    ) -> Dict[str, Any]:
        """
        Extract content from specific URLs using Tavily Extract API.
        This fetches the actual page content, not search results.

        Args:
            urls: List of URLs to extract content from

        Returns: {"answer": str, "citations": list, "raw_content": dict}
        """
        client = self._get_client()
        api_key = get_tavily_api_key()

        if not api_key or api_key.startswith("your_"):
            return {"answer": "Tavily API key not configured", "citations": [], "raw_content": {}}

        payload = {
            "api_key": api_key,
            "urls": urls,
        }

        try:
            resp = await client.post("https://api.tavily.com/extract", json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Build answer from extracted content
            raw_content = {}
            citations = []
            answer_parts = []

            for result in data.get("results", []):
                url = result.get("url", "")
                content = result.get("raw_content", "")
                raw_content[url] = content

                # Summarize for answer
                if content:
                    # Use up to 15K chars, skipping nav boilerplate
                    excerpt = content
                    if len(content) > 5000:
                        # Try to find where product content starts
                        for marker in ["About this item", "Product Description", "Product details", "Description"]:
                            idx = content.find(marker)
                            if idx != -1:
                                excerpt = content[max(0, idx - 300):]
                                break
                        else:
                            # Skip first 30% (usually nav)
                            excerpt = content[len(content) // 3:]
                    answer_parts.append(f"Content from {url}:\n{excerpt[:15000]}")

                citations.append({
                    "title": result.get("title", url),
                    "url": url,
                    "snippet": content[:500] if content else "",
                })

            answer = "\n\n".join(answer_parts) if answer_parts else "No content extracted"
            logger.info(f"Tavily extract: fetched {len(citations)} URLs")
            for c in citations:
                logger.info(f"  - {c['url']} ({len(raw_content.get(c['url'], ''))} chars)")

            return {"answer": answer, "citations": citations, "raw_content": raw_content}

        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily extract error: {e.response.status_code}")
            return {"answer": f"[Tavily extract error: {e.response.status_code}]", "citations": [], "raw_content": {}}
        except Exception as e:
            logger.error(f"Tavily extract failed: {e}")
            return {"answer": f"[Tavily extract failed: {e}]", "citations": [], "raw_content": {}}

    async def search_real_estate(
        self,
        query: str,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Search real estate listings with focused domain filtering.
        Returns specific listing URLs from Idealista, Atemporal, etc.
        """
        return await self.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=self.REAL_ESTATE_DOMAINS,
        )


# ======================================================================
# Idealista API (Direct Real Estate Listings)
# ======================================================================

class IdealistaSearch:
    """
    Direct access to Idealista listings via their official API.
    Requires API key from https://developers.idealista.com/access-request

    Returns actual listing URLs with prices, features, and photos.
    """

    TOKEN_URL = "https://api.idealista.com/oauth/token"
    SEARCH_URL = "https://api.idealista.com/3.5/{country}/search"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._token: Optional[str] = None
        self._token_expires: float = 0

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def is_healthy(self) -> bool:
        """Check if Idealista API credentials are configured."""
        api_key = get_idealista_api_key()
        api_secret = get_idealista_api_secret()
        return bool(api_key and api_secret and
                    not api_key.startswith("your_") and
                    not api_secret.startswith("your_"))

    async def _get_token(self) -> Optional[str]:
        """Get OAuth token for Idealista API."""
        import time

        # Return cached token if still valid
        if self._token and time.time() < self._token_expires:
            return self._token

        api_key = get_idealista_api_key()
        api_secret = get_idealista_api_secret()

        if not api_key or not api_secret:
            logger.warning("Idealista API credentials not configured")
            return None

        # Create Basic auth header
        credentials = f"{api_key}:{api_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        client = self._get_client()
        try:
            resp = await client.post(
                self.TOKEN_URL,
                headers={
                    "Authorization": f"Basic {encoded}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "client_credentials",
                    "scope": "read",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            self._token = data.get("access_token")
            # Token typically valid for 1 hour, refresh 5 min early
            self._token_expires = time.time() + data.get("expires_in", 3600) - 300

            logger.info("Obtained Idealista API token")
            return self._token

        except httpx.HTTPStatusError as e:
            logger.error(f"Idealista token error: {e.response.status_code} - {e.response.text}")
            return None

    async def search(
        self,
        location: str = "barcelona",
        country: str = "es",
        operation: str = "rent",
        property_type: str = "homes",
        max_price: Optional[int] = None,
        min_size: Optional[int] = None,
        bedrooms: Optional[int] = None,
        has_terrace: bool = False,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """
        Search Idealista listings directly.

        Args:
            location: City or area name (e.g., "barcelona", "madrid")
            country: Country code (es, it, pt)
            operation: "rent" or "sale"
            property_type: "homes", "premises", "garages", etc.
            max_price: Maximum price in euros
            min_size: Minimum size in m²
            bedrooms: Number of bedrooms (0=studio, 1, 2, 3, 4+)
            has_terrace: Filter for properties with terrace/balcony
            max_results: Maximum listings to return

        Returns: {"answer": str, "citations": list[dict], "listings": list[dict]}
        """
        token = await self._get_token()
        if not token:
            return {
                "answer": "Idealista API not configured. Request access at https://developers.idealista.com/access-request",
                "citations": [],
                "listings": [],
            }

        # Build search parameters
        # Note: Idealista uses center point + distance, so we need geocoding
        # For now, use pre-defined centers for common cities
        centers = {
            "barcelona": "41.3851,2.1734",
            "madrid": "40.4168,-3.7038",
            "valencia": "39.4699,-0.3763",
            "lisbon": "38.7223,-9.1393",
            "porto": "41.1579,-8.6291",
            "milan": "45.4642,9.1900",
            "rome": "41.9028,12.4964",
        }

        center = centers.get(location.lower(), centers["barcelona"])

        params = {
            "center": center,
            "distance": 10000,  # 10km radius
            "operation": operation,
            "propertyType": property_type,
            "maxItems": max_results,
            "numPage": 1,
            "order": "priceDown",
            "sort": "desc",
        }

        if max_price:
            params["maxPrice"] = max_price
        if min_size:
            params["minSize"] = min_size
        if bedrooms is not None:
            if bedrooms == 0:
                params["studio"] = True
            else:
                params["bedrooms"] = f"{bedrooms},{bedrooms}"
        if has_terrace:
            params["terrace"] = True

        client = self._get_client()
        try:
            url = self.SEARCH_URL.format(country=country)
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=params,
            )
            resp.raise_for_status()
            data = resp.json()

            listings = []
            citations = []

            for item in data.get("elementList", []):
                listing = {
                    "id": item.get("propertyCode"),
                    "url": item.get("url", f"https://www.idealista.com/inmueble/{item.get('propertyCode')}/"),
                    "price": item.get("price"),
                    "size": item.get("size"),
                    "rooms": item.get("rooms"),
                    "bathrooms": item.get("bathrooms"),
                    "address": item.get("address"),
                    "district": item.get("district"),
                    "neighborhood": item.get("neighborhood"),
                    "has_terrace": item.get("hasTerrace", False),
                    "has_lift": item.get("hasLift", False),
                    "floor": item.get("floor"),
                    "description": item.get("description", "")[:200],
                    "thumbnail": item.get("thumbnail"),
                }
                listings.append(listing)

                # Also add to citations for Groq agent compatibility
                citations.append({
                    "title": f"€{listing['price']}/mo - {listing['rooms'] or 'Studio'} bed, {listing['size']}m² in {listing['neighborhood'] or listing['district']}",
                    "url": listing["url"],
                    "snippet": f"{listing['description'][:150]}... Terrace: {'Yes' if listing['has_terrace'] else 'No'}",
                })

            # Build summary answer
            total = data.get("total", len(listings))
            answer = f"Found {total} listings. Showing {len(listings)} results:\n\n"
            for i, l in enumerate(listings[:5], 1):
                answer += f"{i}. €{l['price']}/mo - {l['rooms'] or 'Studio'} bed, {l['size']}m² in {l['neighborhood'] or l['district']}\n"
                answer += f"   {l['url']}\n"

            logger.info(f"Idealista returned {len(listings)} listings")

            return {
                "answer": answer,
                "citations": citations,
                "listings": listings,
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"Idealista search error: {e.response.status_code} - {e.response.text}")
            return {
                "answer": f"[Idealista error: {e.response.status_code}]",
                "citations": [],
                "listings": [],
            }


# ======================================================================
# Crawl4AI Search (URL Scraping with Playwright JS Rendering)
# ======================================================================

class Crawl4AISearch:
    """
    URL scraper using Crawl4AI + Playwright for JS rendering and clean markdown output.
    Better than Tavily for:
    - JavaScript-heavy SPAs
    - Clean markdown output (not raw HTML)
    - Free (no per-request cost)
    """

    def __init__(self):
        self._crawler = None

    async def is_healthy(self) -> bool:
        """Check if Crawl4AI is available."""
        try:
            from crawl4ai import AsyncWebCrawler
            return True
        except ImportError:
            logger.warning("Crawl4AI not installed - run: pip install crawl4ai")
            return False

    async def extract(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Extract clean content from URL using Playwright for JS rendering.

        Args:
            url: URL to scrape
            timeout: Request timeout in seconds

        Returns: {"success": bool, "answer": str, "raw_content": dict, "metadata": dict}
        """
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

            # Detect e-commerce / JS-heavy sites that need special handling
            url_lower = url.lower()
            is_ecommerce = any(d in url_lower for d in [
                "amazon.com", "amazon.", "ebay.com", "walmart.com",
                "etsy.com", "aliexpress.com", "target.com",
            ])

            # Configure based on site type
            if is_ecommerce:
                config = CrawlerRunConfig(
                    wait_until="networkidle",          # Wait for all AJAX to finish
                    delay_before_return_html=3.0,      # Extra 3s for dynamic content
                    scan_full_page=True,               # Scroll to load lazy content
                    excluded_selector="nav, header, footer, #navbar, #nav-belt, #skiplink, .nav-sprite, #rhf",
                    remove_forms=True,
                    exclude_external_images=True,
                    word_count_threshold=1,
                    page_timeout=45000,
                )
                logger.info(f"Crawl4AI: using e-commerce config for {url[:60]}")
            else:
                config = CrawlerRunConfig(
                    wait_until="domcontentloaded",
                    delay_before_return_html=0.5,
                    page_timeout=30000,
                )

            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=config)

                if not result.success:
                    logger.warning(f"Crawl4AI failed for {url}: extraction unsuccessful")
                    return {
                        "success": False,
                        "error": "Extraction failed",
                        "answer": "",
                        "raw_content": {},
                        "citations": [],
                    }

                # Get clean markdown (preferred) or cleaned HTML
                content = result.markdown or result.cleaned_html or ""

                logger.info(f"Crawl4AI extracted {len(content)} chars from {url}")

                # Smart truncation: skip nav/header boilerplate, find product content
                useful_content = content
                if len(content) > 10000:
                    # Look for product-content markers
                    content_markers = [
                        "## About this item", "About this item",
                        "## Product Description", "Product Description",
                        "## Product details", "Product details",
                        "## Product information", "Product information",
                        "## Features", "## Specifications",
                        "## Description",
                    ]
                    best_start = None
                    for marker in content_markers:
                        idx = content.find(marker)
                        if idx != -1 and (best_start is None or idx < best_start):
                            best_start = idx

                    if best_start is not None:
                        start = max(0, best_start - 500)
                        useful_content = content[start:]
                        logger.info(f"Crawl4AI: found content marker at char {best_start}, using from {start}")
                    else:
                        # No markers found -- find the first substantial text block
                        # (paragraphs with 50+ words, not just headers/links)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            word_count = len(line.split())
                            if word_count >= 15 and not line.startswith(('![', '[', '#', '|', '<')):
                                start = max(0, content.find(line) - 200)
                                useful_content = content[start:]
                                logger.info(f"Crawl4AI: found first text block at line {i}, char {start}")
                                break
                        else:
                            # Last resort: skip first 40%
                            skip = int(len(content) * 0.4)
                            useful_content = content[skip:]
                            logger.info(f"Crawl4AI: no text blocks found, skipping first {skip} chars")

                # Allow up to 20K chars for the answer (Groq has 128K context)
                answer = useful_content[:20000] if useful_content else ""

                return {
                    "success": True,
                    "answer": answer,
                    "raw_content": {url: content},
                    "citations": [{
                        "title": result.metadata.get("title", url) if result.metadata else url,
                        "url": url,
                        "snippet": answer[:300] if answer else "",
                    }],
                    "metadata": {
                        "title": result.metadata.get("title", "") if result.metadata else "",
                        "description": result.metadata.get("description", "") if result.metadata else "",
                    },
                }

        except ImportError:
            logger.error("Crawl4AI not installed")
            return {
                "success": False,
                "error": "Crawl4AI not installed - run: pip install crawl4ai",
                "answer": "",
                "raw_content": {},
                "citations": [],
            }
        except Exception as e:
            logger.error(f"Crawl4AI extraction failed for {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "raw_content": {},
                "citations": [],
            }
