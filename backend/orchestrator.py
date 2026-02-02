"""
Query Orchestrator Module
Uses Groq (Llama 3.3 70B) for FREE orchestration - better than Haiku at tool use.
Updated 2026-01-31: Switched from Haiku to Groq for cost savings and better tool routing.
"""

import logging
import httpx
import json
import os
from typing import Dict, Optional
from datetime import datetime

from query_classifier import QueryClassifier, QueryIntent

logger = logging.getLogger(__name__)


def get_groq_api_key() -> str:
    """Get Groq API key from environment."""
    return os.getenv("GROQ_API_KEY", "")


class QueryOrchestrator:
    """Uses Groq (Llama 3.3 70B) to analyze queries and route to optimal model - FREE."""

    # Model definitions (loaded from env vars with defaults)
    MODEL_OPUS = os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-5-20251101")
    MODEL_SONNET = os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-5-20250929")
    MODEL_HAIKU = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")
    MODEL_LOCAL = "local"  # Ollama - only for static knowledge chat
    MODEL_PERPLEXITY = "perplexity"  # Routes to Perplexity API
    MODEL_DEEP_AGENT = "deep_agent"  # Routes to smolagents multi-step agent

    # Groq model for orchestration (free, fast, excellent at tool use)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    GROQ_ROUTING_PROMPT = """You are a query router. Analyze the query and decide which service should handle it.

Available options:
- LOCAL: Free local LLM (Ollama). Use ONLY for: static knowledge from training data (pre-2024), casual chat, greetings, simple factual questions that don't need current info.
- PERPLEXITY_LOW: Web search. Use for: current events, prices, news, recent data, anything needing real-time or 2024+ information, travel planning, rentals, financial info.
- PERPLEXITY_HIGH: Deep web research. Use for: comprehensive research, multi-source analysis, detailed reports.
- SONNET: Claude Sonnet. Use for: complex reasoning, analysis, coding, writing that doesn't need web search.
- OPUS: Claude Opus. Use for: very complex multi-step reasoning, legal analysis, critical decisions, difficult problems.

ROUTING RULES:
1. If query asks about CURRENT prices, events, news, dates, or anything time-sensitive → PERPLEXITY_LOW
2. If query mentions specific years (2024, 2025, 2026), locations for travel/rentals, or "best time to" → PERPLEXITY_LOW
3. If query is about travel, moving, rentals, financial planning, market conditions → PERPLEXITY_LOW
4. If query is simple greeting (hi, hello, thanks) → LOCAL
5. If query is about static facts from before 2024 (science, history, math, definitions) → LOCAL
6. If query needs complex reasoning but not current data → SONNET
7. If query is extremely complex, legal, or high-stakes → OPUS
8. When in doubt between LOCAL and PERPLEXITY, choose PERPLEXITY (better to have current info)

Today's date: {current_date}

Respond with ONLY valid JSON, no other text:
{{"model": "LOCAL|PERPLEXITY_LOW|PERPLEXITY_HIGH|SONNET|OPUS", "reason": "brief reason"}}

Query: {query}"""

    def __init__(self):
        self._http_client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def analyze_query(self, query: str, project: Optional[str] = None) -> Dict:
        """
        Use Groq (FREE) to analyze query and return recommended model.
        Falls back to Perplexity if Groq unavailable.
        """
        query_lower = query.lower()

        # 0. Classify query intent first
        classification = QueryClassifier.classify(query)

        # Route automation requests to deep_agent
        if classification["intent"] == QueryIntent.AUTOMATION:
            logger.info(f"Classifier: automation request -> DEEP_AGENT")
            return {
                "model": self.MODEL_DEEP_AGENT,
                "reason": "Automation task requiring multi-step execution",
                "complexity": "high",
                "classification": classification,
            }

        # 1. Fast-path: simple greetings use LOCAL
        greetings = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"]
        if any(query_lower.strip() == g or query_lower.startswith(g + " ") or query_lower.startswith(g + ",") for g in greetings):
            logger.info(f"Fast-path to LOCAL: greeting")
            return {
                "model": self.MODEL_LOCAL,
                "reason": "Simple greeting",
                "complexity": "low",
                "classification": classification,
            }

        # 2. Ask Groq to classify the query (FREE!)
        try:
            result = await self._ask_groq(query, classification)
            if result:
                result["classification"] = classification
                return result
        except Exception as e:
            logger.warning(f"Groq routing failed: {e}, falling back to Perplexity")

        # 3. Fallback: when in doubt, use Perplexity (cheap and accurate)
        return {
            "model": self.MODEL_PERPLEXITY,
            "reason": "Fallback to web search for accuracy",
            "complexity": "medium",
            "search_mode": "low",
            "classification": classification,
        }

    async def _ask_groq(self, query: str, classification: Dict = None) -> Optional[Dict]:
        """Call Groq (FREE) to classify the query."""
        api_key = get_groq_api_key()
        if not api_key:
            logger.warning("No Groq API key configured, falling back")
            return None

        current_date = datetime.now().strftime("%B %d, %Y")
        prompt = self.GROQ_ROUTING_PROMPT.format(
            current_date=current_date,
            query=query
        )

        client = await self._get_client()
        try:
            response = await client.post(
                self.GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            data = response.json()

            text = ""
            if data.get("choices"):
                text = data["choices"][0].get("message", {}).get("content", "").strip()

            logger.debug(f"Groq routing response: {text}")

            # Parse JSON from response
            try:
                # Handle potential markdown code blocks
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                result = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Groq response: {text}")
                return None

            # Map model name to model ID and search mode
            model_name = result.get("model", "PERPLEXITY_LOW").upper()

            # Handle Perplexity with search modes
            if model_name.startswith("PERPLEXITY"):
                search_mode = "low"
                if "HIGH" in model_name:
                    search_mode = "high"
                elif "MEDIUM" in model_name:
                    search_mode = "medium"

                logger.info(f"Groq routed to PERPLEXITY ({search_mode}): {result.get('reason', 'no reason')}")
                return {
                    "model": self.MODEL_PERPLEXITY,
                    "reason": result.get("reason", "Needs web search"),
                    "complexity": "medium" if search_mode == "low" else "high",
                    "search_mode": search_mode,
                }

            # Map other models
            model_map = {
                "LOCAL": self.MODEL_LOCAL,
                "HAIKU": self.MODEL_HAIKU,
                "SONNET": self.MODEL_SONNET,
                "OPUS": self.MODEL_OPUS,
            }

            model_id = model_map.get(model_name, self.MODEL_PERPLEXITY)
            logger.info(f"Groq routed to {model_name}: {result.get('reason', 'no reason')}")

            return {
                "model": model_id,
                "reason": result.get("reason", "Groq routing"),
                "complexity": "high" if model_name in ["OPUS", "SONNET"] else "medium",
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"Groq routing HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Groq routing error: {e}")
            return None


# Module-level instance for easy import
orchestrator = QueryOrchestrator()
