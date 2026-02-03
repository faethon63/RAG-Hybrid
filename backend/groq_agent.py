"""
Groq Conversational Agent
Groq (Llama 3.3 70B) as the main conversational brain with tool access.
This is the orchestrator that maintains context and calls tools as needed.
"""

import logging
import httpx
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


class GroqAgent:
    """
    Groq-powered conversational agent that:
    1. Maintains conversation context
    2. Decides when to use tools (web search, file access, delegate to Claude)
    3. Synthesizes responses using tool results
    """

    # Use Llama 4 Scout for reliable tool calling (recommended by Groq)
    GROQ_MODEL = os.getenv("GROQ_TOOL_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # Tool definitions for Groq
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Use for: current events, prices, news, real-time data, anything after 2024, travel info, financial data. For real estate listings, use search_listings instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. MUST include ALL user criteria: price limits, location, features, etc."
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["perplexity", "perplexity_pro", "tavily"],
                            "description": "Search provider. Use 'perplexity_pro' when user asks for 'deep search', 'thorough search', or 'use perplexity pro'. Use 'tavily' for specific URL needs. Default: perplexity"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_listings",
                "description": "Search for real estate listings (apartments, houses for rent or sale). Returns specific listing URLs with prices. Use for: apartment rentals, house hunting, property searches, real estate in any city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search. Include: city, price limit, bedrooms, features (balcony, near beach, etc.)"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["tavily", "idealista"],
                            "description": "Use 'idealista' for Spain/Portugal/Italy (returns direct listing URLs). Use 'tavily' for other countries or if Idealista unavailable. Default: tavily"
                        },
                        "city": {
                            "type": "string",
                            "description": "City name for Idealista API (barcelona, madrid, lisbon, etc.)"
                        },
                        "max_price": {
                            "type": "string",
                            "description": "Maximum price in euros as a number (e.g., '1400')"
                        },
                        "bedrooms": {
                            "type": "string",
                            "description": "Number of bedrooms as a number (0 for studio, 1, 2, 3, etc.)"
                        },
                        "has_terrace": {
                            "type": "boolean",
                            "description": "Filter for properties with balcony/terrace"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "deep_research",
                "description": "Conduct deep web research with Perplexity Pro (sonar-pro). Use for: comprehensive analysis, comparing options, detailed reports. More thorough than web_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The research query. Be detailed about what information is needed."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "complex_reasoning",
                "description": "Delegate to Claude for complex reasoning, analysis, coding, or writing tasks that don't need web search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task requiring complex reasoning"
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant context from the conversation"
                        }
                    },
                    "required": ["task"]
                }
            }
        },
    ]

    SYSTEM_PROMPT = """You are a helpful AI assistant. Today's date is {current_date}.

MANDATORY: For ANY question about CURRENT DATA (prices, weather, news, events, statistics), you MUST use the web_search tool FIRST. Do NOT answer from memory - your training data is outdated.

TOOL SELECTION GUIDE:
- CURRENT DATA (prices, weather, stocks, crypto, news, sports scores): ALWAYS use web_search tool
- REAL ESTATE (apartments, rentals, listings): Use search_listings tool
  - For Spain/Portugal/Italy: use provider="idealista"
  - For other locations: use provider="tavily"
- COMPREHENSIVE RESEARCH: Use deep_research tool

ABSOLUTE RULES - VIOLATION IS UNACCEPTABLE:
1. NEVER make up numbers, prices, or statistics. If the tool didn't return specific data, say "I couldn't find that specific information."
2. QUOTE EXACTLY from tool results. Do not round, estimate, or paraphrase numerical data.
3. If tool results show multiple different values, report ALL of them with their sources.
4. Include source URLs for every fact you state.
5. If you're uncertain, SAY SO. Never guess.

GOOD RESPONSE for price query:
"According to [source], Bitcoin is currently $78,875.00 USD.
Source: https://coinmarketcap.com/..."

BAD RESPONSE (NEVER DO THIS):
"Bitcoin is around $63,000" (Making up a number not in the tool results)

{project_instructions}"""

    def __init__(self):
        self._http_client = None
        self._tool_handlers = {}

    def register_tool_handler(self, name: str, handler):
        """Register a function to handle tool calls."""
        self._tool_handlers[name] = handler

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def chat(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None,
        project_config: Optional[Dict] = None,
        max_tool_calls: int = 3,
    ) -> Dict[str, Any]:
        """
        Main chat method. Groq processes the query, optionally calls tools,
        and returns a synthesized response.

        Returns: {"answer": str, "sources": list, "tool_calls": list, "usage": dict}
        """
        api_key = get_groq_api_key()
        if not api_key:
            logger.warning("No Groq API key, falling back to direct response")
            return {
                "answer": "Groq API key not configured. Please add GROQ_API_KEY to .env",
                "sources": [],
                "tool_calls": [],
                "usage": {},
            }

        # Build system prompt with project context
        current_date = datetime.now().strftime("%B %d, %Y")
        project_instructions = ""
        if project_config:
            if project_config.get("system_prompt"):
                project_instructions = f"\nProject context: {project_config['system_prompt']}"

        system_prompt = self.SYSTEM_PROMPT.format(
            current_date=current_date,
            project_instructions=project_instructions,
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 5 exchanges
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current query
        messages.append({"role": "user", "content": query})

        # Track tool calls and sources
        all_tool_calls = []
        all_sources = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        # Store raw Perplexity answer for direct passthrough (prevents hallucination)
        perplexity_direct_answer = None

        # Agentic loop - Groq may call tools multiple times
        for iteration in range(max_tool_calls + 1):
            client = await self._get_client()

            try:
                # Let model decide when to use tools (auto)
                # "required" causes issues with some model outputs
                tool_choice = "auto"

                payload = {
                    "model": self.GROQ_MODEL,
                    "messages": messages,
                    "tools": self.TOOLS,
                    "tool_choice": tool_choice,
                    "max_tokens": 2000,
                    "temperature": 0.3,
                }

                # Debug: log the payload
                logger.info(f"Groq request - model: {payload['model']}, messages: {len(payload['messages'])}, tools: {len(payload['tools'])}, tool_choice: {payload.get('tool_choice', 'not set')}")
                logger.info(f"Groq last message: {messages[-1]['content'][:100]}...")

                response = await client.post(
                    self.GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Track usage
                if data.get("usage"):
                    total_usage["input_tokens"] += data["usage"].get("prompt_tokens", 0)
                    total_usage["output_tokens"] += data["usage"].get("completion_tokens", 0)

                choices = data.get("choices", [])
                if not choices:
                    return {
                        "answer": "No response from Groq API",
                        "usage": total_usage,
                        "tool_results": tool_results,
                    }
                choice = choices[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason")

                # Check if Groq wants to call tools
                tool_calls = message.get("tool_calls", [])

                if tool_calls and iteration < max_tool_calls:
                    # Add assistant message with tool calls
                    messages.append(message)

                    # Execute each tool call
                    for tool_call in tool_calls:
                        func_name = tool_call["function"]["name"]
                        func_args = json.loads(tool_call["function"]["arguments"])

                        logger.info(f"Groq calling tool: {func_name}")
                        logger.info(f"Tool args: {json.dumps(func_args, indent=2)}")
                        all_tool_calls.append({"tool": func_name, "args": func_args})

                        # Execute the tool
                        if func_name in self._tool_handlers:
                            try:
                                result = await self._tool_handlers[func_name](**func_args)

                                # Include citations in tool result so Groq can reference them
                                tool_result = result.get("answer", str(result))
                                citations = result.get("citations", [])
                                logger.info(f"Tool {func_name} returned {len(citations)} citations")

                                if citations:
                                    # Log all citation URLs for debugging
                                    citation_urls = [c.get('url', '') for c in citations if c.get('url')]
                                    logger.info(f"Perplexity returned {len(citation_urls)} citation URLs:")
                                    for url in citation_urls:
                                        logger.info(f"  - {url}")

                                    # For web_search, store Perplexity's answer for direct passthrough
                                    # This prevents Groq from hallucinating numbers
                                    if func_name == "web_search":
                                        perplexity_direct_answer = result.get("answer", "")
                                        print(f"[DEBUG] Stored Perplexity answer: {perplexity_direct_answer[:200]}...", flush=True)

                                    # Append URLs to the tool result so Groq sees them
                                    links_text = "\n\nSources with URLs (INCLUDE THESE IN YOUR RESPONSE):\n" + "\n".join(
                                        f"- {c.get('url', '')}"
                                        for c in citations if c.get('url')
                                    )
                                    tool_result += links_text
                                    all_sources.extend(citations)
                                    logger.info(f"Tool result with links (last 300 chars): ...{tool_result[-300:]}")
                                elif result.get("sources"):
                                    all_sources.extend(result["sources"])

                            except Exception as e:
                                logger.error(f"Tool {func_name} failed: {e}")
                                # Give Groq a clear message about the failure
                                if "perplexity" in str(e).lower() or func_name == "web_search":
                                    tool_result = "Web search service is temporarily unavailable. Please answer based on your training knowledge, and clearly state that you couldn't verify with current web data."
                                else:
                                    tool_result = f"Tool temporarily unavailable: {func_name}. Please proceed without this tool and explain the limitation to the user."
                        else:
                            tool_result = f"Tool {func_name} not registered. Please answer based on your knowledge and explain you couldn't use this tool."

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result,
                        })

                    # Continue loop to let Groq process tool results
                    continue

                else:
                    # No tool calls or max iterations reached - return response
                    answer = message.get("content", "")

                    # Debug passthrough condition
                    print(f"[DEBUG] Passthrough check: perplexity_direct_answer={bool(perplexity_direct_answer)}, "
                          f"tool_calls_count={len(all_tool_calls)}, "
                          f"first_tool={all_tool_calls[0]['tool'] if all_tool_calls else 'none'}", flush=True)

                    # If web_search was the only tool called, use Perplexity's answer directly
                    # This prevents Groq from hallucinating numbers/prices
                    if perplexity_direct_answer and len(all_tool_calls) == 1 and all_tool_calls[0]["tool"] == "web_search":
                        print("[DEBUG] PASSTHROUGH ACTIVATED!", flush=True)
                        # Add source URLs to the Perplexity answer
                        if all_sources:
                            source_urls = "\n\nSources:\n" + "\n".join(
                                f"- {s.get('url', '')}" for s in all_sources if s.get('url')
                            )
                            answer = perplexity_direct_answer + source_urls
                        else:
                            answer = perplexity_direct_answer

                    return {
                        "answer": answer,
                        "sources": all_sources,
                        "tool_calls": all_tool_calls,
                        "usage": total_usage,
                    }

            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                logger.error(f"Groq API error: {e.response.status_code} - {error_body}")
                return {
                    "answer": f"[Groq error: {e.response.status_code}] {error_body}",
                    "sources": [],
                    "tool_calls": all_tool_calls,
                    "usage": total_usage,
                }
            except Exception as e:
                logger.error(f"Groq agent error: {e}")
                return {
                    "answer": f"[Error: {str(e)}]",
                    "sources": [],
                    "tool_calls": all_tool_calls,
                    "usage": total_usage,
                }

        # Fallback if loop exhausted
        return {
            "answer": "Max tool iterations reached. Please try a simpler query.",
            "sources": all_sources,
            "tool_calls": all_tool_calls,
            "usage": total_usage,
        }


# Module-level instance
groq_agent = GroqAgent()
