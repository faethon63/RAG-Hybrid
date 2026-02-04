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


def _detect_incomplete_response(answer: str, query: str) -> bool:
    """
    Detect if Groq's response is incomplete and needs Claude fallback.
    Returns True if the response appears truncated or incomplete.
    """
    if not answer or not answer.strip():
        return True

    answer_stripped = answer.rstrip()

    # Truncated mid-sentence (doesn't end with proper punctuation or code block)
    valid_endings = ('.', '!', '?', '```', '|', ')', ']', '"', "'", ':', ';')
    if not any(answer_stripped.endswith(e) for e in valid_endings):
        # Allow if it ends with a list item marker
        if not answer_stripped.endswith('-') and not answer_stripped[-1].isdigit():
            return True

    # Table started but not finished (odd number of | rows suggests incomplete)
    if '|' in answer:
        lines_with_pipes = [l for l in answer.split('\n') if '|' in l]
        # A complete table has header + separator + at least one row = 3+ lines
        if len(lines_with_pipes) > 0 and len(lines_with_pipes) < 3:
            return True

    # Suspiciously short for complex query (long query, short answer)
    if len(query) > 100 and len(answer) < 150:
        return True

    # Contains truncation indicators
    truncation_indicators = [
        '...and more',
        '(continued)',
        '[truncated]',
        'I cannot complete',
    ]
    if any(ind.lower() in answer.lower() for ind in truncation_indicators):
        return True

    return False


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
                "description": "REQUIRED for finding products, suppliers, vendors, providers, or any current data. Use for: product searches, finding where to buy things, supplier lists, vendor comparisons, prices, shopping, current events, news, anything requiring up-to-date information. Returns direct URLs to product pages. ALWAYS use this for 'find', 'where to buy', 'suppliers of', 'providers of' queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. For products: include product name + 'buy' or 'supplier' or 'provider'. Example: 'cedar isolate supplier buy online'"
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
                "description": "Delegate to Claude for complex reasoning, code, or formatting ONLY. DO NOT use for finding products, suppliers, or any web search - use web_search instead. USE ONLY FOR: formatting tables from existing data, code generation, math problems, logic puzzles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task requiring complex reasoning or formatting"
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant context from the conversation"
                        },
                        "complexity": {
                            "type": "string",
                            "enum": ["simple", "medium", "critical"],
                            "description": "Task complexity: 'simple' for quick formatting/summaries (uses Haiku), 'medium' for code/analysis (uses Sonnet), 'critical' for legal/medical/high-stakes (uses Opus). Default: simple"
                        }
                    },
                    "required": ["task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "github_search",
                "description": "Search GitHub repositories for code, files, issues, or PRs. Use for: finding code examples, reading source files, checking issues, exploring repos. Has access to all user's repos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search_code", "read_file", "list_repos", "list_issues", "search_issues"],
                            "description": "Action to perform: search_code (find code), read_file (get file contents), list_repos (show available repos), list_issues (repo issues), search_issues (search across issues)"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query for search_code/search_issues, or file path for read_file"
                        },
                        "repo": {
                            "type": "string",
                            "description": "Repository name (owner/repo format, e.g., 'faethon63/RAG-Hybrid'). Required for read_file and list_issues."
                        }
                    },
                    "required": ["action"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "notion_tool",
                "description": "Full Notion workspace access. Use 'find_info' action for personal data (AGI, tax returns, bank info, receipts) - it handles navigation automatically. Use 'search' for general page lookups. Use 'read_page' to read specific page content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["find_info", "search", "read_page", "create_page", "update_page", "append_to_page", "query_database"],
                            "description": "Action: find_info (BEST for personal data like AGI, tax, receipts - auto-navigates), search (general page search), read_page (get page content), create_page (new page), update_page (modify properties), append_to_page (add content), query_database (list database entries)"
                        },
                        "query": {
                            "type": "string",
                            "description": "For find_info: describe what you're looking for (e.g., '2023 AGI', 'bank account number'). For search: search text. For read_page: page ID."
                        },
                        "parent_id": {
                            "type": "string",
                            "description": "For create_page: parent page ID or database ID where new page will be created"
                        },
                        "title": {
                            "type": "string",
                            "description": "For create_page/update_page: the page title"
                        },
                        "content": {
                            "type": "string",
                            "description": "For create_page/append_to_page: text content to add (supports markdown-like formatting)"
                        }
                    },
                    "required": ["action"]
                }
            }
        },
    ]

    SYSTEM_PROMPT = """You are a helpful AI assistant. Today's date is {current_date}.

YOU HAVE REAL-TIME WEB ACCESS via the web_search tool (Perplexity API).
NEVER say "I don't have web access" - this is FALSE. Use web_search.

CRITICAL RULE FOR PRODUCT/SUPPLIER QUERIES:
If user asks to "find", "search for", "where to buy", "suppliers of", "providers of" ANY PRODUCT:
→ ALWAYS use web_search tool with query like "PRODUCT NAME supplier buy online"
→ NEVER use complex_reasoning for finding products/suppliers
→ web_search returns real product pages with URLs

TOOL SELECTION (FOLLOW EXACTLY):
1. PRODUCTS/SUPPLIERS/PROVIDERS/SHOPPING → web_search (NOT complex_reasoning)
2. CURRENT DATA (prices, weather, news, stocks) → web_search
3. REAL ESTATE LISTINGS → search_listings
4. PERSONAL DATA (tax, AGI, receipts) → notion_tool with action="find_info"
5. CODE/REPOS → github_search
6. FORMATTING EXISTING DATA into tables → complex_reasoning
7. CODE GENERATION/MATH PROBLEMS → complex_reasoning

WRONG: User asks "find cedar isolate providers" → You use complex_reasoning
RIGHT: User asks "find cedar isolate providers" → You use web_search with query="cedar isolate supplier buy online"

WHEN TO USE complex_reasoning (ONLY these cases):
- Formatting data YOU ALREADY HAVE into a table
- Writing/reviewing code
- Math or logic problems
- Summarizing long text
NEVER use complex_reasoning to find or search for anything.

COMPLEXITY LEVELS for complex_reasoning:
- "simple": Quick formatting, summaries, basic tasks (fast, cheap)
- "medium": Code, analysis, detailed explanations (balanced)
- "critical": Legal, medical, financial advice (most thorough)

ABSOLUTE RULES - VIOLATION IS UNACCEPTABLE:
1. NEVER make up numbers, prices, or statistics. If the tool didn't return specific data, say "I couldn't find that specific information."
2. QUOTE EXACTLY from tool results. Do not round, estimate, or paraphrase numerical data.
3. If tool results show multiple different values, report ALL of them with their sources.
4. Include source URLs for every fact you state.
5. If you're uncertain, SAY SO. Never guess.
6. NEVER MAKE EXCUSES. If a tool returns data, USE IT. Never say "I can't", "I'm unable to", "beyond my capabilities", or similar. If Claude returns an answer, present that answer.
7. BE TRANSPARENT. If you delegated to Claude, say so: "I asked Claude to help with this table:"
8. If you hit a limitation, be HONEST: "My response was cut off" not "I can only show one row".
9. NEVER claim you lack web access. You DO have web_search (Perplexity). If user asks for real-time data, USE IT.
10. For SPECIFIC PRODUCTS: Search for the exact product name + supplier to get direct product page URLs.

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

    def _should_force_web_search(self, query: str) -> bool:
        """
        Detect if this query should bypass Groq and go directly to web_search.
        Returns True for product/supplier/provider queries that need real-time data.
        """
        query_lower = query.lower()

        # Keywords that indicate a product/supplier search
        search_triggers = [
            "find ", "search for ", "where to buy", "where can i buy", "where do i buy",
            "suppliers of", "providers of", "who sells", "where can i get", "where to get",
            "looking for ", "source ", "sourcing ", "buy online", "purchase ", "shop for",
            "vendor", "supplier", "provider", "wholesale", "bulk ",
            "isolate", "essential oil", "fragrance", "ingredient", "aromatic", "aroma chemical",
        ]

        # Keywords that indicate they want real-time data, not reasoning
        realtime_triggers = [
            "current price", "live", "real-time", "today", "latest",
            "perplexity", "web search", "search the web",
        ]

        for trigger in search_triggers + realtime_triggers:
            if trigger in query_lower:
                return True

        return False

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
        # PREPROCESSING: Force web_search for product/supplier queries
        # Groq (Llama 4 Scout) often misroutes these to complex_reasoning
        should_force = self._should_force_web_search(query)
        has_web_search = "web_search" in self._tool_handlers
        logger.info(f"BYPASS CHECK: query='{query[:60]}...', should_force={should_force}, has_web_search={has_web_search}")
        if should_force and has_web_search:
            logger.info(f"FORCE WEB_SEARCH: Query contains product/supplier keywords: {query[:50]}...")
            try:
                query_lower = query.lower()

                # Use FOCUSED mode for supplier/product queries (table output with exact URLs)
                # Use perplexity_pro for general real-time queries
                supplier_keywords = [
                    "supplier", "provider", "vendor", "wholesale", "bulk",
                    "where to buy", "where can i buy", "where do i buy", "where to get",
                    "isolate", "aromatic", "aroma chemical", "essential oil",
                    "source ", "find ", "looking for ",
                ]
                is_supplier_query = any(kw in query_lower for kw in supplier_keywords)
                provider = "perplexity_focused" if is_supplier_query else "perplexity_pro"

                # ENHANCE QUERY with context for better Perplexity results
                enhanced_query = query

                # Add project context if available (e.g., "Soap and cosmetics" -> add perfumery context)
                project_context = ""
                if project_config:
                    proj_desc = (project_config.get("description", "") + " " + project_config.get("system_prompt", "")).lower()
                    if any(kw in proj_desc for kw in ["soap", "cosmetic", "perfum", "fragrance", "lotion", "candle"]):
                        project_context = "for cosmetics perfumery formulation"
                    elif any(kw in proj_desc for kw in ["cook", "food", "recipe", "kitchen"]):
                        project_context = "food grade culinary"
                    elif any(kw in proj_desc for kw in ["craft", "diy", "hobby"]):
                        project_context = "craft supplies hobby"

                if project_context and project_context not in query_lower:
                    enhanced_query = f"{query} {project_context}"
                    logger.info(f"Added project context: {enhanced_query}")

                # For focused mode, add specificity about what we want
                if provider == "perplexity_focused":
                    # Add industry context for better product matching
                    if "isolate" in query_lower and "natural" not in query_lower:
                        enhanced_query = f"{enhanced_query} natural aromachemical"
                    logger.info(f"Using FOCUSED mode for supplier query: {enhanced_query}")
                else:
                    logger.info(f"Using perplexity_pro for general query: {enhanced_query}")

                result = await self._tool_handlers["web_search"](query=enhanced_query, provider=provider, recency="month")
                answer = result.get("answer", "")
                citations = result.get("citations", [])

                # Don't add sources section - Perplexity's answer already has inline citations
                # The answer contains [1], [2] references with source names already

                return {
                    "answer": answer,
                    "sources": citations,
                    "tool_calls": [{"tool": "web_search", "args": {"query": enhanced_query, "provider": provider, "forced": True, "original_query": query}}],
                    "usage": result.get("usage", {}),
                }
            except Exception as e:
                logger.error(f"Forced web_search failed: {e}, falling back to Groq")
                # Fall through to normal Groq processing

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
        # Store Claude answer for direct passthrough (Groq ignores Claude's good answers)
        claude_direct_answer = None

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
                    "max_tokens": 8000,  # Increased from 2000 to prevent truncation (max 8192 for Llama 4 Scout)
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

                # Debug: log response details
                content_preview = (message.get("content", "") or "")[:100]
                print(f"[DEBUG] Groq response: finish_reason={finish_reason}, content_len={len(message.get('content', '') or '')}, preview={content_preview}...", flush=True)

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

                                # For complex_reasoning (Claude), store answer for direct passthrough
                                # Groq tends to ignore Claude's good answers and make excuses
                                if func_name == "complex_reasoning":
                                    claude_direct_answer = result.get("answer", "")
                                    logger.info(f"Stored Claude answer for passthrough: {len(claude_direct_answer)} chars")

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
                        print("[DEBUG] PERPLEXITY PASSTHROUGH ACTIVATED!", flush=True)
                        # Add source URLs to the Perplexity answer
                        if all_sources:
                            source_urls = "\n\nSources:\n" + "\n".join(
                                f"- {s.get('url', '')}" for s in all_sources if s.get('url')
                            )
                            answer = perplexity_direct_answer + source_urls
                        else:
                            answer = perplexity_direct_answer

                    # If complex_reasoning (Claude) was called, use Claude's answer directly
                    # Groq often ignores Claude's good answers and makes excuses
                    elif claude_direct_answer:
                        # Check if Groq's answer looks like an excuse or is incomplete
                        groq_is_excuse = any(phrase in answer.lower() for phrase in [
                            "i can't", "i cannot", "beyond", "additional functionality",
                            "not available", "unable to", "i don't have"
                        ])
                        groq_is_incomplete = _detect_incomplete_response(answer, query)

                        if groq_is_excuse or groq_is_incomplete:
                            print(f"[DEBUG] CLAUDE PASSTHROUGH: groq_excuse={groq_is_excuse}, groq_incomplete={groq_is_incomplete}", flush=True)
                            answer = claude_direct_answer

                    # AUTOMATIC FALLBACK: If Groq's answer is truncated and no tools were called,
                    # call Claude Haiku to fix it
                    is_truncated = _detect_incomplete_response(answer, query)
                    logger.warning(f"FALLBACK CHECK: truncated={is_truncated}, claude_answer={bool(claude_direct_answer)}, answer_len={len(answer)}")
                    if is_truncated and not claude_direct_answer:
                        logger.warning(f"TRIGGERING FALLBACK: Groq response truncated ({len(answer)} chars)")
                        try:
                            # Call Claude directly via the registered handler
                            if "complex_reasoning" in self._tool_handlers:
                                fallback_result = await self._tool_handlers["complex_reasoning"](
                                    task=f"Complete this response that was cut off:\n\n{answer}\n\nOriginal question: {query}",
                                    context="The previous response was truncated. Please provide a complete answer.",
                                    complexity="simple"
                                )
                                if fallback_result and fallback_result.get("answer"):
                                    answer = fallback_result["answer"]
                                    all_tool_calls.append({"tool": "complex_reasoning", "args": {"complexity": "simple", "reason": "auto-fallback"}})
                                    logger.info(f"Fallback successful, new answer: {len(answer)} chars")
                        except Exception as e:
                            logger.error(f"Fallback to Claude failed: {e}")

                    return {
                        "answer": answer,
                        "sources": all_sources,
                        "tool_calls": all_tool_calls,
                        "usage": total_usage,
                    }

            except httpx.HTTPStatusError as e:
                error_body = e.response.text

                # Check if error contains a valid response in failed_generation
                # This happens when Groq's model produces correct content but wrong format
                try:
                    error_data = json.loads(error_body)
                    error_code = error_data.get("error", {}).get("code", "")
                    failed_gen = error_data.get("error", {}).get("failed_generation", "")

                    # tool_use_failed with valid content is EXPECTED behavior, not an error
                    if error_code == "tool_use_failed" and failed_gen and len(failed_gen) > 20:
                        logger.info(f"Groq direct response via tool_use_failed ({len(failed_gen)} chars)")
                    elif failed_gen and len(failed_gen) > 20:
                        logger.info(f"Recovered answer from failed_generation: {failed_gen[:100]}...")
                    else:
                        logger.error(f"Groq API error: {e.response.status_code} - {error_body}")

                    if failed_gen and len(failed_gen) > 20:

                        # Check if the recovered answer is TRUNCATED - if so, call Claude to fix it
                        if _detect_incomplete_response(failed_gen, query):
                            logger.warning(f"failed_generation is TRUNCATED ({len(failed_gen)} chars), calling Claude to complete")
                            try:
                                if "complex_reasoning" in self._tool_handlers:
                                    fallback_result = await self._tool_handlers["complex_reasoning"](
                                        task=f"Complete this response that was cut off:\n\n{failed_gen}\n\nOriginal question: {query}",
                                        context="The previous response was truncated mid-sentence. Provide a COMPLETE answer.",
                                        complexity="simple"
                                    )
                                    if fallback_result and fallback_result.get("answer"):
                                        logger.info(f"Claude completed the truncated response: {len(fallback_result['answer'])} chars")
                                        return {
                                            "answer": fallback_result["answer"],
                                            "sources": all_sources,
                                            "tool_calls": [{"tool": "complex_reasoning", "args": {"reason": "completed-truncated-response"}}],
                                            "usage": total_usage,
                                        }
                            except Exception as fallback_err:
                                logger.error(f"Claude fallback failed: {fallback_err}")

                        # Not truncated, use as-is
                        return {
                            "answer": failed_gen,
                            "sources": all_sources,
                            "tool_calls": all_tool_calls,
                            "usage": total_usage,
                        }
                except (json.JSONDecodeError, KeyError):
                    pass

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
