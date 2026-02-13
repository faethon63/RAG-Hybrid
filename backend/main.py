"""
RAG-Hybrid System - Main Backend API
FastAPI server that orchestrates all RAG components
"""

import os
import sys
import re
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

# Ensure backend directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
from config import (
    reload_env, get_log_level, get_fastapi_port, get_project_kb_path,
    get_synced_kb_path, get_chromadb_collection, get_chromadb_path,
    get_claude_haiku_model, get_claude_sonnet_model, get_claude_opus_model,
    get_synced_projects_path,
)

from fastapi import FastAPI, HTTPException, status, UploadFile, File as FastAPIFile, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# KB file upload constants
ALLOWED_KB_EXTENSIONS = {'.txt', '.md', '.json', '.py', '.js', '.ts', '.html', '.css', '.yaml', '.yml', '.rst', '.csv', '.pdf', '.png', '.jpg', '.jpeg'}
MAX_KB_FILE_SIZE = 20 * 1024 * 1024  # 20 MB (needed for large instruction PDFs)

from rag_core import RAGCore
from search_integrations import ClaudeSearch, PerplexitySearch, TavilySearch, IdealistaSearch, Crawl4AISearch
from orchestrator import QueryOrchestrator
from groq_agent import groq_agent, GroqAgent, get_current_project, get_current_project_config, get_current_conversation_history
from deep_agent import get_deep_agent, is_deep_research_query
from auth import authenticate_user
from query_classifier import QueryClassifier, classify_query

logging.basicConfig(level=get_log_level())

# Initialize FastAPI
app = FastAPI(
    title="RAG-Hybrid System",
    description="Hybrid RAG system with Claude, Perplexity, and local knowledge",
    version="1.0.0"
)

# API Key authentication middleware
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication. Skipped if API_KEY env var not set (local dev)."""

    EXEMPT_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/"}

    async def dispatch(self, request: Request, call_next):
        api_key = os.getenv("API_KEY", "")

        # Skip auth if no API_KEY configured (local development)
        if not api_key:
            return await call_next(request)

        # Skip auth for exempt paths and static files
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Skip auth for form operations (downloads + sync) and tracker
        if "/forms/download/" in request.url.path or "/forms/sync" in request.url.path or "/tracker/" in request.url.path:
            return await call_next(request)

        # Check API key from header or query param
        request_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if request_key != api_key:
            return Response(
                content='{"detail": "Invalid or missing API key"}',
                status_code=401,
                media_type="application/json"
            )

        return await call_next(request)

app.add_middleware(APIKeyMiddleware)

# CORS middleware - restrict to known origins
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,https://rag.coopeverything.org").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize RAG components
rag_core = RAGCore()
claude_search = ClaudeSearch()
perplexity_search = PerplexitySearch()
tavily_search = TavilySearch()
crawl4ai_search = Crawl4AISearch()
idealista_search = IdealistaSearch()
query_orchestrator = QueryOrchestrator()

# Register tool handlers for GroqAgent
async def _tool_web_search(query: str, provider: str = "perplexity", recency: str = None) -> Dict[str, Any]:
    """
    Web search tool with provider selection.
    - perplexity: Fast, default (Sonar)
    - perplexity_pro: Thorough (Sonar Pro) - use when user asks for deep/thorough search
    - perplexity_focused: Table format with exact URLs (for supplier/product queries)
    - tavily: Better for specific URLs
    """
    logger = logging.getLogger(__name__)
    logger.info(f"web_search called with provider={provider}, query={query[:100]}...")

    if provider == "perplexity_focused":
        # Focused search: returns clean table with exact product URLs
        # Best for supplier/product queries where we need real links, not prose
        logger.info("Using Perplexity Sonar Pro FOCUSED mode (table output)")
        result = await perplexity_search.focused_search(
            query=query,
            num_results=8,
            recency=recency or "month",
        )
        result["provider_used"] = "perplexity_focused"
        return result
    elif provider == "perplexity_pro":
        logger.info("Using Perplexity Sonar Pro (high mode)")
        result = await perplexity_search.search(query=query, search_mode="high", recency=recency)
        result["provider_used"] = "perplexity_pro"
        return result
    elif provider == "tavily":
        # Check if query contains a specific URL
        import re
        url_match = re.search(r'(https?://[^\s]+)', query)
        if url_match:
            url = url_match.group(1)
            url = url.rstrip('.,;:!?"\')')

            question_part = query.replace(url, "").strip()

            # 1. Try Crawl4AI first (best quality, JS rendering, clean markdown)
            logger.info(f"Trying Crawl4AI for URL: {url}")
            try:
                crawl_result = await crawl4ai_search.extract(url)
                if crawl_result.get("success") and len(crawl_result.get("answer", "")) > 200:
                    answer_text = crawl_result["answer"]
                    logger.info(f"Crawl4AI succeeded for {url} - {len(answer_text)} chars")
                    if question_part:
                        crawl_result["answer"] = f"Page content:\n\n{answer_text}\n\nUser question: {question_part}"
                    crawl_result["provider_used"] = "crawl4ai"
                    return crawl_result
                else:
                    logger.info(f"Crawl4AI returned insufficient content, trying Tavily")
            except Exception as e:
                logger.warning(f"Crawl4AI failed for {url}: {e}")

            # 2. Try Tavily extract as fallback
            logger.info(f"Using Tavily EXTRACT to fetch specific URL: {url}")
            result = await tavily_search.extract(urls=[url])

            # Check if extraction returned meaningful content (not just nav chrome)
            raw = result.get("raw_content", {})
            content = raw.get(url, "") if isinstance(raw, dict) else ""
            answer_text = result.get("answer", "")
            content_seems_valid = len(content) > 500 or len(answer_text) > 200

            if not content_seems_valid:
                # Extraction failed - fallback to Perplexity
                logger.info(f"Tavily extract returned insufficient content, falling back to Perplexity")
                fallback = await perplexity_search.search(query=query, search_mode="high", recency="week")
                fallback["provider_used"] = "perplexity"
                return fallback

            # Extraction worked - add question context
            if question_part:
                result["answer"] = f"Based on the page content:\n\n{result['answer']}\n\nUser question: {question_part}"
            result["provider_used"] = "tavily_extract"
            return result
        else:
            logger.info("Using Tavily search (no specific URL)")
            result = await tavily_search.search(query=query, search_depth="advanced")
            result["provider_used"] = "tavily"
            return result
    else:
        # Default: perplexity (Sonar, lower cost)
        result = await perplexity_search.search(query=query, search_mode="low", recency=recency)
        result["provider_used"] = "perplexity"
        return result


async def _tool_search_listings(
    query: str,
    provider: str = "tavily",
    city: str = None,
    max_price: int = None,
    bedrooms: int = None,
    has_terrace: bool = False,
) -> Dict[str, Any]:
    """
    Search for real estate listings with specific URLs.
    - idealista: Direct API for Spain/Portugal/Italy (requires API access)
    - tavily: Web search focused on real estate domains
    """
    logger = logging.getLogger(__name__)

    # Type coercion: LLM may pass strings instead of integers
    if max_price is not None and isinstance(max_price, str):
        try:
            max_price = int(max_price.replace("$", "").replace("€", "").replace(",", "").strip())
        except ValueError:
            max_price = None
    if bedrooms is not None and isinstance(bedrooms, str):
        try:
            bedrooms = int(bedrooms.strip())
        except ValueError:
            bedrooms = None

    logger.info(f"search_listings: provider={provider}, city={city}, max_price={max_price}, query={query[:100]}...")

    if provider == "idealista":
        # Check if Idealista API is configured
        if await idealista_search.is_healthy():
            logger.info("Using Idealista API for direct listings")
            return await idealista_search.search(
                location=city or "barcelona",
                max_price=max_price,
                bedrooms=bedrooms,
                has_terrace=has_terrace,
            )
        else:
            logger.warning("Idealista API not configured, falling back to Tavily")
            provider = "tavily"

    # Default: Tavily with real estate domain focus
    logger.info("Using Tavily for real estate search")
    return await tavily_search.search_real_estate(query=query)


async def _tool_deep_research(query: str) -> Dict[str, Any]:
    """Deep research tool using Perplexity Sonar Pro."""
    return await perplexity_search.research(query=query, depth="deep")


async def _tool_complex_reasoning(task: str, context: str = "", complexity: str = "simple") -> Dict[str, Any]:
    """
    Delegate complex reasoning to Claude with tiered model selection.

    Complexity levels:
    - simple: Quick formatting, summaries, basic questions → Haiku ($1/$5 per 1M tokens)
    - medium: Code generation, analysis, detailed explanations → Sonnet ($3/$15 per 1M tokens)
    - critical: Legal, medical, high-stakes decisions → Opus ($15/$75 per 1M tokens)
    """
    logger = logging.getLogger(__name__)

    # Auto-bump complexity for specialized domains
    original_complexity = complexity
    current_config = get_current_project_config()
    if complexity == "simple" and current_config:
        project_prompt = (
            current_config.get("system_prompt", "") + " " +
            current_config.get("description", "")
        ).lower()
        domain_keywords = [
            "bankruptcy", "legal", "court", "compliance", "attorney", "law",
            "medical", "clinical", "diagnosis", "patient",
            "financial", "tax", "accounting", "audit", "fiduciary",
        ]
        if any(kw in project_prompt for kw in domain_keywords):
            complexity = "medium"
            logger.info(f"AUTO-BUMP: complexity simple→medium (specialized domain detected in project)")

    # Map complexity to Claude model
    model_map = {
        "simple": get_claude_haiku_model(),
        "medium": get_claude_sonnet_model(),
        "critical": get_claude_opus_model(),
    }
    model = model_map.get(complexity, get_claude_haiku_model())

    logger.info(f"complex_reasoning: complexity={original_complexity}→{complexity} → model={model}")

    # Include conversation history from groq_agent for context
    conversation_context = ""
    if get_current_conversation_history():
        history_lines = []
        for msg in get_current_conversation_history()[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:500]  # Truncate long messages
            history_lines.append(f"{role.upper()}: {content}")
        if history_lines:
            conversation_context = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(history_lines)
            logger.info(f"Including {len(history_lines)} messages of conversation history")

    # Inject KB context: search project knowledge base for relevant docs
    kb_context = ""
    if get_current_project():
        try:
            kb_results = await rag_core.search(query=task, project=get_current_project(), top_k=3)
            if kb_results:
                kb_chunks = []
                for r in kb_results:
                    source = r.get("metadata", {}).get("source", "unknown")
                    kb_chunks.append(f"[{source}]: {r.get('content', '')[:500]}")
                kb_context = "\n\nPROJECT KNOWLEDGE BASE (relevant documents):\n" + "\n---\n".join(kb_chunks)
                logger.info(f"Injected {len(kb_results)} KB docs into complex_reasoning context")
        except Exception as e:
            logger.warning(f"KB search for complex_reasoning context failed: {e}")

    # Add instructions to prevent unhelpful responses
    instructions = """IMPORTANT RULES:
- NEVER say "I cannot search", "search directly", or "I don't have access". You have web search data provided in the context.
- Use the information provided to answer the question directly.
- If the data is limited, give the best answer you can with what's available.
- Do not redirect the user to search elsewhere - answer their question.
- Never include citation markers like [1], [2] in your response.
- NEVER fabricate or guess URLs. Only use URLs that appear in the provided context/data. If a website could not be accessed, say so honestly. Do NOT invent URL paths.
- If tool results say "FAILED" or "could not fetch", report that honestly. Do not fill gaps with made-up information.

"""
    full_query = instructions + task
    if context:
        full_query += f"\n\nContext: {context}"
    if kb_context:
        full_query += kb_context
    if conversation_context:
        full_query += conversation_context

    result = await claude_search.search(full_query, model=model)

    # Add model info to result for transparency
    result["claude_model"] = model
    result["complexity"] = complexity

    return result


async def _tool_github_search(action: str, query: str = None, repo: str = None) -> Dict[str, Any]:
    """
    GitHub tool using gh CLI.
    Actions: search_code, read_file, list_repos, list_issues, search_issues
    """
    import subprocess
    logger = logging.getLogger(__name__)
    logger.info(f"github_search: action={action}, query={query}, repo={repo}")

    try:
        if action == "list_repos":
            # List user's repos
            result = subprocess.run(
                ["gh", "repo", "list", "--limit", "30", "--json", "name,description,url"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return {"answer": f"Your repositories:\n{result.stdout}", "sources": []}
            return {"answer": f"Error listing repos: {result.stderr}", "sources": []}

        elif action == "search_code":
            if not query:
                return {"answer": "Error: query required for search_code", "sources": []}
            result = subprocess.run(
                ["gh", "search", "code", query, "--limit", "10", "--json", "path,repository,textMatches"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return {"answer": f"Code search results for '{query}':\n{result.stdout}", "sources": []}
            return {"answer": f"Error searching code: {result.stderr}", "sources": []}

        elif action == "read_file":
            if not repo or not query:
                return {"answer": "Error: repo and query (file path) required for read_file", "sources": []}
            result = subprocess.run(
                ["gh", "api", f"repos/{repo}/contents/{query}", "-q", ".content", "--jq", "@base64d"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                # Decode base64 content
                import base64
                try:
                    content = base64.b64decode(result.stdout.strip()).decode('utf-8')
                    return {"answer": f"File {query} from {repo}:\n```\n{content[:5000]}\n```", "sources": [{"url": f"https://github.com/{repo}/blob/main/{query}"}]}
                except (ValueError, UnicodeDecodeError):
                    return {"answer": f"File content:\n{result.stdout[:5000]}", "sources": []}
            return {"answer": f"Error reading file: {result.stderr}", "sources": []}

        elif action == "list_issues":
            if not repo:
                return {"answer": "Error: repo required for list_issues", "sources": []}
            result = subprocess.run(
                ["gh", "issue", "list", "--repo", repo, "--limit", "15", "--json", "number,title,state,url"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return {"answer": f"Issues in {repo}:\n{result.stdout}", "sources": []}
            return {"answer": f"Error listing issues: {result.stderr}", "sources": []}

        elif action == "search_issues":
            if not query:
                return {"answer": "Error: query required for search_issues", "sources": []}
            result = subprocess.run(
                ["gh", "search", "issues", query, "--limit", "10", "--json", "number,title,repository,url"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return {"answer": f"Issue search results for '{query}':\n{result.stdout}", "sources": []}
            return {"answer": f"Error searching issues: {result.stderr}", "sources": []}

        else:
            return {"answer": f"Unknown action: {action}", "sources": []}

    except subprocess.TimeoutExpired:
        return {"answer": "GitHub request timed out", "sources": []}
    except Exception as e:
        logger.error(f"GitHub tool error: {e}")
        return {"answer": f"GitHub error: {str(e)}", "sources": []}


async def _tool_notion(action: str, query: str = None, parent_id: str = None, title: str = None, content: str = None) -> Dict[str, Any]:
    """
    Full Notion tool using Notion API.
    Actions: search, read_page, create_page, update_page, append_to_page, query_database
    """
    import httpx
    logger = logging.getLogger(__name__)
    logger.info(f"notion_tool: action={action}, query={query}, parent_id={parent_id}, title={title}")

    notion_token = os.getenv("NOTION_API_KEY", "")
    if not notion_token:
        return {"answer": "Notion API key not configured. Add NOTION_API_KEY to .env file.", "sources": []}

    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    def text_to_blocks(text: str) -> list:
        """Convert text content to Notion blocks."""
        blocks = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                blocks.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}
                })
            elif line.startswith("## "):
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]}
                })
            elif line.startswith("### "):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:]}}]}
                })
            elif line.startswith("- ") or line.startswith("* "):
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}
                })
            else:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]}
                })
        return blocks

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # === SMART find_info ACTION ===
            # Intelligently navigates Notion to find personal data
            if action == "find_info":
                if not query:
                    return {"answer": "Error: query required for find_info", "sources": []}

                # Category mapping - detect what kind of info and map to category
                query_lower = query.lower()
                category_map = {
                    # Tax-related
                    "agi": "tax",
                    "tax return": "tax",
                    "tax": "tax",
                    "w-2": "tax",
                    "w2": "tax",
                    "1099": "tax",
                    "income": "tax",
                    "deduction": "tax",
                    "irs": "tax",
                    "refund": "tax",
                    # Finance-related
                    "bank": "bank",
                    "account number": "bank",
                    "routing": "bank",
                    "credit card": "finance",
                    "investment": "finance",
                    # Expenses
                    "receipt": "expenses",
                    "expense": "expenses",
                    "purchase": "expenses",
                    "bill": "expenses",
                    # Personal
                    "password": "passwords",
                    "login": "passwords",
                    "credential": "passwords",
                    "address": "personal",
                    "phone": "personal",
                    "ssn": "personal",
                    "social security": "personal",
                }

                # Find matching category
                search_category = None
                for keyword, category in category_map.items():
                    if keyword in query_lower:
                        search_category = category
                        break

                # If no category matched, use the query as-is
                if not search_category:
                    search_category = query.split()[0] if query.split() else query

                # Extract year from query for smarter searching
                year_match = re.search(r'\b(20\d{2})\b', query)
                target_year = year_match.group(1) if year_match else None

                logger.info(f"find_info: query='{query}' → category='{search_category}', year={target_year}")

                # Step 1: Search for the category (and optionally year-specific)
                results = []

                # First search: year + category if year is specified
                if target_year:
                    year_search_resp = await client.post(
                        "https://api.notion.com/v1/search",
                        headers=headers,
                        json={"query": f"{target_year} {search_category}", "page_size": 5}
                    )
                    if year_search_resp.status_code == 200:
                        results.extend(year_search_resp.json().get("results", []))

                # Second search: just the category
                search_resp = await client.post(
                    "https://api.notion.com/v1/search",
                    headers=headers,
                    json={"query": search_category, "page_size": 5}
                )

                if search_resp.status_code != 200:
                    return {"answer": f"Notion search error: {search_resp.status_code}", "sources": []}

                # Combine results, deduplicate by ID
                seen_ids = {r["id"] for r in results}
                for r in search_resp.json().get("results", []):
                    if r["id"] not in seen_ids:
                        results.append(r)
                        seen_ids.add(r["id"])

                # Log all found pages for debugging
                logger.info(f"find_info: found {len(results)} pages:")
                for r in results[:5]:
                    title = ""
                    if r.get("properties", {}).get("title", {}).get("title"):
                        title = "".join([t.get("plain_text", "") for t in r["properties"]["title"]["title"]])
                    elif r.get("properties", {}).get("Name", {}).get("title"):
                        title = "".join([t.get("plain_text", "") for t in r["properties"]["Name"]["title"]])
                    logger.info(f"  - {title or '(untitled)'} (ID: {r.get('id', 'N/A')[:8]}...)")

                if not results:
                    # Try a broader search with the first word
                    first_word = query.split()[0] if query.split() else query
                    if first_word != search_category:
                        search_resp = await client.post(
                            "https://api.notion.com/v1/search",
                            headers=headers,
                            json={"query": first_word, "page_size": 5}
                        )
                        if search_resp.status_code == 200:
                            results = search_resp.json().get("results", [])

                if not results:
                    return {"answer": f"No Notion pages found for category '{search_category}'. Try a different search term.", "sources": []}

                # Step 2: Pick the best page - smart year matching
                # Helper to get page title
                def get_page_title(item):
                    if item.get("properties", {}).get("title"):
                        title_prop = item["properties"]["title"]
                        if title_prop.get("title"):
                            return "".join([t.get("plain_text", "") for t in title_prop["title"]])
                    if item.get("properties", {}).get("Name"):
                        name_prop = item["properties"]["Name"]
                        if name_prop.get("title"):
                            return "".join([t.get("plain_text", "") for t in name_prop["title"]])
                    return ""

                # Score pages for relevance
                best_page = None
                best_score = -1

                for item in results:
                    if item.get("object") != "page":
                        continue
                    title = get_page_title(item).lower()
                    score = 0

                    # Prefer pages matching the year
                    if target_year and target_year in title:
                        score += 10

                    # Prefer "previous years" or generic tax pages for historical data
                    if target_year and target_year != "2024":
                        if "previous" in title or "history" in title or "archive" in title:
                            score += 5

                    # Prefer pages with the category in the name
                    if search_category.lower() in title:
                        score += 3

                    # Prefer pages without a specific different year
                    if target_year:
                        other_years = re.findall(r'\b(20\d{2})\b', title)
                        if other_years and target_year not in other_years:
                            score -= 5  # Penalize pages with different years

                    if score > best_score:
                        best_score = score
                        best_page = item

                # Fallback to first page result
                if not best_page:
                    for item in results:
                        if item.get("object") == "page":
                            best_page = item
                            break
                if not best_page:
                    best_page = results[0]

                page_id = best_page.get("id", "")
                page_title = get_page_title(best_page)

                logger.info(f"find_info: selected page '{page_title}' (ID: {page_id})")

                # Step 3: Read the page content (with recursive dropdown reading)
                async def read_blocks_recursive_inner(block_id: str, indent: int = 0) -> list:
                    """Recursively read blocks including toggle/dropdown content."""
                    resp = await client.get(
                        f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100",
                        headers=headers
                    )
                    if resp.status_code != 200:
                        return []

                    lines = []
                    prefix_indent = "  " * indent
                    for block in resp.json().get("results", []):
                        block_type = block.get("type", "")
                        block_id_child = block.get("id", "")
                        has_children = block.get("has_children", False)

                        text = ""
                        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "toggle", "to_do", "callout", "quote"]:
                            text_content = block.get(block_type, {}).get("rich_text", [])
                            text = "".join([t.get("plain_text", "") for t in text_content])
                        elif block_type == "child_page":
                            text = f"[Subpage: {block.get('child_page', {}).get('title', 'Untitled')}]"
                        elif block_type == "child_database":
                            text = f"[Database: {block.get('child_database', {}).get('title', 'Untitled')}]"

                        if text:
                            if "heading_1" in block_type:
                                lines.append(f"{prefix_indent}# {text}")
                            elif "heading_2" in block_type:
                                lines.append(f"{prefix_indent}## {text}")
                            elif "heading_3" in block_type:
                                lines.append(f"{prefix_indent}### {text}")
                            elif "toggle" in block_type:
                                lines.append(f"{prefix_indent}[DROPDOWN] {text}")
                            elif "list" in block_type or "to_do" in block_type:
                                lines.append(f"{prefix_indent}- {text}")
                            else:
                                lines.append(f"{prefix_indent}{text}")

                        if has_children:
                            child_lines = await read_blocks_recursive_inner(block_id_child, indent + 1)
                            lines.extend(child_lines)

                    return lines

                content_lines = await read_blocks_recursive_inner(page_id)
                page_url = f"https://notion.so/{page_id.replace('-', '')}"

                if content_lines:
                    content_text = "\n".join(content_lines)
                    return {
                        "answer": f"Found in page '{page_title}':\n\n{content_text}",
                        "sources": [{"url": page_url}]
                    }
                return {
                    "answer": f"Page '{page_title}' found but has no readable content.",
                    "sources": [{"url": page_url}]
                }

            # === REGULAR SEARCH ACTION ===
            if action == "search":
                if not query:
                    return {"answer": "Error: query required for search", "sources": []}
                response = await client.post(
                    "https://api.notion.com/v1/search",
                    headers=headers,
                    json={"query": query, "page_size": 10}
                )
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    for item in data.get("results", []):
                        title_text = ""
                        if item.get("properties", {}).get("title"):
                            title_prop = item["properties"]["title"]
                            if title_prop.get("title"):
                                title_text = "".join([t.get("plain_text", "") for t in title_prop["title"]])
                        elif item.get("properties", {}).get("Name"):
                            name_prop = item["properties"]["Name"]
                            if name_prop.get("title"):
                                title_text = "".join([t.get("plain_text", "") for t in name_prop["title"]])
                        results.append({
                            "id": item["id"],
                            "type": item["object"],
                            "title": title_text or "(untitled)",
                            "url": item.get("url", "")
                        })
                    return {"answer": f"Notion search results for '{query}':\n{json.dumps(results, indent=2)}", "sources": [{"url": r["url"]} for r in results if r.get("url")]}
                return {"answer": f"Notion search error: {response.status_code} - {response.text}", "sources": []}

            elif action == "read_page":
                if not query:
                    return {"answer": "Error: page ID required for read_page", "sources": []}

                async def read_blocks_recursive(block_id: str, indent: int = 0) -> list:
                    """Recursively read blocks including toggle/dropdown content."""
                    resp = await client.get(
                        f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100",
                        headers=headers
                    )
                    if resp.status_code != 200:
                        return []

                    lines = []
                    prefix_indent = "  " * indent
                    for block in resp.json().get("results", []):
                        block_type = block.get("type", "")
                        block_id_child = block.get("id", "")
                        has_children = block.get("has_children", False)

                        # Extract text from common block types
                        text = ""
                        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "toggle", "to_do", "callout", "quote"]:
                            text_content = block.get(block_type, {}).get("rich_text", [])
                            text = "".join([t.get("plain_text", "") for t in text_content])
                        elif block_type == "child_page":
                            text = f"[Subpage: {block.get('child_page', {}).get('title', 'Untitled')}]"
                        elif block_type == "child_database":
                            text = f"[Database: {block.get('child_database', {}).get('title', 'Untitled')}]"

                        if text:
                            if "heading_1" in block_type:
                                lines.append(f"{prefix_indent}# {text}")
                            elif "heading_2" in block_type:
                                lines.append(f"{prefix_indent}## {text}")
                            elif "heading_3" in block_type:
                                lines.append(f"{prefix_indent}### {text}")
                            elif "toggle" in block_type:
                                lines.append(f"{prefix_indent}▼ {text}")  # Dropdown/toggle indicator
                            elif "list" in block_type or "to_do" in block_type:
                                lines.append(f"{prefix_indent}- {text}")
                            else:
                                lines.append(f"{prefix_indent}{text}")

                        # Recursively fetch children (for toggles/dropdowns)
                        if has_children:
                            child_lines = await read_blocks_recursive(block_id_child, indent + 1)
                            lines.extend(child_lines)

                    return lines

                content_lines = await read_blocks_recursive(query)
                if content_lines:
                    return {"answer": f"Page content:\n\n" + "\n".join(content_lines), "sources": [{"url": f"https://notion.so/{query.replace('-', '')}"}]}
                return {"answer": "Page is empty or has no readable content.", "sources": [{"url": f"https://notion.so/{query.replace('-', '')}"}]}

            elif action == "create_page":
                if not parent_id:
                    return {"answer": "Error: parent_id required for create_page (page or database ID)", "sources": []}
                if not title:
                    return {"answer": "Error: title required for create_page", "sources": []}

                # Determine if parent is a page or database
                # Try as page first
                page_data = {
                    "parent": {"page_id": parent_id},
                    "properties": {
                        "title": {"title": [{"text": {"content": title}}]}
                    }
                }
                if content:
                    page_data["children"] = text_to_blocks(content)

                response = await client.post(
                    "https://api.notion.com/v1/pages",
                    headers=headers,
                    json=page_data
                )

                # If failed, try as database parent
                if response.status_code == 400 and "database" in response.text.lower():
                    page_data["parent"] = {"database_id": parent_id}
                    page_data["properties"] = {
                        "Name": {"title": [{"text": {"content": title}}]}
                    }
                    response = await client.post(
                        "https://api.notion.com/v1/pages",
                        headers=headers,
                        json=page_data
                    )

                if response.status_code == 200:
                    data = response.json()
                    return {"answer": f"Page created successfully!\nTitle: {title}\nID: {data['id']}\nURL: {data.get('url', 'N/A')}", "sources": [{"url": data.get("url", "")}]}
                return {"answer": f"Error creating page: {response.status_code} - {response.text}", "sources": []}

            elif action == "append_to_page":
                if not query:
                    return {"answer": "Error: page ID required for append_to_page", "sources": []}
                if not content:
                    return {"answer": "Error: content required for append_to_page", "sources": []}

                blocks = text_to_blocks(content)
                response = await client.patch(
                    f"https://api.notion.com/v1/blocks/{query}/children",
                    headers=headers,
                    json={"children": blocks}
                )
                if response.status_code == 200:
                    return {"answer": f"Content appended to page successfully!", "sources": [{"url": f"https://notion.so/{query.replace('-', '')}"}]}
                return {"answer": f"Error appending to page: {response.status_code} - {response.text}", "sources": []}

            elif action == "update_page":
                if not query:
                    return {"answer": "Error: page ID required for update_page", "sources": []}

                update_data = {"properties": {}}
                if title:
                    update_data["properties"]["title"] = {"title": [{"text": {"content": title}}]}

                if not update_data["properties"]:
                    return {"answer": "Error: title required for update_page", "sources": []}

                response = await client.patch(
                    f"https://api.notion.com/v1/pages/{query}",
                    headers=headers,
                    json=update_data
                )
                if response.status_code == 200:
                    return {"answer": f"Page updated successfully!", "sources": [{"url": f"https://notion.so/{query.replace('-', '')}"}]}
                return {"answer": f"Error updating page: {response.status_code} - {response.text}", "sources": []}

            elif action == "query_database":
                if not query:
                    return {"answer": "Error: database ID required for query_database", "sources": []}
                response = await client.post(
                    f"https://api.notion.com/v1/databases/{query}/query",
                    headers=headers,
                    json={"page_size": 20}
                )
                if response.status_code == 200:
                    data = response.json()
                    entries = []
                    for item in data.get("results", []):
                        props = {"_id": item["id"]}
                        for key, val in item.get("properties", {}).items():
                            if val.get("title"):
                                props[key] = "".join([t.get("plain_text", "") for t in val["title"]])
                            elif val.get("rich_text"):
                                props[key] = "".join([t.get("plain_text", "") for t in val["rich_text"]])
                            elif val.get("number") is not None:
                                props[key] = val["number"]
                            elif val.get("select"):
                                props[key] = val["select"].get("name", "")
                        entries.append(props)
                    return {"answer": f"Database entries:\n{json.dumps(entries, indent=2)}", "sources": []}
                return {"answer": f"Error querying database: {response.status_code} - {response.text}", "sources": []}

            else:
                return {"answer": f"Unknown action: {action}. Available: search, read_page, create_page, update_page, append_to_page, query_database", "sources": []}

    except Exception as e:
        logger.error(f"Notion tool error: {e}")
        return {"answer": f"Notion error: {str(e)}", "sources": []}


async def _tool_find_suppliers(
    product: str,
    max_suppliers: int = 5,
) -> Dict[str, Any]:
    """
    Advanced supplier research using Playwright browser automation.
    Navigates to supplier sites, uses their search, extracts prices.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"find_suppliers called for product={product}, max_suppliers={max_suppliers}")

    try:
        from procurement_agent import get_procurement_agent
        from supplier_db import get_supplier_db

        # Get/create procurement agent with dependencies
        supplier_db = get_supplier_db()
        agent = get_procurement_agent(
            perplexity_search=perplexity_search,
            supplier_db=supplier_db,
        )

        # Run the research
        result = await agent.research_product(
            query=product,
            max_suppliers=min(max_suppliers, 10),  # Cap at 10
        )

        return {
            "answer": result.get("answer", "No results found"),
            "sources": result.get("sources", []),
            "products": result.get("products", []),
        }

    except ImportError as e:
        logger.error(f"Procurement agent import error: {e}")
        # Fallback to web_search if procurement agent not available
        logger.info("Falling back to web_search for supplier research")
        return await _tool_web_search(
            query=f"{product} supplier buy online",
            provider="perplexity_pro",
        )
    except Exception as e:
        logger.error(f"find_suppliers error: {e}")
        return {
            "answer": f"Supplier research failed: {str(e)}. Try using web_search instead.",
            "sources": [],
        }


async def _tool_search_knowledge_base(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search project's indexed knowledge base (ChromaDB)."""
    logger = logging.getLogger(__name__)
    project_name = get_current_project()
    if not project_name:
        return {"answer": "No project selected. Select a project to search its knowledge base.", "sources": []}

    top_k = min(top_k or 5, 10)
    logger.info(f"KB search: project={project_name}, query={query[:80]}, top_k={top_k}")

    try:
        results = await rag_core.search(query=query, project=project_name, top_k=top_k)
        if not results:
            return {"answer": f"No documents found in {project_name} knowledge base for: {query}", "sources": []}

        # Format results for Groq
        chunks = []
        sources = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            metadata = r.get("metadata", {})
            score = r.get("score", 0)
            source_file = metadata.get("source", metadata.get("filename", "unknown"))
            chunks.append(f"[Doc {i}] (source: {source_file}, relevance: {score:.2f})\n{content}")
            sources.append({"title": source_file, "url": "", "snippet": content[:200]})

        answer = f"Found {len(results)} relevant documents in {project_name} KB:\n\n" + "\n\n---\n\n".join(chunks)
        logger.info(f"KB search returned {len(results)} results")
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"KB search error: {e}")
        return {"answer": f"Knowledge base search failed: {e}", "sources": []}


async def _tool_list_directory(path: str) -> Dict[str, Any]:
    """List directory contents within allowed project paths."""
    logger = logging.getLogger(__name__)
    config = get_current_project_config() or {}
    allowed_paths = config.get("allowed_paths", [])
    if not allowed_paths:
        return {"answer": "No file access configured for this project. Add allowed paths in project settings.", "sources": []}

    from file_tools import FileTools
    result = FileTools.list_dir(path, allowed_paths)
    if result.get("success"):
        entries = result["entries"]
        lines = [f"Directory: {result['path']} ({result['count']} items)"]
        for e in entries:
            type_icon = "DIR " if e["type"] == "directory" else "FILE"
            size_str = f" ({e['size']} bytes)" if e.get("size") else ""
            lines.append(f"  {type_icon} {e['name']}{size_str}")
        return {"answer": "\n".join(lines), "sources": []}
    else:
        return {"answer": f"Error: {result.get('error', 'Unknown error')}", "sources": []}


async def _tool_read_file(path: str) -> Dict[str, Any]:
    """Read file contents from allowed project paths."""
    logger = logging.getLogger(__name__)
    config = get_current_project_config() or {}
    allowed_paths = config.get("allowed_paths", [])
    if not allowed_paths:
        return {"answer": "No file access configured for this project.", "sources": []}

    from file_tools import FileTools
    result = FileTools.read_file(path, allowed_paths)
    if result.get("success"):
        content = result["content"]
        # Truncate very long files for Groq context
        if len(content) > 8000:
            content = content[:8000] + f"\n\n[... truncated, {result['size']} chars total]"
        return {"answer": f"File: {result['path']}\n\n{content}", "sources": []}
    else:
        return {"answer": f"Error: {result.get('error', 'Unknown error')}", "sources": []}


async def _tool_search_files(path: str, pattern: str) -> Dict[str, Any]:
    """Search for files matching a pattern in allowed project paths."""
    logger = logging.getLogger(__name__)
    config = get_current_project_config() or {}
    allowed_paths = config.get("allowed_paths", [])
    if not allowed_paths:
        return {"answer": "No file access configured for this project.", "sources": []}

    from file_tools import FileTools
    result = FileTools.search_files(path, pattern, allowed_paths)
    if result.get("success"):
        matches = result["matches"]
        if not matches:
            return {"answer": f"No files matching '{pattern}' in {path}", "sources": []}
        lines = [f"Found {result['count']} files matching '{pattern}' in {result['base_path']}:"]
        for m in matches:
            lines.append(f"  {m['type'].upper()} {m['path']}")
        if result.get("truncated"):
            lines.append("  [... more results truncated]")
        return {"answer": "\n".join(lines), "sources": []}
    else:
        return {"answer": f"Error: {result.get('error', 'Unknown error')}", "sources": []}


async def _tool_browse_website(url: str, action: str = "read", search_term: str = None, link_filter: str = None) -> Dict[str, Any]:
    """Browse a website. Tries Playwright first, falls back to Crawl4AI."""
    logger = logging.getLogger(__name__)
    logger.info(f"browse_website: url={url}, action={action}, search_term={search_term}")

    playwright_failed = False
    try:
        from procurement_agent import get_procurement_agent
        from supplier_db import get_supplier_db

        supplier_db = get_supplier_db()
        agent = get_procurement_agent(
            perplexity_search=perplexity_search,
            supplier_db=supplier_db,
        )

        result = await agent.browse_page(
            url=url,
            action=action,
            search_term=search_term,
            link_filter=link_filter,
        )
        # Check if Playwright actually returned content
        answer = result.get("answer", "")
        if answer and "failed" not in answer.lower() and "not available" not in answer.lower() and len(answer) > 100:
            return result
        logger.info(f"Playwright returned insufficient content, falling back to Crawl4AI")
        playwright_failed = True
    except Exception as e:
        logger.warning(f"Playwright browse failed: {e}, falling back to Crawl4AI")
        playwright_failed = True

    # Fallback: use Crawl4AI to fetch the page
    if playwright_failed:
        try:
            logger.info(f"browse_website fallback: using Crawl4AI for {url}")
            crawl_result = await crawl4ai_search.extract(url)
            if crawl_result.get("success") and len(crawl_result.get("answer", "")) > 200:
                answer = crawl_result["answer"]
                if search_term:
                    answer = f"Page content from {url} (searched for: {search_term}):\n\n{answer}"
                crawl_result["provider_used"] = "crawl4ai"
                return crawl_result
            else:
                logger.warning(f"Crawl4AI also returned insufficient content for {url}")
        except Exception as e2:
            logger.error(f"Crawl4AI fallback also failed for {url}: {e2}")

    return {
        "answer": f"FAILED: Could not fetch content from {url}. The page could not be loaded.",
        "sources": [],
    }


groq_agent.register_tool_handler("web_search", _tool_web_search)
groq_agent.register_tool_handler("search_listings", _tool_search_listings)
groq_agent.register_tool_handler("deep_research", _tool_deep_research)
groq_agent.register_tool_handler("complex_reasoning", _tool_complex_reasoning)
groq_agent.register_tool_handler("github_search", _tool_github_search)
groq_agent.register_tool_handler("notion_tool", _tool_notion)
groq_agent.register_tool_handler("find_suppliers", _tool_find_suppliers)
groq_agent.register_tool_handler("browse_website", _tool_browse_website)
groq_agent.register_tool_handler("search_knowledge_base", _tool_search_knowledge_base)
groq_agent.register_tool_handler("list_directory", _tool_list_directory)
groq_agent.register_tool_handler("read_file", _tool_read_file)
groq_agent.register_tool_handler("search_files", _tool_search_files)


# --- Bankruptcy Form-Filling Tool Handlers ---

async def _tool_fill_bankruptcy_form(
    action: str,
    form_id: str,
    input_pdf: str = "",
    output_pdf: str = "",
) -> Dict[str, Any]:
    """Handle fill_bankruptcy_form tool calls from Groq."""
    logger = logging.getLogger(__name__)
    logger.info(f"fill_bankruptcy_form: action={action}, form_id={form_id}")

    try:
        from form_filler_engine import handle_fill_form_tool
        project_name = get_current_project() or "Chapter_7_Assistant"
        result = await handle_fill_form_tool(
            action=action,
            form_id=form_id,
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            project_name=project_name,
        )

        # Format result for Groq
        if result.get("markdown"):
            return {"answer": result["markdown"], "sources": []}
        elif result.get("audit"):
            audit = result["audit"]
            summary = audit.get("summary", "")
            approved = audit.get("approved", False)
            concerns = audit.get("concerns", [])
            concern_text = "\n".join(f"- [{c.get('severity', 'info')}] {c.get('field', '')}: {c.get('issue', '')}" for c in concerns)
            return {
                "answer": f"**Opus Audit {'APPROVED' if approved else 'REJECTED'}**\n\n{summary}\n\n{concern_text}" if concerns else f"**Opus Audit {'APPROVED' if approved else 'REJECTED'}**\n\n{summary}",
                "sources": [],
            }
        elif result.get("success"):
            msg = result.get("message", json.dumps(result))
            # Add download link for filled forms
            output_pdf = result.get("output_pdf", "")
            if output_pdf:
                filename = Path(output_pdf).name
                verification = result.get("verification", {})
                verified = verification.get("verified", False)
                v_detail = verification.get("message", "")
                download_url = f"/api/v1/projects/{project_name}/forms/download/{filename}"
                msg += f"\n\n**Verification:** {'PASSED' if verified else 'CHECK NEEDED'} - {v_detail}"
                msg += f"\n\n**[Download filled form]({download_url})**"
                windows_copy = result.get("windows_copy")
                if windows_copy:
                    msg += f"\n\nSaved to: `{windows_copy}`"
            return {"answer": msg, "sources": []}
        else:
            return {"answer": f"Error: {result.get('error', 'Unknown error')}", "sources": []}

    except Exception as e:
        logger.error(f"fill_bankruptcy_form error: {e}")
        return {"answer": f"Form fill error: {str(e)}", "sources": []}


async def _tool_build_data_profile(
    document_paths: List[str],
    use_dual_verification: bool = True,
) -> Dict[str, Any]:
    """Handle build_data_profile tool calls."""
    logger = logging.getLogger(__name__)
    logger.info(f"build_data_profile: {len(document_paths)} documents, dual={use_dual_verification}")

    try:
        from data_extractor import build_data_profile
        project_name = get_current_project() or "Chapter_7_Assistant"
        result = await build_data_profile(
            project_name=project_name,
            document_paths=document_paths,
            use_dual_verification=use_dual_verification,
        )

        if result.get("success"):
            summary = result.get("profile_summary", {})
            discrepancies = result.get("discrepancies", [])
            disc_text = ""
            if discrepancies:
                disc_text = "\n\n**Discrepancies found (need review):**\n"
                for d in discrepancies[:10]:
                    disc_text += f"- {d.get('field', '')}: Sonnet={d.get('sonnet')}, Opus={d.get('opus')} ({d.get('reason', '')})\n"

            return {
                "answer": f"**Data Profile Built**\n\nDocuments processed: {', '.join(result.get('documents_processed', []))}\nTotal fields extracted: {result.get('total_fields_extracted', 0)}\nProfile sections: {json.dumps(summary.get('sections', {}))}{disc_text}",
                "sources": [],
            }
        return {"answer": f"Error: {result.get('error', 'Unknown error')}", "sources": []}

    except Exception as e:
        logger.error(f"build_data_profile error: {e}")
        return {"answer": f"Profile build error: {str(e)}", "sources": []}


async def _tool_check_data_consistency() -> Dict[str, Any]:
    """Handle check_data_consistency tool calls."""
    logger = logging.getLogger(__name__)
    logger.info("check_data_consistency called")

    try:
        from data_profile import DataProfile
        from audit_rules import run_all_checks

        project_name = get_current_project() or "Chapter_7_Assistant"
        profile = DataProfile(project_name)
        if not profile.load():
            return {"answer": "No data profile found. Build one first with build_data_profile.", "sources": []}

        result = run_all_checks(profile)

        # Format issues
        issues_text = ""
        for issue in result.get("issues", []):
            severity = issue.get("severity", "info").upper()
            issues_text += f"\n- [{severity}] {issue.get('rule', '')}: {issue.get('message', '')}"

        return {
            "answer": f"**Consistency Check: {result.get('summary', '')}**{issues_text}" if issues_text else f"**Consistency Check: {result.get('summary', '')}**\n\nNo issues found.",
            "sources": [],
        }

    except Exception as e:
        logger.error(f"check_data_consistency error: {e}")
        return {"answer": f"Consistency check error: {str(e)}", "sources": []}


async def _tool_get_data_profile(section: str = "all") -> Dict[str, Any]:
    """Return the extracted data profile (truth file) with structured financial data."""
    logger = logging.getLogger(__name__)
    project_name = get_current_project() or "Chapter_7_Assistant"
    profile_path = Path(get_project_kb_path()) / project_name / "data_profile.json"

    if not profile_path.exists():
        return {"answer": "No data profile found. Build one first with build_data_profile.", "sources": []}

    try:
        with open(profile_path) as f:
            data = json.load(f)

        if section and section != "all" and section in data:
            result = {section: data[section]}
        else:
            result = data

        answer = json.dumps(result, indent=2)
        if len(answer) > 8000:
            answer = answer[:8000] + "\n... (truncated)"
        return {"answer": f"**Data Profile:**\n```json\n{answer}\n```", "sources": []}
    except Exception as e:
        logger.error(f"get_data_profile error: {e}")
        return {"answer": f"Error reading data profile: {e}", "sources": []}


async def _tool_download_file(url: str, filename: str = "") -> Dict[str, Any]:
    """Handle download_file tool calls - general file download to project KB."""
    logger = logging.getLogger(__name__)
    logger.info(f"download_file: url={url}, filename={filename}")
    try:
        import httpx
        from pdf_tools import BANKRUPTCY_FORM_URLS

        project_name = get_current_project() or "Chapter_7_Assistant"
        docs_dir = Path(get_synced_kb_path()) / project_name / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Check if url is a bankruptcy form ID shorthand
        url_upper = url.upper().replace(" ", "").replace("_", "")
        if url_upper in BANKRUPTCY_FORM_URLS or any(
            k.replace("-", "") == url_upper.replace("-", "") for k in BANKRUPTCY_FORM_URLS
        ):
            # It's a form ID, resolve to URL
            actual_url = BANKRUPTCY_FORM_URLS.get(url_upper)
            if not actual_url:
                for k, v in BANKRUPTCY_FORM_URLS.items():
                    if k.replace("-", "") == url_upper.replace("-", ""):
                        actual_url = v
                        break
            if actual_url:
                if not filename:
                    filename = actual_url.split("/")[-1]
                url = actual_url

        # Validate it looks like a URL
        if not url.startswith("http://") and not url.startswith("https://"):
            return {"answer": f"Error: '{url}' is not a valid URL or known form ID.", "sources": []}

        # Derive filename from URL if not provided
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename:
                filename = "downloaded_file"

        output_path = docs_dir / filename

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)

        size_kb = len(response.content) / 1024
        return {
            "answer": f"Downloaded **{filename}** ({size_kb:.0f} KB) to project documents folder.",
            "sources": [],
        }
    except httpx.HTTPError as e:
        return {"answer": f"Download failed: {e}", "sources": []}
    except Exception as e:
        logger.error(f"download_file error: {e}")
        return {"answer": f"Download error: {str(e)}", "sources": []}

groq_agent.register_tool_handler("fill_bankruptcy_form", _tool_fill_bankruptcy_form)
groq_agent.register_tool_handler("download_file", _tool_download_file)
groq_agent.register_tool_handler("build_data_profile", _tool_build_data_profile)
groq_agent.register_tool_handler("check_data_consistency", _tool_check_data_consistency)
groq_agent.register_tool_handler("get_data_profile", _tool_get_data_profile)


async def _tool_update_filing_data(updates: dict, source: str = "user_provided") -> Dict[str, Any]:
    """Update one or more fields in the data profile from user-provided data."""
    logger = logging.getLogger(__name__)
    project_name = get_current_project() or "Chapter_7_Assistant"

    try:
        from data_profile import DataProfile, DataField

        profile = DataProfile(project_name)
        profile.load()

        updated_fields = []
        for path, value in updates.items():
            parts = path.split(".", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid field path (no section.field): {path}")
                continue

            section, field_name = parts
            # Auto-add section if valid
            if section not in profile.SECTIONS:
                logger.warning(f"Unknown section '{section}', skipping field {path}")
                continue

            field = DataField(
                value=value,
                source=source,
                confidence=0.9 if source != "user_provided" else 1.0,
                verified_by="user" if source == "user_provided" else None,
            )
            profile.set_field(section, field_name, field)
            updated_fields.append(f"{path} = {value}")

        if updated_fields:
            profile.save()
            field_list = "\n".join(f"- {f}" for f in updated_fields)
            return {
                "answer": f"Updated {len(updated_fields)} field(s) in data profile:\n{field_list}\n\nData saved and synced.",
                "sources": [],
            }
        else:
            return {"answer": "No valid fields to update. Use format: section.field_name (e.g., expenses.rent)", "sources": []}

    except Exception as e:
        logger.error(f"update_filing_data error: {e}")
        return {"answer": f"Error updating data: {e}", "sources": []}


async def _tool_get_filing_status() -> Dict[str, Any]:
    """Return the filing completion status as formatted markdown."""
    logger = logging.getLogger(__name__)
    project_name = get_current_project() or "Chapter_7_Assistant"

    try:
        from filing_tracker_export import format_missing_summary
        summary = format_missing_summary(project_name)
        return {"answer": summary, "sources": []}
    except Exception as e:
        logger.error(f"get_filing_status error: {e}")
        return {"answer": f"Error getting filing status: {e}", "sources": []}


groq_agent.register_tool_handler("update_filing_data", _tool_update_filing_data)
groq_agent.register_tool_handler("get_filing_status", _tool_get_filing_status)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AttachedFile(BaseModel):
    name: str
    type: str
    data: str  # base64 for images/PDFs, text content for text files
    isImage: bool = False


class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"  # "auto", "private", "research", "deep_agent"
    model: str = "auto"  # "auto" (orchestrator decides), "local", "perplexity", or Claude model IDs
    project: Optional[str] = None
    max_results: int = 5
    include_sources: bool = True
    conversation_history: List[Dict[str, str]] = []  # List of {"role": "user"|"assistant", "content": "..."}
    files: Optional[List[AttachedFile]] = None  # Attached files (images, PDFs, text)
    

class Source(BaseModel):
    type: str  # "local_doc", "web", "notion", "drive"
    title: str
    url: Optional[str] = None
    snippet: str
    score: Optional[float] = None


class RoutingInfo(BaseModel):
    """Transparency info showing how the query was routed."""
    orchestrator: str  # e.g., "groq", "direct"
    tools_used: List[str] = []  # e.g., ["web_search", "complex_reasoning"]
    claude_model: Optional[str] = None  # e.g., "haiku", "sonnet", "opus" if Claude was used
    reasoning: Optional[str] = None  # Brief explanation of routing decision


class QueryResponse(BaseModel):
    query: str
    answer: str
    mode: str
    sources: List[Source]
    processing_time: float
    timestamp: str
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None  # {"input": X, "output": Y}
    estimated_cost: Optional[float] = None
    agent_steps: Optional[List[Dict[str, Any]]] = None  # For deep_agent mode
    routing_info: Optional[RoutingInfo] = None  # Shows orchestrator, tools used, Claude model if any


class IndexRequest(BaseModel):
    documents: List[Dict[str, Any]]
    project: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    timestamp: str

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RAG-Hybrid System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "login": "/api/v1/login",
            "query": "/api/v1/query",
            "index": "/api/v1/index",
            "health": "/api/v1/health",
            "projects": "/api/v1/projects"
        }
    }


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/login")
async def login(request: LoginRequest):
    """Authenticate and receive a JWT token."""
    token = authenticate_user(request.username, request.password)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    return {"token": token, "username": request.username}


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with weighted status logic."""
    services = {
        "local_rag": await rag_core.is_healthy(),
        "claude_api": await claude_search.is_healthy(),
        "perplexity_api": await perplexity_search.is_healthy(),
        "ollama": await rag_core.check_ollama(),
        "chromadb": await rag_core.check_chromadb()
    }

    # Core services: RAG core and ChromaDB are required
    core_healthy = services["local_rag"] and services["chromadb"]

    # Optional services: Ollama, Claude, Perplexity
    optional_count = sum([
        services["ollama"],
        services["claude_api"],
        services["perplexity_api"]
    ])

    if not core_healthy:
        status = "offline"
    elif optional_count == 0:
        status = "degraded"  # Core works but no LLM available
    elif all(services.values()):
        status = "healthy"
    else:
        status = "healthy"  # Core works + at least one LLM

    return HealthResponse(
        status=status,
        services=services,
        timestamp=datetime.utcnow().isoformat()
    )


def _is_image_followup(query: str, history: Optional[List[Dict]] = None) -> bool:
    """Detect if user is asking about a previous image without re-attaching it."""
    if not history:
        return False

    # Keywords that suggest asking about image content
    image_keywords = [
        "image", "picture", "photo", "screenshot", "text in", "lines",
        "what does it say", "read", "extract", "ocr", "the letter",
        "the document", "show me", "can you see", "in the image",
        "from the image", "the file", "attached"
    ]
    query_lower = query.lower()
    asks_about_image = any(kw in query_lower for kw in image_keywords)

    if not asks_about_image:
        return False

    # Check if recent history mentions image analysis
    for msg in history[-4:]:  # Last 2 exchanges
        content = msg.get("content", "").lower()
        if msg.get("role") == "assistant":
            if any(phrase in content for phrase in [
                "image shows", "image appears", "image contains",
                "can see", "the picture", "the photo", "the document",
                "analyzed", "vision model"
            ]):
                return True
        elif msg.get("role") == "user":
            if "[attached:" in content.lower():
                return True

    return False


@app.post("/api/v1/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, query_request: QueryRequest):
    """
    Main query endpoint - orchestrates search across all sources

    Modes:
    - auto: Groq orchestrates, routes to cheapest capable model (default)
    - private: Local only (Ollama), no external APIs
    - research: Deep Perplexity search
    - deep_agent: Multi-step smolagents for complex tasks
    """
    start_time = datetime.utcnow()

    # Route based on mode first, then model
    perplexity_search_mode = "low"

    # Mode determines the overall behavior
    # - auto: orchestrator decides everything
    # - private: local only, no external APIs
    # - research: deep Perplexity search

    # Load global config for instructions
    global_config = await rag_core.get_global_config()
    global_instructions = global_config.get("global_instructions", "")

    # Classify the query early for better routing and response generation
    query_classification = classify_query(query_request.query, query_request.conversation_history)
    logging.getLogger(__name__).info(
        f"Query classification: intent={query_classification['intent'].value}, "
        f"is_correction={query_classification['is_correction']}, "
        f"is_technical={query_classification['is_technical']}"
    )

    try:
        # Log incoming request details for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"Query received: mode={query_request.mode}, model={query_request.model}, has_files={query_request.files is not None and len(query_request.files) > 0 if query_request.files else False}")
        if query_request.files:
            for f in query_request.files:
                logger.info(f"  File: {f.name}, type={f.type}, isImage={f.isImage}, data_len={len(f.data) if f.data else 0}")

        # Check for attached files - route to vision model if images present
        if query_request.files and len(query_request.files) > 0:
            has_images = any(f.isImage for f in query_request.files)
            if has_images:
                logger.info("Routing to vision model for image analysis")
            else:
                logger.info(f"Processing {len(query_request.files)} attached file(s) (no images)")
            result = await query_vision(query_request, query_request.files)

        # Detect follow-up questions about images when no image is attached
        elif _is_image_followup(query_request.query, query_request.conversation_history):
            logging.getLogger(__name__).info("Detected image follow-up without image attached")
            result = {
                "answer": "I don't have access to the image from the previous message. Please re-attach the image with your question so I can analyze it again.",
                "sources": [],
                "confidence": 1.0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "model_used": "system",
            }

        elif query_request.mode == "private":
            # Force local-only mode (no external APIs)
            query_request.model = "local"
            result = await query_local(query_request, global_instructions, query_classification)

        elif query_request.mode == "research":
            # Detect supplier queries - use focused format instead of essay
            query_lower = query_request.query.lower()

            # If query contains a URL, scrape page first so Perplexity has context
            has_url = "https://" in query_lower or "http://" in query_lower
            if has_url:
                logging.getLogger(__name__).info("Research mode: URL detected, scraping page first")
                try:
                    page_result = await _tool_web_search(query_request.query, provider="tavily")
                    page_content = page_result.get("answer", "")
                    if page_content and len(page_content) > 200:
                        enriched_query = f"{query_request.query}\n\nPage content:\n{page_content[:20000]}"
                        query_request.query = enriched_query
                        query_request.conversation_history = None  # Don't pass confusing history
                        logging.getLogger(__name__).info(f"Research mode: scraped {len(page_content)} chars, enriching query")
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Research mode URL scrape failed: {e}")

            supplier_keywords = [
                "supplier", "vendor", "wholesale", "where to buy",
                "isolate", "absolute", "terpene", "essential oil",
                "find ", "looking for ", "source "
            ]
            is_supplier_query = any(kw in query_lower for kw in supplier_keywords)

            if is_supplier_query:
                # Use focused search (table format with URLs) instead of essay
                logging.getLogger(__name__).info("Research mode: detected supplier query, using focused_search")
                focused_result = await perplexity_search.focused_search(
                    query=query_request.query,
                    recency="month"
                )
                result = {
                    "answer": focused_result["answer"],
                    "sources": [
                        Source(type="research", title=s["title"], url=s["url"], snippet="")
                        for s in focused_result["citations"]
                    ],
                    "usage": focused_result.get("usage", {}),
                }
            else:
                # Regular deep research for non-supplier queries
                result = await query_research(query_request)

        elif query_request.mode == "deep_agent":
            # Force multi-step agent research via smolagents
            result = await query_deep_agent(query_request)

        else:  # auto mode (default) - Groq agent with tools
            # ALWAYS use Groq agent in auto mode - it handles web search, Notion, etc.
            # Model selection (haiku/sonnet/opus) only affects complex_reasoning delegation
            # Get project config for context
            project_config = None
            if query_request.project:
                project_config = await rag_core.get_project_config(query_request.project)

            logging.getLogger(__name__).info(f"Using Groq agent with tools (user model pref: {query_request.model})")

            # Groq agent handles everything - routing, tool calls, synthesis
            agent_result = await groq_agent.chat(
                query=query_request.query,
                conversation_history=query_request.conversation_history,
                project_config=project_config,
            )

            # Log tool usage
            if agent_result.get("tool_calls"):
                tools_used = [tc["tool"] for tc in agent_result["tool_calls"]]
                logging.getLogger(__name__).info(f"Groq used tools: {tools_used}")

            # Convert sources to Source objects
            sources = []
            for s in agent_result.get("sources", []):
                if isinstance(s, dict):
                    sources.append(Source(
                        type="web",
                        title=s.get("title", ""),
                        url=s.get("url", ""),
                        snippet=s.get("snippet", ""),
                    ))

            # Build routing info for transparency
            tools_used = []
            claude_model = None
            routing_reasoning = "Groq handled directly"

            # Parse tool calls to show actual providers used and capture tool token usage
            tool_usage = {}  # Token usage from paid tools (Perplexity, Claude)
            for tc in agent_result.get("tool_calls", []):
                tool_name = tc["tool"]
                args = tc.get("args", {})
                result_data = tc.get("result", {}) if isinstance(tc.get("result"), dict) else {}

                # Show actual provider for web_search (what actually ran, not what was requested)
                if tool_name == "web_search":
                    # provider_used tracks what actually succeeded (crawl4ai, tavily_extract, perplexity)
                    actual = result_data.get("provider_used")
                    requested = args.get("provider", "perplexity")
                    provider = actual or requested
                    tools_used.append(provider)
                    # Capture Perplexity token usage for accurate costing
                    if result_data.get("usage"):
                        u = result_data["usage"]
                        tool_usage["input_tokens"] = tool_usage.get("input_tokens", 0) + u.get("input_tokens", 0)
                        tool_usage["output_tokens"] = tool_usage.get("output_tokens", 0) + u.get("output_tokens", 0)
                    if args.get("forced"):
                        routing_reasoning = f"Forced {provider} search (bypassed Groq routing)"
                elif tool_name == "complex_reasoning":
                    tools_used.append("claude")
                    complexity = args.get("complexity", "simple")
                    model_names = {"simple": "haiku", "medium": "sonnet", "critical": "opus"}
                    claude_model = model_names.get(complexity, "haiku")
                    routing_reasoning = f"Delegated to Claude {claude_model.title()} for {complexity} task"
                    if result_data.get("usage"):
                        u = result_data["usage"]
                        tool_usage["input_tokens"] = tool_usage.get("input_tokens", 0) + u.get("input_tokens", 0)
                        tool_usage["output_tokens"] = tool_usage.get("output_tokens", 0) + u.get("output_tokens", 0)
                elif tool_name == "deep_research":
                    tools_used.append("perplexity_pro")
                elif tool_name == "browse_website":
                    tools_used.append("browser")
                elif tool_name in ("search_knowledge_base", "list_directory", "read_file", "search_files"):
                    tools_used.append(tool_name.replace("search_knowledge_base", "knowledge_base"))
                else:
                    tools_used.append(tool_name)

            if tools_used and "Delegated" not in routing_reasoning and "Forced" not in routing_reasoning:
                routing_reasoning = f"Groq used: {', '.join(tools_used)}"

            # Determine pricing model based on tools used (first paid tool takes priority)
            pricing_model = "groq"  # Default free
            if tools_used:
                first_tool = tools_used[0]
                if first_tool in ["tavily", "tavily_extract", "perplexity", "perplexity_pro", "perplexity_focused", "crawl4ai"]:
                    pricing_model = first_tool
                elif first_tool == "claude" and claude_model:
                    _model_lookup = {"haiku": get_claude_haiku_model(), "sonnet": get_claude_sonnet_model(), "opus": get_claude_opus_model()}
                    pricing_model = _model_lookup.get(claude_model, get_claude_haiku_model())

            # Use tool token usage (Perplexity/Claude) if available, otherwise Groq's own
            effective_usage = tool_usage if tool_usage else agent_result.get("usage", {})

            result = {
                "answer": agent_result["answer"],
                "sources": sources,
                "usage": effective_usage,
                "pricing_model": pricing_model,
                "routing_info": {
                    "orchestrator": "groq",
                    "tools_used": tools_used,
                    "claude_model": claude_model,
                    "reasoning": routing_reasoning,
                },
            }
            query_request.model = "groq"  # For response metadata
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Extract usage info from result
        usage = result.get("usage", {})
        tokens = None
        estimated_cost = None
        pricing_model = result.get("pricing_model") or query_request.model
        if usage:
            tokens = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0)
            }
        # Always calculate cost (flat-rate providers don't need tokens)
        estimated_cost = calculate_cost(
            pricing_model,
            tokens["input"] if tokens else 0,
            tokens["output"] if tokens else 0
        )

        # Build routing_info if available
        routing_info = None
        if result.get("routing_info"):
            ri = result["routing_info"]
            routing_info = RoutingInfo(
                orchestrator=ri.get("orchestrator", "unknown"),
                tools_used=ri.get("tools_used", []),
                claude_model=ri.get("claude_model"),
                reasoning=ri.get("reasoning"),
            )

        return QueryResponse(
            query=query_request.query,
            answer=result["answer"],
            mode=query_request.mode,
            sources=result["sources"],
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            confidence=result.get("confidence"),
            model_used=result.get("model_used") or query_request.model,
            tokens=tokens,
            estimated_cost=estimated_cost,
            routing_info=routing_info,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


async def query_chat(request: QueryRequest, global_instructions: str = "", query_classification: Optional[Dict] = None) -> Dict[str, Any]:
    """Direct chat with LLM, no RAG context but with project instructions"""
    # Get project config for custom instructions
    project_config = await rag_core.get_project_config(request.project) if request.project else None

    result = await rag_core.chat(
        query=request.query,
        conversation_history=request.conversation_history,
        model=request.model,
        project_config=project_config,
        global_instructions=global_instructions,
        query_classification=query_classification,
    )

    return {
        "answer": result["text"],
        "sources": [],
        "confidence": None,
        "usage": result.get("usage", {}),
    }


async def query_local(request: QueryRequest, global_instructions: str = "", query_classification: Optional[Dict] = None) -> Dict[str, Any]:
    """Search local documents only, with Haiku orchestration for tool use"""
    # Get project config for custom instructions
    project_config = await rag_core.get_project_config(request.project) if request.project else None

    # Use Haiku orchestration if project has allowed_paths (file access)
    # This ensures reliable tool use for file operations
    if project_config and project_config.get("allowed_paths"):
        logging.getLogger(__name__).info(
            f"Using Haiku orchestrator for project '{request.project}' with file access"
        )

        haiku_result = await rag_core.execute_with_haiku(
            query=request.query,
            project_config=project_config,
            conversation_history=request.conversation_history,
            global_instructions=global_instructions,
        )

        # Convert tool steps to sources for display
        tool_sources = [
            Source(
                type="tool_result",
                title=f"📁 {step.get('tool', 'unknown')}: {step.get('path', '')}",
                url=step.get("path"),
                snippet=step.get("result", "")[:200] + "..." if len(step.get("result", "")) > 200 else step.get("result", ""),
            )
            for step in haiku_result.get("tool_steps", [])
        ]

        return {
            "answer": haiku_result["text"],
            "sources": tool_sources,
            "confidence": 0.9 if haiku_result.get("tool_steps") else 0.5,
            "usage": haiku_result.get("usage", {}),
            "orchestrator": haiku_result.get("orchestrator"),
        }

    # Standard RAG search (no file tools)
    results = await rag_core.search(
        query=request.query,
        project=request.project,
        top_k=request.max_results
    )

    # Generate answer with Ollama
    context = "\n\n".join([r["content"] for r in results])

    # Track which documents were actually read (for citation constraints)
    documents_read = [
        {"title": r["metadata"].get("title", "Document"), "path": r["metadata"].get("path")}
        for r in results
    ]

    gen_result = await rag_core.generate_answer(
        query=request.query,
        context=context,
        conversation_history=request.conversation_history,
        project_config=project_config,
        model=request.model,
        global_instructions=global_instructions,
        query_classification=query_classification,
        documents_read=documents_read,
    )

    sources = [
        Source(
            type="local_doc",
            title=r["metadata"].get("title", "Document"),
            url=r["metadata"].get("path"),
            snippet=r["content"][:200] + ("..." if len(r["content"]) > 200 else ""),
            score=r["score"]
        )
        for r in results
    ]

    return {
        "answer": gen_result["text"],
        "sources": sources,
        "confidence": calculate_confidence(results),
        "usage": gen_result.get("usage", {}),
    }


async def query_web(request: QueryRequest) -> Dict[str, Any]:
    """Search web only (Claude or Perplexity)"""
    # Try Claude first (included in Pro subscription)
    try:
        result = await claude_search.search(request.query, model=request.model)
        return {
            "answer": result["answer"],
            "sources": [
                Source(
                    type="web",
                    title=s["title"],
                    url=s["url"],
                    snippet=s["snippet"]
                )
                for s in result["sources"]
            ],
            "usage": result.get("usage", {}),
        }
    except Exception as e:
        # Fallback to Perplexity (no token tracking)
        result = await perplexity_search.search(request.query)
        return {
            "answer": result["answer"],
            "sources": [
                Source(
                    type="web",
                    title=s["title"],
                    url=s["url"],
                    snippet=s["snippet"]
                )
                for s in result["citations"]
            ],
            "usage": {},
        }


async def query_perplexity(request: QueryRequest, search_mode: str = "low") -> Dict[str, Any]:
    """
    Direct Perplexity search (orchestrator-routed).
    Uses appropriate search_mode: low (cheap), medium, high (thorough).
    Includes conversation history for context continuity.
    """
    result = await perplexity_search.search(
        query=request.query,
        search_mode=search_mode,
        conversation_history=request.conversation_history,
    )

    return {
        "answer": result["answer"],
        "sources": [
            Source(
                type="web",
                title=s["title"],
                url=s["url"],
                snippet=s.get("snippet", "")
            )
            for s in result["citations"]
        ],
        "usage": result.get("usage", {}),
    }


async def query_research(request: QueryRequest) -> Dict[str, Any]:
    """Deep research with Perplexity Pro"""
    result = await perplexity_search.research(
        query=request.query,
        depth="deep",
        conversation_history=request.conversation_history,
    )

    return {
        "answer": result["answer"],
        "sources": [
            Source(
                type="research",
                title=s["title"],
                url=s["url"],
                snippet=s.get("snippet", "")
            )
            for s in result["citations"]
        ],
        "usage": result.get("usage", {}),
    }


async def query_vision(request: QueryRequest, files: List[AttachedFile]) -> Dict[str, Any]:
    """Handle queries with images using Claude's vision API for accurate OCR"""
    import httpx
    import base64
    from config import get_anthropic_api_key

    logger = logging.getLogger(__name__)
    logger.info(f"query_vision called with {len(files)} files")

    try:
        api_key = get_anthropic_api_key()
        if not api_key:
            return {
                "answer": "[Claude API key not configured - cannot process images]",
                "sources": [],
                "confidence": 0,
                "usage": {},
                "model_used": "none",
            }

        # Build content blocks for Claude (images + text)
        content_blocks = []
        file_context = []

        for f in files:
            logger.info(f"Processing file: {f.name}, type: {f.type}, isImage: {f.isImage}")
            if f.isImage:
                # Extract base64 data and media type
                data = f.data
                media_type = f.type or "image/jpeg"
                if data.startswith('data:'):
                    # Parse data URL: data:image/png;base64,xxxxx
                    header, data = data.split(',', 1)
                    if 'image/' in header:
                        media_type = header.split(';')[0].replace('data:', '')

                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    }
                })
            else:
                # For PDFs/text, extract content and add to context
                if f.type == 'application/pdf':
                    try:
                        import io
                        from pypdf import PdfReader
                        pdf_bytes = base64.b64decode(f.data.split(',', 1)[1] if f.data.startswith('data:') else f.data)
                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                        file_context.append(f"--- Content from {f.name} ---\n{text[:5000]}")
                    except Exception as e:
                        logger.error(f"PDF extraction failed: {e}")
                        file_context.append(f"[Could not extract text from {f.name}: {e}]")
                else:
                    # Text file - use content directly
                    file_context.append(f"--- Content from {f.name} ---\n{f.data[:5000]}")

        # Build the prompt
        prompt = request.query or "Please analyze this image and describe what you see."
        if file_context:
            prompt = f"{prompt}\n\n" + "\n\n".join(file_context)

        # Add text prompt after images
        content_blocks.append({"type": "text", "text": prompt})

        # Build messages for Claude API
        messages = []

        # Add conversation history (if any) - text only for history
        if request.conversation_history:
            for msg in request.conversation_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current message with images
        messages.append({"role": "user", "content": content_blocks})

        logger.info(f"Calling Claude vision with {len([b for b in content_blocks if b['type'] == 'image'])} images")

        # Call Claude API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": get_claude_sonnet_model(),
                    "max_tokens": 4096,
                    "messages": messages,
                }
            )
            response.raise_for_status()
            result = response.json()

        # Extract answer from Claude response
        answer = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                answer += block.get("text", "")

        if not answer:
            answer = "No response from Claude vision"

        logger.info(f"Claude vision responded with {len(answer)} chars")

        # Build sources showing what files were analyzed
        sources = [
            Source(
                type="local_doc",
                title=f"📎 {f.name}",
                url=None,
                snippet=f"{'Image' if f.isImage else 'Document'} analyzed by Claude",
            )
            for f in files
        ]

        # Extract usage info from Claude response
        usage = result.get("usage", {})

        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.95,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            "model_used": "claude-sonnet-4.5 (vision)",
            "pricing_model": get_claude_sonnet_model(),
        }

    except httpx.HTTPStatusError as e:
        error_text = e.response.text
        logger.error(f"Claude API error {e.response.status_code}: {error_text}")
        # Return error message instead of crashing
        return {
            "answer": f"Claude vision error: {error_text}",
            "sources": [],
            "confidence": 0,
            "usage": {},
            "model_used": "claude-sonnet-4.5 (error)",
        }
    except Exception as e:
        logger.error(f"query_vision failed: {e}", exc_info=True)
        raise


async def query_deep_agent(request: QueryRequest) -> Dict[str, Any]:
    """Multi-step research with smolagents CodeAgent"""
    agent = get_deep_agent()

    result = await agent.research(
        query=request.query,
        project=request.project,
    )

    # Convert steps to sources for display
    sources = [
        Source(
            type="agent_step",
            title=f"Step: {step.get('tool', 'unknown')}",
            url=None,
            snippet=step.get('input', '')[:200],
        )
        for step in result.get("steps", [])
    ]

    return {
        "answer": result["answer"],
        "sources": sources,
        "confidence": None,
        "usage": {},
        "agent_steps": result.get("steps", []),
    }


async def query_hybrid(request: QueryRequest, global_instructions: str = "", query_classification: Optional[Dict] = None) -> Dict[str, Any]:
    """Hybrid search: local + web"""
    # Run searches in parallel
    local_task = query_local(request, global_instructions, query_classification)
    web_task = query_web(request)

    local_result, web_result = await asyncio.gather(
        local_task, web_task, return_exceptions=True
    )

    # Handle failures gracefully
    if isinstance(local_result, Exception):
        return web_result
    if isinstance(web_result, Exception):
        return local_result

    # Combine results
    combine_result = await combine_answers(
        local_result["answer"],
        web_result["answer"],
        request.query
    )

    combined_sources = local_result["sources"] + web_result["sources"]

    # Merge usage from all sources
    total_usage = {
        "input_tokens": (
            local_result.get("usage", {}).get("input_tokens", 0) +
            web_result.get("usage", {}).get("input_tokens", 0) +
            combine_result.get("usage", {}).get("input_tokens", 0)
        ),
        "output_tokens": (
            local_result.get("usage", {}).get("output_tokens", 0) +
            web_result.get("usage", {}).get("output_tokens", 0) +
            combine_result.get("usage", {}).get("output_tokens", 0)
        ),
    }

    return {
        "answer": combine_result["text"],
        "sources": combined_sources,
        "confidence": (local_result.get("confidence", 0.5) + 0.5) / 2,
        "usage": total_usage,
    }


async def combine_answers(local_answer: str, web_answer: str, query: str) -> Dict[str, Any]:
    """Intelligently combine local and web answers. Returns {"text": str, "usage": dict}."""

    # Validate sources - don't synthesize if one is empty/error
    def is_valid_answer(answer: str) -> bool:
        if not answer or len(answer.strip()) < 20:
            return False
        error_indicators = ["error", "failed", "unavailable", "couldn't find", "no results"]
        answer_lower = answer.lower()
        return not any(ind in answer_lower for ind in error_indicators)

    local_valid = is_valid_answer(local_answer)
    web_valid = is_valid_answer(web_answer)

    # If only one source is valid, return it directly (no synthesis)
    if web_valid and not local_valid:
        return {"text": web_answer, "usage": {}}
    if local_valid and not web_valid:
        return {"text": local_answer, "usage": {}}
    if not local_valid and not web_valid:
        return {"text": "I couldn't find reliable information to answer this question.", "usage": {}}

    # Both sources valid - synthesize them
    prompt = f"""Given the query: "{query}"

Source 1 (Local knowledge):
{local_answer}

Source 2 (Web search):
{web_answer}

Synthesize these into a single comprehensive answer. If they agree, present the unified information. If they contradict, note the discrepancy."""

    # Try Claude first, fall back to Ollama
    try:
        if await claude_search.is_healthy():
            return await claude_search.generate(prompt)
    except Exception:
        pass

    # Fallback: use local Ollama for synthesis
    return await rag_core.generate_answer(query, f"Local: {local_answer}\n\nWeb: {web_answer}")


def calculate_confidence(results: List[Dict]) -> float:
    """Calculate confidence score from search results"""
    if not results:
        return 0.0
    avg_score = sum(r["score"] for r in results) / len(results)
    return min(avg_score, 1.0)


# Pricing per 1M tokens: (input, output)
TOKEN_PRICING = {
    # Claude
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # Perplexity (token cost only; search request fees not included)
    "perplexity": (1.0, 1.0),           # Sonar
    "perplexity_pro": (3.0, 15.0),      # Sonar Pro
    "perplexity_focused": (3.0, 15.0),  # Sonar Pro (focused mode)
}

# Flat per-request pricing (not token-based)
FLAT_PRICING = {
    "tavily": 0,                # Free tier: 1,000 credits/month
    "tavily_extract": 0,        # Free tier: 1,000 credits/month
    "crawl4ai": 0,              # Free (self-hosted)
    "groq": 0,                  # Free tier
    "local": 0,                 # Free (Ollama)
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost based on model and token usage."""
    if model in TOKEN_PRICING:
        rates = TOKEN_PRICING[model]
        return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
    if model in FLAT_PRICING:
        return FLAT_PRICING[model]
    return 0.0


@app.post("/api/v1/index")
async def index_documents(request: IndexRequest):
    """Index new documents into the RAG system"""
    try:
        indexed_count = await rag_core.index_documents(
            documents=request.documents,
            project=request.project
        )
        
        return {
            "status": "success",
            "indexed_count": indexed_count,
            "project": request.project,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects")
async def list_projects():
    """List available project knowledge bases"""
    projects = await rag_core.list_projects()

    # Enrich with config info
    for project in projects:
        config = await rag_core.get_project_config(project["name"])
        if config:
            project["description"] = config.get("description", "")
            project["has_config"] = True
        else:
            project["has_config"] = False

    return {
        "projects": projects,
        "count": len(projects),
        "timestamp": datetime.utcnow().isoformat()
    }


class ProjectConfigRequest(BaseModel):
    description: Optional[str] = ""
    system_prompt: Optional[str] = ""
    instructions: Optional[str] = ""
    allowed_paths: Optional[List[str]] = []


class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    system_prompt: Optional[str] = ""
    instructions: Optional[str] = ""
    allowed_paths: Optional[List[str]] = []


class ProjectFileInfo(BaseModel):
    name: str
    size: int
    modified: str
    indexed: bool


class ProjectFilesResponse(BaseModel):
    project: str
    files: List[ProjectFileInfo]
    count: int


class UploadFilesResponse(BaseModel):
    status: str
    project: str
    uploaded: List[str]
    failed: List[dict]
    indexed_chunks: int


@app.post("/api/v1/projects")
async def create_project(request: CreateProjectRequest):
    """Create a new project with optional configuration. Auto-syncs to VPS."""
    try:
        config = {
            "description": request.description,
            "system_prompt": request.system_prompt,
            "instructions": request.instructions,
            "allowed_paths": request.allowed_paths or [],
        }
        result = await rag_core.create_project(request.name, config)

        # Auto-sync to VPS
        response = {
            "status": "success",
            "project": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            await sync_push_to_vps("https://rag.coopeverything.org")
            response["synced_to_vps"] = True
        except Exception as sync_err:
            response["synced_to_vps"] = False
            response["sync_error"] = str(sync_err)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{name}")
async def get_project_config(name: str):
    """Get a project's full configuration"""
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found or has no config")

    return {
        "project": name,
        "config": config,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.put("/api/v1/projects/{name}")
async def update_project_config(name: str, request: ProjectConfigRequest):
    """Update a project's configuration. Auto-syncs to VPS."""
    # Load existing config or create new
    existing = await rag_core.get_project_config(name) or {}

    # Update fields
    updated = {
        **existing,
        "description": request.description if request.description is not None else existing.get("description", ""),
        "system_prompt": request.system_prompt if request.system_prompt is not None else existing.get("system_prompt", ""),
        "instructions": request.instructions if request.instructions is not None else existing.get("instructions", ""),
        "allowed_paths": request.allowed_paths if request.allowed_paths is not None else existing.get("allowed_paths", []),
    }

    success = await rag_core.save_project_config(name, updated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save project config")

    # Auto-sync to VPS
    response = {
        "status": "success",
        "project": name,
        "config": updated,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        await sync_push_to_vps("https://rag.coopeverything.org")
        response["synced_to_vps"] = True
    except Exception as sync_err:
        response["synced_to_vps"] = False
        response["sync_error"] = str(sync_err)

    return response


@app.delete("/api/v1/projects/{name}")
async def delete_project(name: str):
    """Delete a project entirely: config, KB files, ChromaDB collection."""
    try:
        deleted = []

        # 1. Delete synced config (config/projects/{name}.json)
        synced_config = Path(get_synced_projects_path()) / f"{name}.json"
        if synced_config.exists():
            synced_config.unlink()
            deleted.append("synced_config")

        # 2. Delete synced KB docs (config/project-kb/{name}/)
        synced_kb = Path(get_synced_kb_path()) / name
        if synced_kb.exists():
            shutil.rmtree(synced_kb)
            deleted.append("synced_kb")

        # 3. Delete local project data (data/project-kb/{name}/)
        local_kb = Path(get_project_kb_path()) / name
        if local_kb.exists():
            shutil.rmtree(local_kb)
            deleted.append("local_kb")

        # 4. Delete ChromaDB collection
        if await rag_core.delete_project(name):
            deleted.append("chromadb_collection")

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

        return {
            "status": "success",
            "project": name,
            "deleted": deleted,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/projects/{name}/index")
async def index_project_files(
    name: str,
    force_reindex: bool = False,
    auto_sync: bool = True,
    vps_url: str = "https://rag.coopeverything.org"
):
    """
    Index files from project's allowed_paths directories.

    - Incremental by default: only indexes new/modified files
    - Auto-syncs to VPS after indexing (disable with auto_sync=false)
    - Use force_reindex=true to re-index all files
    """
    result = await rag_core.index_project_files(name, force_reindex=force_reindex)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    response = {
        "status": "success",
        "project": name,
        "indexed_chunks": result.get("indexed_chunks", 0),
        "files": result.get("files", []),
        "skipped": result.get("skipped", 0),
        "total_files": result.get("total_files", 0),
        "message": result.get("message", ""),
        "timestamp": datetime.utcnow().isoformat()
    }

    # Auto-sync to VPS if enabled and we indexed something
    if auto_sync and result.get("indexed_chunks", 0) > 0:
        try:
            sync_result = await sync_push_to_vps(vps_url)
            response["synced_to_vps"] = True
            response["sync_message"] = "Data synced to VPS"
        except Exception as e:
            response["synced_to_vps"] = False
            response["sync_error"] = str(e)

    return response


# ============================================================================
# PROJECT KB FILE ENDPOINTS
# ============================================================================

@app.get("/api/v1/projects/{name}/forms/download/{filename}")
async def download_filled_form(name: str, filename: str):
    """Download a filled form PDF."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    filled_dir = Path(get_project_kb_path()) / name / "filled_forms"
    file_path = filled_dir / safe_name

    # If not found locally, try pulling from Postgres
    if not file_path.exists():
        try:
            from form_sync import sync_filled_forms_from_db
            sync_filled_forms_from_db(name)
        except Exception:
            pass

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")

    return FileResponse(
        path=str(file_path),
        filename=safe_name,
        media_type="application/pdf",
    )


@app.post("/api/v1/projects/{name}/forms/sync")
async def sync_project_forms(name: str):
    """Manually trigger sync of filled forms and data profile from Postgres."""
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    try:
        from form_sync import sync_filled_forms_from_db, sync_data_profile_from_db
        allowed_paths = config.get("allowed_paths", [])
        forms_synced = sync_filled_forms_from_db(name, allowed_paths)
        profile_synced = sync_data_profile_from_db(name)
        return {
            "status": "success",
            "forms_synced": forms_synced,
            "data_profile_synced": profile_synced,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{name}/forms/list")
async def list_filled_forms(name: str):
    """List available filled form PDFs."""
    filled_dir = Path(get_project_kb_path()) / name / "filled_forms"
    if not filled_dir.exists():
        return {"forms": [], "count": 0}

    forms = []
    for f in sorted(filled_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        forms.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "download_url": f"/api/v1/projects/{name}/forms/download/{f.name}",
        })
    return {"forms": forms, "count": len(forms)}


# ============================================================================
# FILING TRACKER ENDPOINTS
# ============================================================================

@app.get("/api/v1/projects/{name}/tracker/status")
async def tracker_status(name: str):
    """Return filing completion statistics as JSON."""
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    try:
        from filing_tracker_export import get_tracker_status
        return get_tracker_status(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{name}/tracker/export")
async def tracker_export(name: str):
    """Download the filing tracker as a CSV file."""
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    try:
        from filing_tracker_export import generate_tracker_csv
        csv_content = generate_tracker_csv(name)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=filing_tracker_{name}.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{name}/files")
async def list_project_files(name: str):
    """
    List KB files uploaded to a project's documents directory.
    Returns file metadata including name, size, modified time, and indexed status.
    """
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

    files = await rag_core.list_project_files(name)
    return ProjectFilesResponse(
        project=name,
        files=[ProjectFileInfo(**f) for f in files],
        count=len(files)
    )


@app.post("/api/v1/projects/{name}/files")
async def upload_project_files(
    name: str,
    files: List[UploadFile] = FastAPIFile(...),
    auto_index: bool = True,
):
    """
    Upload KB files to a project's documents directory.
    Files are validated for extension and size, then optionally auto-indexed.
    """
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

    # Ensure documents directory exists (synced via git)
    project_dir = Path(get_synced_kb_path()) / name / "documents"
    project_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    failed = []

    for file in files:
        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_KB_EXTENSIONS:
            failed.append({"name": file.filename, "error": f"Extension '{ext}' not allowed"})
            continue

        # Read file content
        try:
            content = await file.read()
        except Exception as e:
            failed.append({"name": file.filename, "error": f"Failed to read: {str(e)}"})
            continue

        # Validate size
        if len(content) > MAX_KB_FILE_SIZE:
            failed.append({"name": file.filename, "error": f"File exceeds {MAX_KB_FILE_SIZE // (1024*1024)}MB limit"})
            continue

        # Save file
        try:
            file_path = project_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(content)
            uploaded.append(file.filename)
        except Exception as e:
            failed.append({"name": file.filename, "error": f"Failed to save: {str(e)}"})

    # Auto-index if enabled and we uploaded files
    indexed_chunks = 0
    if auto_index and uploaded:
        try:
            result = await rag_core.index_project_files(name)
            indexed_chunks = result.get("indexed_chunks", 0)
        except Exception as e:
            logger.warning(f"Auto-index failed for project {name}: {e}")

    return UploadFilesResponse(
        status="success" if uploaded else "failed",
        project=name,
        uploaded=uploaded,
        failed=failed,
        indexed_chunks=indexed_chunks
    )


@app.delete("/api/v1/projects/{name}/files/{filename}")
async def delete_project_file(name: str, filename: str):
    """
    Delete a KB file from a project's documents directory.
    Also removes the file's chunks from ChromaDB.
    """
    config = await rag_core.get_project_config(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")

    # Build file path (synced via git)
    file_path = Path(get_synced_kb_path()) / name / "documents" / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Delete from ChromaDB first
    await rag_core.delete_document_by_path(name, str(file_path))

    # Delete the file
    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    return {"status": "success", "project": name, "deleted": filename}


# ============================================================================
# GLOBAL SETTINGS ENDPOINTS
# ============================================================================

class GlobalSettingsRequest(BaseModel):
    default_model: Optional[str] = None
    default_mode: Optional[str] = None
    global_instructions: Optional[str] = None


@app.get("/api/v1/settings")
async def get_global_settings():
    """Get global RAG settings"""
    settings = await rag_core.get_global_config()
    return {
        "settings": settings,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.put("/api/v1/settings")
async def update_global_settings(request: GlobalSettingsRequest):
    """Update global RAG settings"""
    # Load existing settings
    existing = await rag_core.get_global_config()

    # Update only provided fields
    updated = {
        "default_model": request.default_model if request.default_model is not None else existing.get("default_model", "auto"),
        "default_mode": request.default_mode if request.default_mode is not None else existing.get("default_mode", "auto"),
        "global_instructions": request.global_instructions if request.global_instructions is not None else existing.get("global_instructions", ""),
    }

    success = await rag_core.save_global_config(updated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save global settings")

    return {
        "status": "success",
        "settings": updated,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# CHAT HISTORY ENDPOINTS
# ============================================================================

class ChatRequest(BaseModel):
    name: Optional[str] = None
    project: Optional[str] = None
    messages: List[Dict[str, Any]] = []  # Allow metadata objects in messages


class ChatRenameRequest(BaseModel):
    name: str


@app.get("/api/v1/chats")
async def list_chats(project: Optional[str] = None, limit: int = 50):
    """List chat summaries, optionally filtered by project."""
    chats = await rag_core.list_chats(project=project, limit=limit)
    return {
        "chats": chats,
        "count": len(chats),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a full chat by ID."""
    chat = await rag_core.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")

    return {
        "chat": chat,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/chats")
async def create_chat(request: ChatRequest):
    """Create a new chat."""
    chat_data = {
        "name": request.name,
        "project": request.project,
        "messages": request.messages,
    }

    try:
        saved_chat = await rag_core.save_chat(chat_data)
        return {
            "status": "success",
            "chat": saved_chat,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")


@app.put("/api/v1/chats/{chat_id}")
async def update_chat(chat_id: str, request: ChatRequest):
    """Update a chat (add messages, rename, etc.)."""
    existing = await rag_core.get_chat(chat_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")

    # Update fields
    if request.name is not None:
        existing["name"] = request.name
    if request.project is not None:
        existing["project"] = request.project
    if request.messages:
        existing["messages"] = request.messages

    try:
        saved_chat = await rag_core.save_chat(existing)
        return {
            "status": "success",
            "chat": saved_chat,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update chat: {str(e)}")


@app.patch("/api/v1/chats/{chat_id}/rename")
async def rename_chat(chat_id: str, request: ChatRenameRequest):
    """Rename a chat."""
    success = await rag_core.rename_chat(chat_id, request.name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")

    return {
        "status": "success",
        "chat_id": chat_id,
        "name": request.name,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.delete("/api/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat."""
    success = await rag_core.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")

    return {
        "status": "success",
        "chat_id": chat_id,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# DATA SYNC (Local <-> VPS)
# ============================================================================

class SyncImportRequest(BaseModel):
    """Request to import synced data."""
    data: str  # Base64-encoded tarball
    sync_key: str  # JWT_SECRET as sync key


@app.get("/api/v1/sync/export")
async def sync_export():
    """
    Export ChromaDB and project-kb data as base64 tarball.
    Used to sync local indexed data to VPS.
    """
    import tarfile
    import base64
    import io

    try:
        # Get paths
        chromadb_path = Path(get_chromadb_path()).resolve()
        project_kb_path = Path(get_project_kb_path()).resolve()

        # Create tarball in memory
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            # Add chromadb directory
            if chromadb_path.exists():
                tar.add(str(chromadb_path), arcname='chromadb')

            # Add project-kb directory
            if project_kb_path.exists():
                tar.add(str(project_kb_path), arcname='project-kb')

        # Get base64 encoded data
        buffer.seek(0)
        data = base64.b64encode(buffer.read()).decode('utf-8')

        return {
            "status": "success",
            "data": data,
            "size_bytes": len(data),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/api/v1/sync/import")
async def sync_import(request: SyncImportRequest):
    """
    Import ChromaDB and project-kb data from base64 tarball.
    Requires sync_key matching JWT_SECRET for security.
    """
    import tarfile
    import base64
    import io
    import shutil
    from config import get_jwt_secret

    # Verify sync key
    if request.sync_key != get_jwt_secret():
        raise HTTPException(status_code=403, detail="Invalid sync key")

    try:
        # Decode tarball
        data = base64.b64decode(request.data)
        buffer = io.BytesIO(data)

        # Get paths
        chromadb_path = Path(get_chromadb_path()).resolve()
        project_kb_path = Path(get_project_kb_path()).resolve()

        # Backup existing data (just rename)
        backup_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if chromadb_path.exists():
            backup_chromadb = chromadb_path.parent / f"chromadb_backup_{backup_suffix}"
            shutil.move(str(chromadb_path), str(backup_chromadb))

        if project_kb_path.exists():
            backup_kb = project_kb_path.parent / f"project-kb_backup_{backup_suffix}"
            shutil.move(str(project_kb_path), str(backup_kb))

        # Extract tarball
        with tarfile.open(fileobj=buffer, mode='r:gz') as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith('/') or '..' in member.name:
                    raise HTTPException(status_code=400, detail="Invalid tarball content")

            # Extract chromadb to correct location
            tar.extractall(path=str(chromadb_path.parent))

        # Reinitialize RAG core to pick up new data
        await rag_core.initialize()

        return {
            "status": "success",
            "message": "Data imported successfully. ChromaDB reinitialized.",
            "backup_suffix": backup_suffix,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@app.post("/api/v1/sync/push")
async def sync_push_to_vps(vps_url: str = "https://rag.coopeverything.org"):
    """
    Push local data to VPS. Call this from local machine.
    Gets export data and POSTs it to VPS /api/v1/sync/import endpoint.
    """
    import httpx
    from config import get_jwt_secret

    try:
        # Get local export
        export_result = await sync_export()
        data = export_result["data"]

        # Push to VPS
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{vps_url}/api/v1/sync/import",
                json={
                    "data": data,
                    "sync_key": get_jwt_secret()
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"VPS import failed: {response.text}"
                )

            return {
                "status": "success",
                "message": "Data pushed to VPS successfully",
                "vps_response": response.json(),
                "timestamp": datetime.utcnow().isoformat()
            }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="VPS request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)}")


# ============================================================================
# COLLECTION SYNC (Lightweight - documents only, regenerate embeddings)
# ============================================================================

@app.get("/api/v1/sync/export-collection")
async def sync_export_collection(project: str):
    """
    Export a project's ChromaDB collection as JSON (documents + metadata, NOT embeddings).
    Use this to sync indexed documents from local to VPS.
    Embeddings are regenerated on import using the same embedding model.
    """
    try:
        export_data = await rag_core.export_collection(project)
        return export_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


class CollectionImportRequest(BaseModel):
    """Request to import collection data."""
    project: str
    collection_name: Optional[str] = None
    exported_at: Optional[str] = None
    documents: List[Dict[str, Any]]
    total_documents: Optional[int] = None
    overwrite: bool = False  # If True, delete existing collection first


@app.post("/api/v1/sync/import-collection")
async def sync_import_collection(request: CollectionImportRequest):
    """
    Import documents into a project's ChromaDB collection.
    Regenerates embeddings using the local embedding model.
    This is the lightweight sync method - only documents and metadata are transferred.
    """
    try:
        import_data = {
            "project": request.project,
            "documents": request.documents,
        }
        result = await rag_core.import_collection(import_data, overwrite=request.overwrite)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@app.post("/api/v1/sync/push-collection")
async def sync_push_collection_to_vps(
    project: str,
    vps_url: str = "https://rag.coopeverything.org",
    overwrite: bool = False
):
    """
    Push a project's collection to VPS. Call this from local machine.
    Exports documents (not embeddings) and POSTs to VPS import endpoint.
    VPS regenerates embeddings with its local embedding model.
    """
    import httpx

    try:
        # Get local export
        export_data = await rag_core.export_collection(project)

        if not export_data.get("documents"):
            return {
                "status": "warning",
                "message": f"No documents to sync for project '{project}'",
                "timestamp": datetime.utcnow().isoformat()
            }

        # Push to VPS
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{vps_url}/api/v1/sync/import-collection",
                json={
                    "project": export_data["project"],
                    "documents": export_data["documents"],
                    "overwrite": overwrite,
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"VPS import failed: {response.text}"
                )

            return {
                "status": "success",
                "message": f"Collection '{project}' pushed to VPS successfully",
                "documents_synced": len(export_data["documents"]),
                "vps_response": response.json(),
                "timestamp": datetime.utcnow().isoformat()
            }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="VPS request timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)}")


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.post("/api/v1/reload")
async def reload_config_endpoint():
    """Reload environment variables and reinitialize components."""
    try:
        # Reload environment variables from .env files
        reload_env()

        return {
            "status": "success",
            "message": "Configuration reloaded. API keys and settings updated.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Starting RAG-Hybrid System...")

    # Initialize RAG core
    await rag_core.initialize()

    # Auto-index all projects that have allowed_paths configured
    # NOTE: Skipped on startup for faster boot. Use POST /api/v1/projects/{name}/index to index manually.
    # await auto_index_projects()

    # Sync filled forms and data profiles from Postgres
    try:
        from form_sync import ensure_sync_tables, sync_filled_forms_from_db, sync_data_profile_from_db
        ensure_sync_tables()
        projects = await rag_core.list_projects()
        for project in projects:
            name = project.get("name")
            if not name:
                continue
            config = await rag_core.get_project_config(name)
            allowed_paths = config.get("allowed_paths", []) if config else []
            synced = sync_filled_forms_from_db(name, allowed_paths)
            if synced:
                print(f"  Synced {synced} filled form(s) for {name}")
            if sync_data_profile_from_db(name):
                print(f"  Synced data profile for {name}")
    except Exception as e:
        print(f"  Form sync check failed (non-fatal): {e}")

    # Check all services
    health = await health_check()
    print(f"System health: {health.status}")
    print(f"  Services: {health.services}")

    print("RAG System ready!")


async def auto_index_projects():
    """Auto-index files from all projects' allowed_paths on startup."""
    projects = await rag_core.list_projects()
    for project in projects:
        name = project.get("name")
        if not name:
            continue
        config = await rag_core.get_project_config(name)
        if config and config.get("allowed_paths"):
            print(f"Auto-indexing project: {name}")
            try:
                result = await rag_core.index_project_files(name)
                print(f"  Indexed {result.get('indexed', 0)} chunks from {len(result.get('files', []))} files")
            except Exception as e:
                print(f"  Error indexing {name}: {e}")


async def cleanup_orphaned_collections():
    """
    Delete orphaned ChromaDB collections without config.json.
    DISABLED: This was causing data loss on restarts.
    To manually clean up orphaned collections, use the /api/v1/admin/cleanup endpoint.
    """
    # DISABLED - was deleting valid collections on reload
    # If you need to clean up, do it manually via API or script
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down RAG System...")
    await rag_core.cleanup()


if __name__ == "__main__":
    import uvicorn

    port = get_fastapi_port()
    is_production = os.getenv("ENVIRONMENT", "development") == "production"

    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": port,
        "log_level": "info",
    }

    if not is_production:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = [".", "../frontend/components"]
        uvicorn_config["reload_includes"] = ["*.py"]

    uvicorn.run(**uvicorn_config)
