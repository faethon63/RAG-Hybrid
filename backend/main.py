"""
RAG-Hybrid System - Main Backend API
FastAPI server that orchestrates all RAG components
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

# Ensure backend directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import reload_env, get_log_level, get_fastapi_port, get_project_kb_path, get_chromadb_collection, get_chromadb_path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from rag_core import RAGCore
from search_integrations import ClaudeSearch, PerplexitySearch, TavilySearch, IdealistaSearch
from orchestrator import QueryOrchestrator
from groq_agent import groq_agent, GroqAgent
from deep_agent import get_deep_agent, is_deep_research_query
from auth import authenticate_user
from query_classifier import QueryClassifier, classify_query

logging.basicConfig(level=get_log_level())

# Initialize FastAPI
app = FastAPI(
    title="TogetherOS RAG System",
    description="Hybrid RAG system with Claude, Perplexity, and local knowledge",
    version="1.0.0"
)

# CORS middleware - configurable for production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
rag_core = RAGCore()
claude_search = ClaudeSearch()
perplexity_search = PerplexitySearch()
tavily_search = TavilySearch()
idealista_search = IdealistaSearch()
query_orchestrator = QueryOrchestrator()

# Register tool handlers for GroqAgent
async def _tool_web_search(query: str, provider: str = "perplexity") -> Dict[str, Any]:
    """
    Web search tool with provider selection.
    - perplexity: Fast, default (Sonar)
    - perplexity_pro: Thorough (Sonar Pro) - use when user asks for deep/thorough search
    - tavily: Better for specific URLs
    """
    logger = logging.getLogger(__name__)
    logger.info(f"web_search called with provider={provider}, query={query[:100]}...")

    if provider == "perplexity_pro":
        logger.info("Using Perplexity Sonar Pro (high mode)")
        return await perplexity_search.search(query=query, search_mode="high")
    elif provider == "tavily":
        logger.info("Using Tavily for specific URLs")
        return await tavily_search.search(query=query, search_depth="advanced")
    else:
        # Default: perplexity
        return await perplexity_search.search(query=query, search_mode="low")


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


async def _tool_complex_reasoning(task: str, context: str = "") -> Dict[str, Any]:
    """Delegate complex reasoning to Claude."""
    full_query = f"{task}\n\nContext: {context}" if context else task
    return await claude_search.search(full_query)


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
                except:
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


groq_agent.register_tool_handler("web_search", _tool_web_search)
groq_agent.register_tool_handler("search_listings", _tool_search_listings)
groq_agent.register_tool_handler("deep_research", _tool_deep_research)
groq_agent.register_tool_handler("complex_reasoning", _tool_complex_reasoning)
groq_agent.register_tool_handler("github_search", _tool_github_search)
groq_agent.register_tool_handler("notion_tool", _tool_notion)

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
        "name": "TogetherOS RAG System",
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
async def query(request: QueryRequest):
    """
    Main query endpoint - orchestrates search across all sources

    Modes:
    - local: Search only local documents (fast, private)
    - web: Web search only (Claude/Perplexity)
    - research: Deep research with Perplexity
    - hybrid: Combine local + web (default)
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
    query_classification = classify_query(request.query, request.conversation_history)
    logging.getLogger(__name__).info(
        f"Query classification: intent={query_classification['intent'].value}, "
        f"is_correction={query_classification['is_correction']}, "
        f"is_technical={query_classification['is_technical']}"
    )

    try:
        # Log incoming request details for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"Query received: mode={request.mode}, model={request.model}, has_files={request.files is not None and len(request.files) > 0 if request.files else False}")
        if request.files:
            for f in request.files:
                logger.info(f"  File: {f.name}, type={f.type}, isImage={f.isImage}, data_len={len(f.data) if f.data else 0}")

        # Check for attached files - route to vision model if images present
        if request.files and len(request.files) > 0:
            has_images = any(f.isImage for f in request.files)
            if has_images:
                logger.info("Routing to vision model for image analysis")
            else:
                logger.info(f"Processing {len(request.files)} attached file(s) (no images)")
            result = await query_vision(request, request.files)

        # Detect follow-up questions about images when no image is attached
        elif _is_image_followup(request.query, request.conversation_history):
            logging.getLogger(__name__).info("Detected image follow-up without image attached")
            result = {
                "answer": "I don't have access to the image from the previous message. Please re-attach the image with your question so I can analyze it again.",
                "sources": [],
                "confidence": 1.0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "model_used": "system",
            }

        elif request.mode == "private":
            # Force local-only mode (no external APIs)
            request.model = "local"
            result = await query_local(request, global_instructions, query_classification)

        elif request.mode == "research":
            # Force deep research via Perplexity
            result = await query_research(request)

        elif request.mode == "deep_agent":
            # Force multi-step agent research via smolagents
            result = await query_deep_agent(request)

        else:  # auto mode (default) - Groq agent with tools
            # Use Groq as the conversational agent with tool access
            if request.model == "auto":
                # Get project config for context
                project_config = None
                if request.project:
                    project_config = await rag_core.get_project_config(request.project)

                logging.getLogger(__name__).info("Using Groq agent with tools")

                # Groq agent handles everything - routing, tool calls, synthesis
                agent_result = await groq_agent.chat(
                    query=request.query,
                    conversation_history=request.conversation_history,
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

                result = {
                    "answer": agent_result["answer"],
                    "sources": sources,
                    "usage": agent_result.get("usage", {}),
                }
                request.model = "groq"  # For response metadata

            # Handle specific model requests (not auto)
            elif request.model == "perplexity":
                result = await query_perplexity(request, search_mode="low")
            elif request.model == "deep_agent":
                result = await query_deep_agent(request)
            elif request.model == "local":
                if request.project:
                    result = await query_local(request, global_instructions, query_classification)
                else:
                    result = await query_chat(request, global_instructions, query_classification)
            else:
                # Claude models - use hybrid (local + web)
                result = await query_hybrid(request, global_instructions, query_classification)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Extract usage info from result
        usage = result.get("usage", {})
        tokens = None
        estimated_cost = None
        if usage:
            tokens = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0)
            }
            # Use actual model for pricing (result may specify pricing_model for vision etc)
            pricing_model = result.get("pricing_model") or request.model
            estimated_cost = calculate_cost(
                pricing_model,
                tokens["input"],
                tokens["output"]
            )

        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            mode=request.mode,
            sources=result["sources"],
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            confidence=result.get("confidence"),
            model_used=result.get("model_used") or request.model,
            tokens=tokens,
            estimated_cost=estimated_cost
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
        depth="deep"
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
                    "model": "claude-sonnet-4-5-20250929",
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
            "pricing_model": "claude-sonnet-4-5-20250929",
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
PRICING = {
    "claude-opus-4-5-20251101": (15.0, 75.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "local": (0, 0),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost based on model and token usage."""
    rates = PRICING.get(model, (0, 0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


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
    messages: List[Dict[str, str]] = []


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
