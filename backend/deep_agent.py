"""
Deep Agent Module - smolagents integration for complex multi-step research tasks.
Uses smolagents CodeAgent for agentic workflows while keeping the existing orchestrator
for simple queries (95% of traffic).
"""

import os
import logging
from typing import Optional, Dict, Any, List
from smolagents import CodeAgent, tool, LiteLLMModel

logger = logging.getLogger(__name__)


# --- Custom Tools for the Agent ---

@tool
def web_search(query: str) -> str:
    """
    Search the web for information. Use this when you need current information,
    facts, or data that might not be in your training data.

    Args:
        query: The search query to execute

    Returns:
        Search results as formatted text
    """
    import httpx

    # Use Perplexity API for web search
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: Perplexity API key not configured"

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": query}],
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        return f"Search error: {resp.status_code}"
    except Exception as e:
        return f"Search failed: {e}"


@tool
def read_document(file_path: str) -> str:
    """
    Read the contents of a document from the local filesystem.
    Only reads from allowed project paths.

    Args:
        file_path: Path to the file to read

    Returns:
        File contents as text
    """
    try:
        # Security: only allow certain base paths
        allowed_bases = [
            os.path.abspath("data/"),
            os.path.abspath("documents/"),
        ]
        abs_path = os.path.abspath(file_path)
        if not any(abs_path.startswith(base) for base in allowed_bases):
            return f"Error: Access denied to {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:50000]  # Limit to 50k chars
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def search_local_knowledge(query: str, project: Optional[str] = None) -> str:
    """
    Search the local ChromaDB knowledge base for relevant documents.

    Args:
        query: The search query
        project: Optional project name to filter results

    Returns:
        Relevant document snippets
    """
    import httpx

    try:
        resp = httpx.post(
            "http://localhost:8000/api/v1/search",
            json={"query": query, "project": project, "max_results": 5},
            timeout=15.0,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if not results:
                return "No relevant documents found in knowledge base."

            output = []
            for r in results:
                output.append(f"**{r.get('title', 'Untitled')}** (score: {r.get('score', 0):.2f})")
                output.append(r.get("snippet", "")[:500])
                output.append("---")
            return "\n".join(output)
        return f"Search error: {resp.status_code}"
    except Exception as e:
        return f"Knowledge search failed: {e}"


def _check_ollama_available() -> bool:
    """Check if Ollama is available (running and responsive)."""
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@tool
def summarize_text(text: str, focus: Optional[str] = None) -> str:
    """
    Summarize a long piece of text, optionally focusing on specific aspects.

    Args:
        text: The text to summarize
        focus: Optional aspect to focus the summary on

    Returns:
        A concise summary
    """
    import httpx

    prompt = f"Summarize the following text concisely"
    if focus:
        prompt += f", focusing on {focus}"
    prompt += f":\n\n{text[:20000]}"  # Limit input

    # Try Ollama first, fall back to Claude if unavailable (e.g., on VPS)
    if _check_ollama_available():
        try:
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b",
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60.0,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "Summary failed")
        except Exception as e:
            logger.warning(f"Ollama summarize failed, trying Claude: {e}")

    # Fallback to Claude Haiku via Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Summarization unavailable: no Ollama or Anthropic API key"

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["content"][0]["text"]
        return f"Claude summarization error: {resp.status_code}"
    except Exception as e:
        return f"Summarization failed: {e}"


@tool
def compare_items(items: List[str], criteria: Optional[str] = None) -> str:
    """
    Compare multiple items (products, frameworks, approaches, etc.) systematically.

    Args:
        items: List of items to compare (2-5 items)
        criteria: Optional specific criteria to compare on

    Returns:
        A structured comparison
    """
    if len(items) < 2:
        return "Need at least 2 items to compare"
    if len(items) > 5:
        items = items[:5]

    # Build comparison query
    items_str = ", ".join(items)
    query = f"Compare {items_str}"
    if criteria:
        query += f" based on {criteria}"

    # Use web search to gather info
    return web_search(query + " comparison 2024")


@tool
def read_pdf_form_fields(pdf_path: str) -> str:
    """
    Read all fillable form fields from a PDF file.
    Use this to discover what fields exist in a PDF form before filling them.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        JSON string with field names and types
    """
    import json
    from pdf_tools import PDFFormReader

    result = PDFFormReader.read_form_fields(pdf_path)
    if result["success"]:
        return json.dumps(result["fields"], indent=2)
    return f"Error: {result.get('error', 'Unknown error')}"


@tool
def fill_pdf_form(input_pdf: str, output_pdf: str, field_values: str) -> str:
    """
    Fill a PDF form with provided values and save to a new file.

    Args:
        input_pdf: Path to the input PDF with form fields
        output_pdf: Path to save the filled PDF
        field_values: JSON string mapping field names to values, e.g., '{"Name": "John", "Date": "2026-01-30"}'

    Returns:
        Success message or error
    """
    import json
    from pdf_tools import PDFFormFiller

    try:
        fields = json.loads(field_values) if isinstance(field_values, str) else field_values
    except json.JSONDecodeError as e:
        return f"Error parsing field_values JSON: {e}"

    result = PDFFormFiller.fill_form(input_pdf, output_pdf, fields)
    if result["success"]:
        return result["message"]
    return f"Error: {result.get('error', 'Unknown error')}"


@tool
def download_bankruptcy_form(form_id: str, output_directory: str) -> str:
    """
    Download an official bankruptcy form from uscourts.gov.
    Available forms: 101, 106A, 106C, 106D, 106E, 106G, 106H, 106I, 106J, 106Sum, 107, 108, 121, 122A-1, 122A-2

    Args:
        form_id: The form identifier (e.g., "101", "106A", "122A-1")
        output_directory: Directory to save the downloaded form

    Returns:
        Path to the downloaded file or error message
    """
    import asyncio
    from pdf_tools import PDFDownloader

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(PDFDownloader.download_form(form_id, output_directory))
    if result["success"]:
        return f"Downloaded to: {result['path']}"
    return f"Error: {result.get('error', 'Unknown error')}"


@tool
def verify_pdf(pdf_path: str) -> str:
    """
    Verify that a PDF file is valid and readable. Check its properties.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Verification results including page count and form status
    """
    import json
    from pdf_tools import PDFVerifier

    result = PDFVerifier.verify_pdf(pdf_path)
    return json.dumps(result, indent=2)


# --- Deep Agent Class ---

class DeepResearchAgent:
    """
    A multi-step research agent using smolagents CodeAgent.
    Used for complex queries that need multiple tool calls and reasoning.
    """

    def __init__(self, model_id: str = "ollama/qwen2.5:14b"):
        """
        Initialize the deep research agent.

        Args:
            model_id: LiteLLM model identifier (e.g., "ollama/qwen2.5:14b", "anthropic/claude-sonnet-4-5-20250929")
        """
        self.model_id = model_id
        self.tools = [
            web_search, read_document, search_local_knowledge, summarize_text, compare_items,
            read_pdf_form_fields, fill_pdf_form, download_bankruptcy_form, verify_pdf
        ]
        self._agent = None

    def _get_agent(self) -> CodeAgent:
        """Lazy-load the agent."""
        if self._agent is None:
            model = LiteLLMModel(model_id=self.model_id)
            self._agent = CodeAgent(
                tools=self.tools,
                model=model,
                max_steps=10,
                verbosity_level=1,
            )
        return self._agent

    async def research(
        self,
        query: str,
        project: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a deep research task.

        Args:
            query: The research question or task
            project: Optional project context
            context: Optional additional context

        Returns:
            Dict with 'answer', 'steps', 'sources'
        """
        agent = self._get_agent()

        # Build the task prompt
        task = f"Research the following question thoroughly:\n\n{query}"
        if project:
            task += f"\n\nContext: This is for the project '{project}'. Search local knowledge first."
        if context:
            task += f"\n\nAdditional context: {context}"

        task += "\n\nProvide a comprehensive answer with sources."

        try:
            # Run the agent (smolagents is sync, wrap for async)
            import asyncio
            result = await asyncio.to_thread(agent.run, task)

            # Extract steps from agent memory
            steps = []
            if hasattr(agent, 'memory') and agent.memory:
                # AgentMemory is not directly iterable - access .steps attribute
                memory_steps = getattr(agent.memory, 'steps', None)
                if memory_steps:
                    for step in memory_steps:
                        if hasattr(step, 'tool_calls'):
                            for tc in step.tool_calls:
                                steps.append({
                                    "tool": tc.name if hasattr(tc, 'name') else str(tc),
                                    "input": str(tc.arguments)[:200] if hasattr(tc, 'arguments') else "",
                                })

            return {
                "answer": str(result),
                "steps": steps,
                "sources": [],  # Could extract from web_search results
                "model_used": self.model_id,
            }

        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            return {
                "answer": f"Research failed: {str(e)}",
                "steps": [],
                "sources": [],
                "error": str(e),
            }


# --- Query Detection ---

def is_deep_research_query(query: str) -> bool:
    """
    Detect if a query should use the deep research agent.

    Args:
        query: The user's query

    Returns:
        True if this needs multi-step research
    """
    query_lower = query.lower()

    # Explicit triggers
    explicit_patterns = [
        "research thoroughly",
        "deep dive",
        "comprehensive analysis",
        "compare multiple",
        "investigate",
        "analyze in depth",
        "full report",
        "detailed comparison",
    ]

    if any(pattern in query_lower for pattern in explicit_patterns):
        return True

    # Complex query indicators (needs multiple sources)
    complexity_indicators = [
        # Multi-part questions
        " and " in query_lower and "?" in query,
        # Comparison requests
        query_lower.startswith("compare "),
        "vs" in query_lower or "versus" in query_lower,
        # Research-style queries
        "what are the pros and cons" in query_lower,
        "advantages and disadvantages" in query_lower,
        # Market research
        "best " in query_lower and " 2024" in query_lower,
        "top " in query_lower and " for " in query_lower,
    ]

    return any(complexity_indicators)


# --- Singleton Instance ---

_deep_agent: Optional[DeepResearchAgent] = None


def get_deep_agent(model_id: str = None) -> DeepResearchAgent:
    """Get or create the singleton deep agent.

    Args:
        model_id: LiteLLM model ID. If None, auto-detects:
                  - Uses Ollama if available (local GPU)
                  - Falls back to Claude Haiku if not (VPS)
    """
    global _deep_agent

    if model_id is None:
        # Auto-detect: use Ollama if available, otherwise Claude
        ollama_ok = _check_ollama_available()
        model_id = "ollama/qwen2.5:14b" if ollama_ok else "anthropic/claude-haiku-4-5-20251001"
        logger.info(f"Deep agent auto-selected model: {model_id} (ollama_available={ollama_ok})")

    if _deep_agent is None or _deep_agent.model_id != model_id:
        _deep_agent = DeepResearchAgent(model_id=model_id)
    return _deep_agent
