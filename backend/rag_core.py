"""
RAG Core Module
ChromaDB vector store, sentence-transformer embeddings, Ollama LLM calls,
document indexing, and project management.

Updated 2026-01-31: Added Claude Haiku orchestration for reliable tool use.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from file_tools import parse_tool_call, execute_tool, TOOL_DEFINITIONS
from config import (
    get_ollama_host,
    get_ollama_model,
    get_chromadb_path,
    get_chromadb_collection,
    get_embedding_model,
    get_top_k,
    get_similarity_threshold,
    get_max_tokens,
    get_temperature,
    get_project_kb_path,
    get_rag_config_path,
    get_anthropic_api_key,
    get_chats_path,
    get_database_url,
)

logger = logging.getLogger(__name__)

# Chunk settings
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks


class RAGCore:
    """Core RAG engine: embed, store, search, generate."""

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        """Load embedding model and connect to ChromaDB."""
        if self._initialized:
            return

        logger.info("Initializing RAG core...")

        # Load sentence-transformer embedding model
        embedding_model = get_embedding_model()
        logger.info(f"Loading embedding model: {embedding_model}")
        self._embedder = SentenceTransformer(embedding_model)

        # Connect to ChromaDB (persistent on disk)
        db_path = str(Path(get_chromadb_path()).resolve())
        os.makedirs(db_path, exist_ok=True)
        logger.info(f"Connecting to ChromaDB at: {db_path}")
        self._chroma_client = chromadb.PersistentClient(path=db_path)

        # HTTP client for Ollama
        self._http_client = httpx.AsyncClient(timeout=120.0)

        self._initialized = True
        logger.info("RAG core initialized.")

    async def cleanup(self):
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def is_healthy(self) -> bool:
        """Overall health: embedder loaded and ChromaDB accessible."""
        return self._embedder is not None and self._chroma_client is not None

    async def check_ollama(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=10.0)
            resp = await self._http_client.get(f"{get_ollama_host()}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def check_chromadb(self) -> bool:
        """Check if ChromaDB is accessible."""
        try:
            if self._chroma_client:
                self._chroma_client.heartbeat()
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using sentence-transformers."""
        if not self._embedder:
            raise RuntimeError("RAG core not initialized. Call initialize() first.")
        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _get_collection(self, project: Optional[str] = None) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a project."""
        collection_name = get_chromadb_collection()
        name = f"{collection_name}_{project}" if project else collection_name
        # Sanitize collection name (ChromaDB requires 3-63 chars, alphanumeric + _ -)
        name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        if len(name) < 3:
            name = name + "___"
        return self._chroma_client.get_or_create_collection(
            name=name[:63],
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Document indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _doc_id(content: str, idx: int = 0) -> str:
        """Generate a deterministic document ID."""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"doc_{h}_{idx}"

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        project: Optional[str] = None,
    ) -> int:
        """
        Index a list of documents into ChromaDB.

        Each document dict should have:
          - "content": str (required)
          - "title": str (optional)
          - "path": str (optional)
          - "metadata": dict (optional, extra metadata)
        """
        if not self._initialized:
            await self.initialize()

        collection = self._get_collection(project)
        total_indexed = 0

        for doc in documents:
            content = doc.get("content", "")
            if not content.strip():
                continue

            title = doc.get("title", "Untitled")
            path = doc.get("path", "")
            extra_meta = doc.get("metadata", {})

            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                doc_id = self._doc_id(content, i)
                embedding = self.embed([chunk])[0]

                metadata = {
                    "title": title,
                    "path": path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                metadata.update(extra_meta)

                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata],
                )
                total_indexed += 1

        logger.info(f"Indexed {total_indexed} chunks into collection '{collection.name}'")
        return total_indexed

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        project: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB for documents similar to the query.

        Returns list of dicts with keys: content, score, metadata.
        """
        if not self._initialized:
            await self.initialize()

        if top_k is None:
            top_k = get_top_k()
        if threshold is None:
            threshold = get_similarity_threshold()

        collection = self._get_collection(project)

        # Check if collection has any documents
        if collection.count() == 0:
            return []

        query_embedding = self.embed([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score: 1 - (distance / 2)
                score = 1.0 - (dist / 2.0)
                if score >= threshold:
                    hits.append({
                        "content": doc,
                        "score": round(score, 4),
                        "metadata": meta or {},
                    })

        return hits

    # ------------------------------------------------------------------
    # LLM generation (Ollama)
    # ------------------------------------------------------------------

    async def generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        project_config: Optional[Dict[str, Any]] = None,
        enable_tools: bool = True,
        max_tool_iterations: int = 3,
        model: str = "local",
        global_instructions: Optional[str] = None,
        query_classification: Optional[Dict[str, Any]] = None,
        documents_read: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer using Ollama or Claude, grounded in the provided context.
        Supports conversation history, project-specific instructions, and file tools.
        Now supports query classification for flexible prompts and citation tracking.
        Returns {"text": str, "usage": dict} with token counts for Claude.
        """
        # Build system prompt from project config
        system_prompt = "You are a helpful assistant."
        allowed_paths = []

        # Determine if we should use flexible mode (for non-task queries)
        use_flexible = False
        if query_classification:
            from query_classifier import QueryIntent
            intent = query_classification.get("intent")
            # Use flexible mode for meta-questions, technical questions, corrections
            if intent in [QueryIntent.META_QUESTION, QueryIntent.TECHNICAL_CODING, QueryIntent.CONVERSATION]:
                use_flexible = True
            # Also use flexible if no documents are required
            if not query_classification.get("requires_documents"):
                use_flexible = True

        if project_config:
            if use_flexible and project_config.get("flexible_prompt"):
                # Use flexible prompt for non-task queries
                system_prompt = project_config["flexible_prompt"]
            else:
                if project_config.get("system_prompt"):
                    system_prompt = project_config["system_prompt"]
                if project_config.get("instructions"):
                    system_prompt += f"\n\nInstructions: {project_config['instructions']}"
            allowed_paths = project_config.get("allowed_paths", [])

        # Add context modifier based on query classification
        if query_classification:
            from query_classifier import QueryClassifier
            context_modifier = QueryClassifier.get_context_modifier(query_classification)
            if context_modifier:
                system_prompt = f"{context_modifier}\n\n{system_prompt}"

        # Add citation constraints based on documents actually read
        if documents_read:
            doc_names = [d.get("title", "Unknown") for d in documents_read]
            citation_notice = (
                f"\n\nCITATION CONSTRAINT: Only cite from these retrieved documents: {', '.join(doc_names)}. "
                f"Do NOT fabricate page numbers or references to other documents."
            )
            system_prompt += citation_notice
        elif context.strip():
            # Context provided but no document tracking
            pass
        else:
            # No documents retrieved
            system_prompt += (
                "\n\nCITATION CONSTRAINT: No documents were retrieved. "
                "Do NOT cite page numbers or document sections unless you use tools to read them."
            )

        # Prepend global instructions if provided
        if global_instructions:
            system_prompt = f"{global_instructions}\n\n{system_prompt}"

        # Add current date to system prompt
        current_date = datetime.now().strftime("%B %d, %Y")
        system_prompt = f"Today's date is {current_date}.\n\n{system_prompt}"

        # Add file tools instructions if enabled and paths are allowed
        tool_instructions = ""
        if enable_tools and allowed_paths:
            tool_instructions = f"\n\n{TOOL_DEFINITIONS}\nAllowed paths: {', '.join(allowed_paths)}"

        # Build conversation history section (last 6 messages max)
        history_section = ""
        if conversation_history:
            recent_history = conversation_history[-6:]
            history_lines = []
            for msg in recent_history:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            if history_lines:
                history_section = "Previous conversation:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""{system_prompt}{tool_instructions}

Answer the following question based on the provided context and conversation history.
If the context doesn't contain enough information, say so honestly.

{history_section}Context:
{context}

Question: {query}

Answer:"""

        # Track cumulative usage across tool iterations
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        # Tool execution loop
        iteration = 0
        response_text = ""
        actual_model_used = model
        while iteration < max_tool_iterations:
            if model == "local":
                # Use Ollama with Claude Haiku fallback
                result = await self._call_ollama_with_fallback(prompt)
                response_text = result["text"]
                actual_model_used = result.get("model_used", "ollama")
                usage = result.get("usage", {})
                total_usage["input_tokens"] += usage.get("input_tokens", 0)
                total_usage["output_tokens"] += usage.get("output_tokens", 0)
            else:
                result = await self._call_claude(prompt, model)
                response_text = result["text"]
                actual_model_used = model
                usage = result.get("usage", {})
                total_usage["input_tokens"] += usage.get("input_tokens", 0)
                total_usage["output_tokens"] += usage.get("output_tokens", 0)

            # Check if response contains a tool call
            if not enable_tools or not allowed_paths:
                # Strip any fake tool output the LLM might have generated
                response_text = self._strip_fake_tool_output(response_text)
                return {"text": response_text, "usage": total_usage, "model_used": actual_model_used}

            tool_call = parse_tool_call(response_text)
            if not tool_call:
                # No tool call found - strip any fake tool output and return
                response_text = self._strip_fake_tool_output(response_text)
                return {"text": response_text, "usage": total_usage, "model_used": actual_model_used}

            # Execute the tool
            logger.info(f"Executing tool: {tool_call.get('tool')} on path: {tool_call.get('path')}")
            tool_result = execute_tool(tool_call, allowed_paths)

            # Format tool result for follow-up prompt
            if tool_result.get("success"):
                if tool_call.get("tool") == "read_file":
                    result_text = f"File contents of {tool_call.get('path')}:\n```\n{tool_result.get('content', '')[:5000]}\n```"
                elif tool_call.get("tool") == "list_dir":
                    entries = tool_result.get("entries", [])
                    entry_list = "\n".join([f"  - {e['name']} ({e['type']})" for e in entries])
                    result_text = f"Directory contents of {tool_call.get('path')}:\n{entry_list}"
                elif tool_call.get("tool") == "search_files":
                    matches = tool_result.get("matches", [])
                    match_list = "\n".join([f"  - {m['path']}" for m in matches])
                    result_text = f"Files matching '{tool_call.get('pattern')}' in {tool_call.get('path')}:\n{match_list}"
                elif tool_call.get("tool") == "write_file":
                    result_text = f"Successfully wrote file: {tool_result.get('message', '')}"
                else:
                    result_text = f"Tool result: {tool_result}"
            else:
                result_text = f"Tool error: {tool_result.get('error', 'Unknown error')}"

            # Create follow-up prompt with tool result
            prompt = f"""{system_prompt}

Tool execution result:
{result_text}

Based on this result, please answer the user's original question:
{query}

Answer:"""

            iteration += 1

        # If we hit max iterations, return the last response
        return {"text": response_text, "usage": total_usage, "model_used": actual_model_used}

    def _strip_fake_tool_output(self, text: str) -> str:
        """
        Remove fake tool output that the LLM generated without actually using tools.
        This prevents responses like "Tool error: File not found" when no tool was invoked.
        """
        import re

        # Only strip if there's no actual <tool> tag (meaning LLM is faking tool output)
        if '<tool>' in text:
            return text

        # Patterns for fake tool errors/output
        fake_patterns = [
            r'\*\*Tool (?:Execution )?(?:Result|Error)[:\*]*\*\*[^\n]*(?:\n|$)',
            r'Tool error:[^\n]*(?:\n|$)',
            r'Tool result:[^\n]*(?:\n|$)',
            r'\[Tool error:[^\]]*\]',
            r'Error reading file:[^\n]*(?:\n|$)',
            r'File not found:[^\n]*(?:\n|$)',
        ]

        cleaned = text
        for pattern in fake_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    async def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama's generate API. Returns None if Ollama unavailable."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=120.0)

        try:
            resp = await self._http_client.post(
                f"{get_ollama_host()}/api/generate",
                json={
                    "model": get_ollama_model(),
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": get_max_tokens(),
                        "temperature": get_temperature(),
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return None  # Signal to fall back to Claude
        except httpx.ConnectError:
            logger.info("Ollama not running, will fall back to Claude Haiku")
            return None  # Signal to fall back to Claude
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None  # Signal to fall back to Claude

    async def _call_ollama_with_fallback(self, prompt: str) -> Dict[str, Any]:
        """
        Try Ollama first, fall back to Claude Haiku if unavailable.
        Returns {"text": str, "usage": dict, "model_used": str}.
        """
        result = await self._call_ollama(prompt)
        if result is not None:
            return {"text": result, "usage": {}, "model_used": "ollama"}
        logger.info("Ollama unavailable, falling back to Claude Haiku")
        claude_result = await self._call_claude(prompt, model="claude-haiku-4-5-20251001")
        claude_result["model_used"] = "claude-haiku-4-5-20251001"
        return claude_result

    async def _call_claude(self, prompt: str, model: str) -> Dict[str, Any]:
        """Call Claude API for generation. Returns {"text": str, "usage": dict}."""
        api_key = get_anthropic_api_key()
        if not api_key or api_key.startswith("your_"):
            return {"text": "[Claude API key not configured]", "usage": {}}

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=120.0)

        try:
            resp = await self._http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": get_max_tokens(),
                    "temperature": get_temperature(),
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
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
                }
            }
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Claude HTTP error {e.response.status_code}: {error_body}")
            return {"text": f"[Claude error {e.response.status_code}: {error_body}]", "usage": {}}
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return {"text": f"[Claude error: {e}]", "usage": {}}

    # ------------------------------------------------------------------
    # Haiku Orchestrator (for reliable tool use)
    # ------------------------------------------------------------------

    async def _call_haiku_orchestrator(
        self,
        query: str,
        project_config: Dict[str, Any],
        available_tools: List[str],
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use Claude Haiku to decide what action to take.
        Returns: {"action": "respond" | "use_tool" | "delegate", ...}
        """
        api_key = get_anthropic_api_key()
        if not api_key or api_key.startswith("your_"):
            return {"action": "respond", "reason": "No API key configured"}

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=60.0)

        allowed_paths = project_config.get("allowed_paths", [])
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Build context from tool results
        context_parts = []
        for tr in tool_results:
            context_parts.append(f"Tool: {tr.get('tool')}\nResult: {tr.get('result', '')[:2000]}")
        context_str = "\n\n".join(context_parts) if context_parts else "No previous tool results."

        system_prompt = f"""You are an orchestrator that decides what action to take to answer user queries.
Today's date: {current_date}

You have access to these file tools: {', '.join(available_tools)}
File access is allowed in these paths: {', '.join(allowed_paths)}

Previous tool results:
{context_str}

Based on the user's query, decide the next action. Respond with ONLY valid JSON:

For file operations:
{{"action": "use_tool", "tool": "list_dir", "path": "<path>"}}
{{"action": "use_tool", "tool": "read_file", "path": "<path>"}}
{{"action": "use_tool", "tool": "search_files", "path": "<path>", "pattern": "<glob>"}}

When you have enough information to answer:
{{"action": "respond", "summary": "<brief summary of what you found>"}}

For complex queries needing Claude Sonnet:
{{"action": "delegate", "model": "sonnet", "reason": "<why>"}}

For web search queries:
{{"action": "delegate", "model": "perplexity", "reason": "<why>"}}

IMPORTANT RULES:
1. If asked about files/documents, ALWAYS use list_dir first to see what exists
2. NEVER guess file names - use list_dir to discover them
3. After reading files, respond with action="respond" and include a summary
4. Only respond with valid JSON, nothing else"""

        try:
            resp = await self._http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 500,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": query}],
                    "system": system_prompt,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            text = ""
            if data.get("content"):
                text = data["content"][0].get("text", "").strip()

            # Parse JSON response
            try:
                decision = json.loads(text)
                logger.info(f"Haiku decision: {decision}")
                return decision
            except json.JSONDecodeError:
                logger.warning(f"Haiku returned non-JSON: {text}")
                return {"action": "respond", "reason": "Could not parse orchestrator response"}

        except Exception as e:
            logger.error(f"Haiku orchestrator error: {e}")
            return {"action": "respond", "reason": f"Orchestrator error: {e}"}

    async def execute_with_haiku(
        self,
        query: str,
        project_config: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        global_instructions: Optional[str] = None,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute a query using Haiku as orchestrator for tool decisions.
        Returns {"text": str, "usage": dict, "tool_steps": list}.
        """
        available_tools = ["list_dir", "read_file", "search_files", "write_file"]
        allowed_paths = project_config.get("allowed_paths", [])
        tool_results = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for i in range(max_iterations):
            decision = await self._call_haiku_orchestrator(
                query, project_config, available_tools, tool_results
            )

            action = decision.get("action", "respond")

            if action == "respond":
                # Generate final response using Ollama (free) with gathered context
                context_parts = []
                for tr in tool_results:
                    context_parts.append(f"### {tr.get('tool')} result:\n{tr.get('result', '')}")
                context = "\n\n".join(context_parts) if context_parts else ""

                # Add summary if provided
                summary = decision.get("summary", "")
                if summary:
                    context = f"Summary: {summary}\n\n{context}"

                result = await self.generate_answer(
                    query=query,
                    context=context,
                    conversation_history=conversation_history,
                    project_config=project_config,
                    enable_tools=False,  # Tools already handled by Haiku
                    global_instructions=global_instructions,
                )

                return {
                    "text": result.get("text", ""),
                    "usage": total_usage,
                    "tool_steps": tool_results,
                    "orchestrator": "haiku",
                }

            elif action == "use_tool":
                tool_name = decision.get("tool", "")
                tool_path = decision.get("path", "")

                if not tool_path or not tool_name:
                    logger.warning(f"Invalid tool call: {decision}")
                    continue

                # Execute the tool
                tool_call = {
                    "tool": tool_name,
                    "path": tool_path,
                    "pattern": decision.get("pattern"),
                    "content": decision.get("content"),
                }

                logger.info(f"Haiku executing tool: {tool_name} on {tool_path}")
                result = execute_tool(tool_call, allowed_paths)

                # Format result for context
                if result.get("success"):
                    if tool_name == "list_dir":
                        entries = result.get("entries", [])
                        result_text = "\n".join([f"- {e['name']} ({e['type']})" for e in entries])
                    elif tool_name == "read_file":
                        result_text = result.get("content", "")[:5000]
                    elif tool_name == "search_files":
                        matches = result.get("matches", [])
                        result_text = "\n".join([m["path"] for m in matches])
                    else:
                        result_text = str(result)
                else:
                    result_text = f"Error: {result.get('error', 'Unknown error')}"

                tool_results.append({
                    "tool": tool_name,
                    "path": tool_path,
                    "result": result_text,
                })

            elif action == "delegate":
                model = decision.get("model", "sonnet")
                reason = decision.get("reason", "Complex query")

                if model == "perplexity":
                    # Use Perplexity for web search
                    from search_integrations import PerplexitySearch
                    perplexity = PerplexitySearch()
                    result = await perplexity.search(query)
                    return {
                        "text": result.get("answer", ""),
                        "usage": result.get("usage", {}),
                        "citations": result.get("citations", []),
                        "orchestrator": "haiku->perplexity",
                        "reason": reason,
                    }
                else:
                    # Use Claude Sonnet for complex reasoning
                    context = "\n\n".join([
                        f"### {tr.get('tool')} result:\n{tr.get('result', '')}"
                        for tr in tool_results
                    ]) if tool_results else ""

                    result = await self.generate_answer(
                        query=query,
                        context=context,
                        conversation_history=conversation_history,
                        project_config=project_config,
                        enable_tools=False,
                        model="claude-sonnet-4-5-20250929",
                        global_instructions=global_instructions,
                    )

                    return {
                        "text": result.get("text", ""),
                        "usage": result.get("usage", {}),
                        "tool_steps": tool_results,
                        "orchestrator": "haiku->sonnet",
                        "reason": reason,
                    }

        # Max iterations reached
        return {
            "text": "I couldn't complete this request within the allowed steps. Please try a more specific query.",
            "usage": total_usage,
            "tool_steps": tool_results,
            "orchestrator": "haiku",
            "error": "max_iterations_reached",
        }

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    async def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects from project-kb directory.

        Iterates over project-kb directories with config.json files,
        and shows document count from ChromaDB if collection exists.
        """
        if not self._initialized:
            await self.initialize()

        projects = []
        project_kb_path = Path(get_project_kb_path()).resolve()
        logger.info(f"Looking for projects in: {project_kb_path}")

        if not project_kb_path.exists():
            logger.warning(f"Project KB path does not exist: {project_kb_path}")
            return projects

        for project_dir in project_kb_path.iterdir():
            logger.debug(f"Checking: {project_dir}")
            if not project_dir.is_dir():
                continue
            config_path = project_dir / "config.json"
            if not config_path.exists():
                logger.debug(f"No config.json in {project_dir}")
                continue
            logger.info(f"Found project: {project_dir.name}")

            # Load config for description
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                config = {}

            # Get document count from ChromaDB (0 if no collection)
            collection_name = f"{get_chromadb_collection()}_{project_dir.name}"
            # Sanitize collection name same way as _get_collection does
            collection_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in collection_name)[:63]
            doc_count = 0
            try:
                col = self._chroma_client.get_collection(collection_name)
                doc_count = col.count()
            except Exception:
                pass

            projects.append({
                "name": project_dir.name,
                "description": config.get("description", ""),
                "document_count": doc_count,
            })

        return projects

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a ChromaDB collection by name.

        Use this to clean up orphaned collections like 'rag_docs_default'.
        """
        if not self._initialized:
            await self.initialize()

        try:
            self._chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")
            return False

    async def delete_project(self, name: str) -> bool:
        """Delete a project's ChromaDB collection."""
        if not self._initialized:
            await self.initialize()

        col_name = f"{get_chromadb_collection()}_{name}"
        col_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in col_name)[:63]
        try:
            self._chroma_client.delete_collection(col_name)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Collection sync (export/import for VPS sync)
    # ------------------------------------------------------------------

    async def export_collection(self, project_name: str) -> Dict[str, Any]:
        """
        Export a project's ChromaDB collection as JSON.
        Exports documents and metadata (NOT embeddings - they will be regenerated on import).
        """
        if not self._initialized:
            await self.initialize()

        collection = self._get_collection(project_name)
        count = collection.count()

        if count == 0:
            return {
                "project": project_name,
                "collection_name": collection.name,
                "exported_at": datetime.now().isoformat() + "Z",
                "documents": [],
                "total_documents": 0,
            }

        # Get all documents from the collection
        results = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        documents = []
        if results.get("ids"):
            for i, doc_id in enumerate(results["ids"]):
                doc = {
                    "id": doc_id,
                    "content": results["documents"][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                }
                documents.append(doc)

        return {
            "project": project_name,
            "collection_name": collection.name,
            "exported_at": datetime.now().isoformat() + "Z",
            "documents": documents,
            "total_documents": len(documents),
        }

    async def import_collection(self, export_data: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        """
        Import documents into a project's ChromaDB collection from export JSON.
        Regenerates embeddings using the local embedding model.

        Args:
            export_data: JSON data from export_collection
            overwrite: If True, deletes existing collection first. If False, upserts.
        """
        if not self._initialized:
            await self.initialize()

        project_name = export_data.get("project")
        if not project_name:
            return {"error": "Missing project name in export data", "imported": 0}

        documents = export_data.get("documents", [])
        if not documents:
            return {"error": "No documents to import", "imported": 0}

        # Optionally delete existing collection first
        if overwrite:
            try:
                col_name = f"{get_chromadb_collection()}_{project_name}"
                col_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in col_name)[:63]
                self._chroma_client.delete_collection(col_name)
                logger.info(f"Deleted existing collection for overwrite: {col_name}")
            except Exception:
                pass  # Collection may not exist

        collection = self._get_collection(project_name)
        imported = 0

        # Process in batches for efficiency
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = []
            contents = []
            metadatas = []

            for doc in batch:
                doc_id = doc.get("id", self._doc_id(doc.get("content", ""), i))
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                if not content.strip():
                    continue

                ids.append(doc_id)
                contents.append(content)
                metadatas.append(metadata)

            if ids:
                # Generate embeddings for this batch
                embeddings = self.embed(contents)

                # Upsert into collection
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                )
                imported += len(ids)

        logger.info(f"Imported {imported} documents into collection '{collection.name}'")
        return {
            "project": project_name,
            "collection_name": collection.name,
            "imported": imported,
            "total_in_export": len(documents),
        }

    # ------------------------------------------------------------------
    # Project configuration
    # ------------------------------------------------------------------

    def _get_project_config_path(self, project_name: str) -> Path:
        """Get the path to a project's config.json file."""
        return Path(get_project_kb_path()) / project_name / "config.json"

    async def get_project_config(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Load a project's configuration from config.json."""
        if not project_name:
            return None

        config_path = self._get_project_config_path(project_name)
        if not config_path.exists():
            return None

        try:
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load project config for '{project_name}': {e}")
            return None

    async def save_project_config(self, project_name: str, config: Dict[str, Any]) -> bool:
        """Save a project's configuration to config.json."""
        if not project_name:
            return False

        import json
        from datetime import datetime

        # Ensure project directory exists
        project_dir = Path(get_project_kb_path()) / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Add/update timestamps
        config["name"] = project_name
        config["updated_at"] = datetime.utcnow().isoformat()
        if "created_at" not in config:
            config["created_at"] = config["updated_at"]

        config_path = self._get_project_config_path(project_name)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved config for project '{project_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to save project config for '{project_name}': {e}")
            return False

    async def create_project(self, name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new project knowledge base with optional configuration."""
        if not self._initialized:
            await self.initialize()

        collection = self._get_collection(name)
        # Create filesystem directory for project docs
        project_kb_path = get_project_kb_path()
        project_dir = os.path.join(project_kb_path, name)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "documents"), exist_ok=True)

        # Initialize config with defaults
        default_config = {
            "name": name,
            "description": "",
            "system_prompt": "",
            "instructions": "",
            "allowed_paths": [os.path.join(project_dir, "documents")],
        }
        if config:
            default_config.update(config)

        # Save config
        await self.save_project_config(name, default_config)

        return {
            "name": name,
            "collection": collection.name,
            "document_count": 0,
            "path": project_dir,
            "config": default_config,
        }

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            import pypdf
            text_parts = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pypdf not installed. Run: pip install pypdf")
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract PDF text from {file_path}: {e}")
            return ""

    async def index_project_files(self, project_name: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index files from a project's allowed_paths directories.
        Uses incremental indexing - only new/modified files are indexed.

        Args:
            project_name: Name of the project
            force_reindex: If True, re-index all files regardless of modification time

        Supports: .txt, .md, .json, .py, .js, .ts, .html, .css, .yaml, .yml, .pdf
        """
        logger.info(f"Starting index for project: {project_name}")
        if not self._initialized:
            await self.initialize()

        config = await self.get_project_config(project_name)
        if not config:
            return {"error": f"Project '{project_name}' not found", "indexed_chunks": 0}

        allowed_paths = config.get("allowed_paths", [])
        if not allowed_paths:
            return {"error": "No allowed_paths configured", "indexed_chunks": 0}

        # Load existing file index (tracks what's been indexed)
        indexed_files = config.get("indexed_files", {})  # {filepath: mtime_timestamp}
        logger.info(f"Found {len(indexed_files)} previously indexed files")

        # Supported file extensions
        text_extensions = {".txt", ".md", ".json", ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".rst", ".csv"}
        all_extensions = text_extensions | {".pdf"}

        documents = []
        files_indexed = []
        files_skipped = []
        new_indexed_files = {}

        for base_path in allowed_paths:
            base_path = Path(base_path)
            if not base_path.exists():
                logger.warning(f"Path does not exist: {base_path}")
                continue

            # Find all supported files recursively
            if base_path.is_file():
                files = [base_path]
            else:
                files = []
                for ext in all_extensions:
                    files.extend(base_path.rglob(f"*{ext}"))
            logger.info(f"Found {len(files)} files in {base_path}")

            for file_path in files:
                try:
                    file_str = str(file_path)
                    file_mtime = file_path.stat().st_mtime

                    # Check if file needs indexing
                    previously_indexed_mtime = indexed_files.get(file_str)
                    if not force_reindex and previously_indexed_mtime is not None:
                        if file_mtime <= previously_indexed_mtime:
                            # File unchanged, skip
                            files_skipped.append(file_str)
                            new_indexed_files[file_str] = previously_indexed_mtime
                            continue

                    # File is new or modified - index it
                    logger.info(f"Indexing: {file_path.name}")
                    if file_path.suffix.lower() == ".pdf":
                        content = self._extract_pdf_text(file_path)
                    else:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")

                    if content.strip():
                        documents.append({
                            "content": content,
                            "title": file_path.name,
                            "path": file_str,
                            "metadata": {
                                "source": "allowed_paths",
                                "project": project_name,
                                "file_type": file_path.suffix.lower(),
                            }
                        })
                        files_indexed.append(file_str)
                        new_indexed_files[file_str] = file_mtime
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        # Index the new/modified documents
        logger.info(f"Files to index: {len(documents)}, skipped: {len(files_skipped)}")
        indexed_count = 0
        if documents:
            indexed_count = await self.index_documents(documents, project=project_name)

        # Update the project config with the file index
        config["indexed_files"] = new_indexed_files
        await self.save_project_config(project_name, config)

        logger.info(f"Indexing complete: {indexed_count} chunks from {len(files_indexed)} files")
        return {
            "indexed_chunks": indexed_count,
            "files": files_indexed,
            "skipped": len(files_skipped),
            "total_files": len(new_indexed_files),
            "message": f"Indexed {indexed_count} chunks from {len(files_indexed)} new/modified files ({len(files_skipped)} unchanged)",
        }

    # ------------------------------------------------------------------
    # Global RAG configuration
    # ------------------------------------------------------------------

    async def get_global_config(self) -> Dict[str, Any]:
        """Load global RAG configuration from rag_config.json."""
        import json

        config_path = Path(get_rag_config_path())
        if not config_path.exists():
            # Return defaults
            return {
                "default_model": "claude-sonnet-4-5-20250929",
                "default_mode": "auto",
            }

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load global config: {e}")
            return {
                "default_model": "claude-sonnet-4-5-20250929",
                "default_mode": "auto",
            }

    async def save_global_config(self, config: Dict[str, Any]) -> bool:
        """Save global RAG configuration to rag_config.json."""
        import json

        config_path = Path(get_rag_config_path())
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info("Saved global RAG config")
            return True
        except Exception as e:
            logger.error(f"Failed to save global config: {e}")
            return False

    # ------------------------------------------------------------------
    # Chat mode (direct LLM, no RAG)
    # ------------------------------------------------------------------

    async def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: str = "local",
        project_config: Optional[Dict[str, Any]] = None,
        global_instructions: Optional[str] = None,
        query_classification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Direct chat without RAG context. Uses specified model.
        model: "local" for Ollama, or Claude model ID for Anthropic API.
        project_config: Optional project config with system_prompt and instructions.
        global_instructions: Optional global instructions to prepend to system prompt.
        query_classification: Optional classification result for flexible prompts.
        Returns {"text": str, "usage": dict, "model_used": str}.
        """
        if model == "local":
            # Try Ollama with Claude Haiku fallback
            result = await self._chat_ollama_with_fallback(
                query, conversation_history, project_config, global_instructions, query_classification
            )
            return result
        else:
            result = await self._chat_claude(query, conversation_history, model, project_config, global_instructions, query_classification)
            result["model_used"] = model
            return result

    async def _chat_ollama(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        project_config: Optional[Dict[str, Any]] = None,
        global_instructions: Optional[str] = None,
        query_classification: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Chat using local Ollama with optional project instructions. Returns None if Ollama unavailable."""
        # Determine if we should use flexible mode (for non-task queries)
        use_flexible = False
        if query_classification:
            from query_classifier import QueryIntent
            intent = query_classification.get("intent")
            if intent in [QueryIntent.META_QUESTION, QueryIntent.TECHNICAL_CODING, QueryIntent.CONVERSATION]:
                use_flexible = True

        # Build system prompt from project config
        system_prompt = "You are a helpful assistant."
        if project_config:
            if use_flexible and project_config.get("flexible_prompt"):
                system_prompt = project_config["flexible_prompt"]
            else:
                if project_config.get("system_prompt"):
                    system_prompt = project_config["system_prompt"]
                if project_config.get("instructions"):
                    system_prompt += f"\n\nInstructions: {project_config['instructions']}"

        # Add context modifier based on query classification
        if query_classification:
            from query_classifier import QueryClassifier
            context_modifier = QueryClassifier.get_context_modifier(query_classification)
            if context_modifier:
                system_prompt = f"{context_modifier}\n\n{system_prompt}"

        # Prepend global instructions if provided
        if global_instructions:
            system_prompt = f"{global_instructions}\n\n{system_prompt}"

        # Add current date to system prompt
        current_date = datetime.now().strftime("%B %d, %Y")
        system_prompt = f"Today's date is {current_date}.\n\n{system_prompt}"

        # Build conversation history section
        history_section = ""
        if conversation_history:
            recent_history = conversation_history[-6:]
            history_lines = []
            for msg in recent_history:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            if history_lines:
                history_section = "Previous conversation:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""{system_prompt}

{history_section}User: {query}

Assistant:"""

        return await self._call_ollama(prompt)

    async def _chat_ollama_with_fallback(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        project_config: Optional[Dict[str, Any]] = None,
        global_instructions: Optional[str] = None,
        query_classification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chat using Ollama with Claude Haiku fallback. Returns {"text": str, "usage": dict, "model_used": str}."""
        # Try Ollama first
        result = await self._chat_ollama(query, conversation_history, project_config, global_instructions, query_classification)
        if result is not None:
            return {"text": result, "usage": {}, "model_used": "ollama"}
        # Fall back to Claude Haiku
        logger.info("Ollama unavailable for chat, falling back to Claude Haiku")
        claude_result = await self._chat_claude(
            query, conversation_history, "claude-haiku-4-5-20251001",
            project_config, global_instructions, query_classification
        )
        claude_result["model_used"] = "claude-haiku-4-5-20251001"
        return claude_result

    async def _chat_claude(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: str = "claude-sonnet-4-5-20250929",
        project_config: Optional[Dict[str, Any]] = None,
        global_instructions: Optional[str] = None,
        query_classification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chat using Claude API with optional project instructions. Returns {"text": str, "usage": dict}."""
        api_key = get_anthropic_api_key()
        if not api_key or api_key.startswith("your_"):
            return {"text": "[Claude API key not configured]", "usage": {}}

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=120.0)

        # Determine if we should use flexible mode (for non-task queries)
        use_flexible = False
        if query_classification:
            from query_classifier import QueryIntent
            intent = query_classification.get("intent")
            if intent in [QueryIntent.META_QUESTION, QueryIntent.TECHNICAL_CODING, QueryIntent.CONVERSATION]:
                use_flexible = True

        # Build system prompt from project config
        system_prompt = None
        if project_config:
            parts = []
            if use_flexible and project_config.get("flexible_prompt"):
                parts.append(project_config["flexible_prompt"])
            else:
                if project_config.get("system_prompt"):
                    parts.append(project_config["system_prompt"])
                if project_config.get("instructions"):
                    parts.append(f"Instructions: {project_config['instructions']}")
            if parts:
                system_prompt = "\n\n".join(parts)

        # Add context modifier based on query classification
        if query_classification:
            from query_classifier import QueryClassifier
            context_modifier = QueryClassifier.get_context_modifier(query_classification)
            if context_modifier:
                if system_prompt:
                    system_prompt = f"{context_modifier}\n\n{system_prompt}"
                else:
                    system_prompt = context_modifier

        # Prepend global instructions if provided
        if global_instructions:
            if system_prompt:
                system_prompt = f"{global_instructions}\n\n{system_prompt}"
            else:
                system_prompt = global_instructions

        # Add current date to system prompt
        current_date = datetime.now().strftime("%B %d, %Y")
        if system_prompt:
            system_prompt = f"Today's date is {current_date}.\n\n{system_prompt}"
        else:
            system_prompt = f"Today's date is {current_date}."

        # Build messages list with conversation history
        messages = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        messages.append({"role": "user", "content": query})

        # Build request payload
        payload = {
            "model": model,
            "max_tokens": get_max_tokens(),
            "temperature": get_temperature(),
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = await self._http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
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
                }
            }
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Claude HTTP error {e.response.status_code}: {error_body}")
            return {"text": f"[Claude error {e.response.status_code}: {error_body}]", "usage": {}}
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return {"text": f"[Claude error: {e}]", "usage": {}}

    # ------------------------------------------------------------------
    # Chat history persistence (PostgreSQL with file fallback)
    # ------------------------------------------------------------------

    def _get_db_connection(self):
        """Get PostgreSQL connection if DATABASE_URL is configured."""
        import psycopg2
        db_url = get_database_url()
        if not db_url:
            return None
        try:
            return psycopg2.connect(db_url)
        except Exception as e:
            logger.warning(f"Failed to connect to database: {e}")
            return None

    def _get_chats_dir(self) -> Path:
        """Get and ensure the chats directory exists (file fallback)."""
        chats_dir = Path(get_chats_path()).resolve()
        chats_dir.mkdir(parents=True, exist_ok=True)
        return chats_dir

    async def list_chats(
        self,
        project: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List chat summaries, optionally filtered by project.
        Returns list sorted by updated_at (newest first).
        Uses PostgreSQL if configured, otherwise falls back to files.
        """
        conn = self._get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    if project:
                        cur.execute("""
                            SELECT id, name, project, created_at, updated_at,
                                   jsonb_array_length(messages) as message_count
                            FROM chats WHERE project = %s
                            ORDER BY updated_at DESC LIMIT %s
                        """, (project, limit))
                    else:
                        cur.execute("""
                            SELECT id, name, project, created_at, updated_at,
                                   jsonb_array_length(messages) as message_count
                            FROM chats ORDER BY updated_at DESC LIMIT %s
                        """, (limit,))
                    rows = cur.fetchall()
                    return [{
                        "id": r[0],
                        "name": r[1] or "Untitled",
                        "project": r[2],
                        "created_at": r[3].isoformat() + "Z" if r[3] else None,
                        "updated_at": r[4].isoformat() + "Z" if r[4] else None,
                        "message_count": r[5] or 0,
                    } for r in rows]
            except Exception as e:
                logger.error(f"Database error listing chats: {e}")
            finally:
                conn.close()

        # Fallback to file storage
        chats_dir = self._get_chats_dir()
        chats = []
        for chat_file in chats_dir.glob("*.json"):
            try:
                with open(chat_file, "r", encoding="utf-8") as f:
                    chat = json.load(f)
                if project and chat.get("project") != project:
                    continue
                chats.append({
                    "id": chat.get("id"),
                    "name": chat.get("name", "Untitled"),
                    "project": chat.get("project"),
                    "created_at": chat.get("created_at"),
                    "updated_at": chat.get("updated_at"),
                    "message_count": len(chat.get("messages", [])),
                })
            except Exception as e:
                logger.warning(f"Failed to read chat file {chat_file}: {e}")
        chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return chats[:limit]

    async def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Load a full chat by ID. Uses PostgreSQL if configured."""
        conn = self._get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, name, project, messages, created_at, updated_at
                        FROM chats WHERE id = %s
                    """, (chat_id,))
                    row = cur.fetchone()
                    if row:
                        return {
                            "id": row[0],
                            "name": row[1],
                            "project": row[2],
                            "messages": row[3] or [],
                            "created_at": row[4].isoformat() + "Z" if row[4] else None,
                            "updated_at": row[5].isoformat() + "Z" if row[5] else None,
                        }
            except Exception as e:
                logger.error(f"Database error getting chat {chat_id}: {e}")
            finally:
                conn.close()

        # Fallback to file storage
        chat_path = self._get_chats_dir() / f"{chat_id}.json"
        if not chat_path.exists():
            return None
        try:
            with open(chat_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read chat {chat_id}: {e}")
            return None

    def _generate_chat_title(self, content: str, max_length: int = 40) -> str:
        """Generate a short, readable chat title from user message."""
        import re

        # Remove file attachment references and newlines
        content = re.sub(r'\[Attached:.*?\]', '', content, flags=re.DOTALL).strip()
        content = re.sub(r'\n+', ' ', content)

        # Remove question marks and trailing punctuation early
        content = re.sub(r'\?+$', '', content).strip()

        # Common filler phrases to remove from start (applied repeatedly)
        filler_starts = [
            r'^(can you |could you |would you |please |i want to |i need to |i\'d like to )',
            r'^(help me |tell me about |tell me |show me |find me |get me |give me )',
            r'^(search for |look for |look up |find )',
            r'^(what is |what are |what\'s |when is |where is |where are )',
            r'^(what do i need to |what does |what should i |what time )',
            r'^(how do i |how can i |how do you |how to |why is |why are )',
            r'^(i want to know |i need to know |i\'d like to know |i want |i need )',
            r'^(which time of the year |which time |which is the best |which )',
            r'^(a |an |the )',
        ]
        for _ in range(4):  # Apply multiple times to catch nested patterns
            for pattern in filler_starts:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE).strip()

        # Handle "compare X to Y" -> "X vs Y"
        content = re.sub(r'^compare\s+(.+?)\s+to\s+', r'\1 vs ', content, flags=re.IGNORECASE)
        content = re.sub(r'^compare\s+', '', content, flags=re.IGNORECASE)

        # Remove trailing filler words
        trailing_fillers = [
            r'\s+(today|currently|right now|now|please|thanks|thank you)$',
            r'\s+in (the )?USD$',
        ]
        for pattern in trailing_fillers:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Abbreviations for common words
        abbreviations = {
            r'\btemperature\b': 'temp',
            r'\btemperatures\b': 'temps',
            r'\bbetween\b': 'vs',
            r'\binformation\b': 'info',
            r'\bdocument\b': 'doc',
            r'\bdocuments\b': 'docs',
            r'\bapplication\b': 'app',
            r'\bconfiguration\b': 'config',
            r'\bimplementation\b': 'impl',
            r'\bdescription\b': 'desc',
            r'\bseawater\b': 'sea water',
            r'\bapartment\b': 'apt',
            r'\bapartments\b': 'apts',
            r'\brental listings\b': 'rentals',
            r'\bcurrent rentals?\b': 'rentals',
            r'\blong.?term\b': 'long-term',
            r'\bBarcelona\b': 'BCN',
            r'\bminimum\b': 'min',
            r'\bmaximum\b': 'max',
            r'\bsalary\b': 'salary',
        }
        for pattern, replacement in abbreviations.items():
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

        # Remove "the" after vs or at start
        content = re.sub(r'\bvs the\b', 'vs', content, flags=re.IGNORECASE)
        content = re.sub(r'^the ', '', content, flags=re.IGNORECASE)

        # Clean up multiple spaces
        content = re.sub(r'\s+', ' ', content).strip()

        # Capitalize first letter
        if content:
            content = content[0].upper() + content[1:]

        # Truncate if still too long
        if len(content) > max_length:
            truncated = content[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.6:
                truncated = truncated[:last_space]
            content = truncated.rstrip('.,!? ') + '...'

        return content if content else "New Chat"

    async def save_chat(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a chat. Uses PostgreSQL if configured.
        Chat dict should have: id (optional), name, project, messages.
        Timestamps are auto-managed.
        Returns the saved chat with generated ID.
        """
        import secrets
        from datetime import datetime

        chat_id = chat.get("id")
        if not chat_id:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            chat_id = f"chat_{timestamp}_{secrets.token_hex(4)}"
            chat["id"] = chat_id

        # Manage timestamps
        now = datetime.utcnow().isoformat() + "Z"
        if "created_at" not in chat:
            chat["created_at"] = now
        chat["updated_at"] = now

        # Auto-generate name from first user message if not set
        if not chat.get("name") or chat.get("name") == "New Chat":
            messages = chat.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    chat["name"] = self._generate_chat_title(content)
                    break
            if not chat.get("name"):
                chat["name"] = "New Chat"

        # Try PostgreSQL first
        conn = self._get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO chats (id, name, project, messages, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            project = EXCLUDED.project,
                            messages = EXCLUDED.messages,
                            updated_at = NOW()
                    """, (
                        chat_id,
                        chat.get("name"),
                        chat.get("project"),
                        json.dumps(chat.get("messages", [])),
                    ))
                conn.commit()
                logger.info(f"Saved chat to database: {chat_id}")
                return chat
            except Exception as e:
                logger.error(f"Database error saving chat {chat_id}: {e}")
                conn.rollback()
            finally:
                conn.close()

        # Fallback to file storage
        chat_path = self._get_chats_dir() / f"{chat_id}.json"
        try:
            with open(chat_path, "w", encoding="utf-8") as f:
                json.dump(chat, f, indent=2)
            logger.info(f"Saved chat to file: {chat_id}")
            return chat
        except Exception as e:
            logger.error(f"Failed to save chat {chat_id}: {e}")
            raise

    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat by ID. Uses PostgreSQL if configured."""
        conn = self._get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
                if deleted:
                    logger.info(f"Deleted chat from database: {chat_id}")
                return deleted
            except Exception as e:
                logger.error(f"Database error deleting chat {chat_id}: {e}")
                conn.rollback()
            finally:
                conn.close()

        # Fallback to file storage
        chat_path = self._get_chats_dir() / f"{chat_id}.json"
        if not chat_path.exists():
            return False
        try:
            chat_path.unlink()
            logger.info(f"Deleted chat file: {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chat {chat_id}: {e}")
            return False

    async def rename_chat(self, chat_id: str, new_name: str) -> bool:
        """Rename a chat. Uses PostgreSQL if configured."""
        conn = self._get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE chats SET name = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (new_name, chat_id))
                    updated = cur.rowcount > 0
                conn.commit()
                return updated
            except Exception as e:
                logger.error(f"Database error renaming chat {chat_id}: {e}")
                conn.rollback()
            finally:
                conn.close()

        # Fallback to file-based rename
        chat = await self.get_chat(chat_id)
        if not chat:
            return False
        chat["name"] = new_name
        await self.save_chat(chat)
        return True