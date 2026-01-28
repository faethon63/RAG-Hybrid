"""
RAG Core Module
ChromaDB vector store, sentence-transformer embeddings, Ollama LLM calls,
document indexing, and project management.
"""

import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

# --- Configuration ---

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "chromadb"))
CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "rag_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
PROJECT_KB_PATH = os.getenv("PROJECT_KB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "project-kb"))

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
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to ChromaDB (persistent on disk)
        db_path = str(Path(CHROMADB_PATH).resolve())
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
            resp = await self._http_client.get(f"{OLLAMA_HOST}/api/tags")
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
        name = f"{CHROMADB_COLLECTION}_{project}" if project else CHROMADB_COLLECTION
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
        top_k: int = TOP_K,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB for documents similar to the query.

        Returns list of dicts with keys: content, score, metadata.
        """
        if not self._initialized:
            await self.initialize()

        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

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

    async def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using Ollama, grounded in the provided context.
        """
        prompt = f"""Answer the following question based on the provided context.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

        return await self._call_ollama(prompt)

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama's generate API."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=120.0)

        try:
            resp = await self._http_client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": MAX_TOKENS,
                        "temperature": TEMPERATURE,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return f"[Ollama error: {e.response.status_code}]"
        except httpx.ConnectError:
            return "[Ollama is not running. Start it with: ollama serve]"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[Ollama error: {e}]"

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    async def list_projects(self) -> List[Dict[str, Any]]:
        """List all project knowledge bases."""
        if not self._initialized:
            await self.initialize()

        projects = []
        prefix = f"{CHROMADB_COLLECTION}_"
        for col in self._chroma_client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            if name.startswith(prefix):
                project_name = name[len(prefix):]
                collection = self._chroma_client.get_collection(name)
                projects.append({
                    "name": project_name,
                    "collection": name,
                    "document_count": collection.count(),
                })
            elif name == CHROMADB_COLLECTION:
                collection = self._chroma_client.get_collection(name)
                projects.append({
                    "name": "default",
                    "collection": name,
                    "document_count": collection.count(),
                })

        return projects

    async def create_project(self, name: str) -> Dict[str, Any]:
        """Create a new project knowledge base."""
        if not self._initialized:
            await self.initialize()

        collection = self._get_collection(name)
        # Also create filesystem directory for project docs
        project_dir = os.path.join(PROJECT_KB_PATH, name)
        os.makedirs(project_dir, exist_ok=True)

        return {
            "name": name,
            "collection": collection.name,
            "document_count": 0,
            "path": project_dir,
        }

    async def delete_project(self, name: str) -> bool:
        """Delete a project's ChromaDB collection."""
        if not self._initialized:
            await self.initialize()

        col_name = f"{CHROMADB_COLLECTION}_{name}"
        col_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in col_name)[:63]
        try:
            self._chroma_client.delete_collection(col_name)
            return True
        except Exception:
            return False
