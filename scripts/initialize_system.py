"""
First-Time System Initialization
Creates ChromaDB collections, tests connections, verifies configuration.

Usage:
    python scripts/initialize_system.py
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


async def main():
    print("=" * 60)
    print("RAG-Hybrid System Initialization")
    print("=" * 60)
    print()

    results = {}

    # --- 1. Check environment ---
    print("[1/6] Checking environment variables...")
    env_keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY", ""),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
        "JWT_SECRET": os.getenv("JWT_SECRET", ""),
        "OLLAMA_HOST": os.getenv("OLLAMA_HOST", ""),
    }
    for key, val in env_keys.items():
        configured = bool(val) and not val.startswith("your_") and val != "change_me_to_random_hex_string"
        status = "OK" if configured else "NOT SET"
        print(f"  {key}: {status}")
        results[key] = configured
    print()

    # --- 2. Initialize RAG core ---
    print("[2/6] Initializing RAG core (ChromaDB + embeddings)...")
    try:
        from rag_core import RAGCore
        rag = RAGCore()
        await rag.initialize()
        print("  ChromaDB: OK")
        print("  Embeddings: OK")
        results["chromadb"] = True
        results["embeddings"] = True
    except Exception as e:
        print(f"  ERROR: {e}")
        results["chromadb"] = False
        results["embeddings"] = False
        rag = None
    print()

    # --- 3. Check Ollama ---
    print("[3/6] Checking Ollama...")
    if rag:
        ollama_ok = await rag.check_ollama()
        print(f"  Ollama: {'OK' if ollama_ok else 'NOT RUNNING'}")
        if not ollama_ok:
            print("  Hint: Start Ollama with 'ollama serve' or launch the Ollama app")
            print(f"  Hint: Pull a model with 'ollama pull {os.getenv('OLLAMA_MODEL', 'qwen3:8b')}'")
        results["ollama"] = ollama_ok
    else:
        print("  Skipped (RAG core failed)")
        results["ollama"] = False
    print()

    # --- 4. Check Claude API ---
    print("[4/6] Checking Claude API...")
    try:
        from search_integrations import ClaudeSearch
        claude = ClaudeSearch()
        claude_ok = await claude.is_healthy()
        print(f"  Claude API: {'OK' if claude_ok else 'NOT AVAILABLE'}")
        results["claude"] = claude_ok
    except Exception as e:
        print(f"  ERROR: {e}")
        results["claude"] = False
    print()

    # --- 5. Check Perplexity API ---
    print("[5/6] Checking Perplexity API...")
    try:
        from search_integrations import PerplexitySearch
        pplx = PerplexitySearch()
        pplx_ok = await pplx.is_healthy()
        print(f"  Perplexity API: {'OK' if pplx_ok else 'NOT AVAILABLE'}")
        if pplx_ok:
            print("  (Free tier - rate limits apply)")
        results["perplexity"] = pplx_ok
    except Exception as e:
        print(f"  ERROR: {e}")
        results["perplexity"] = False
    print()

    # --- 6. Create default collections ---
    print("[6/6] Creating default collections...")
    if rag:
        try:
            await rag.create_project("default")
            print("  Collection 'default': OK")

            # Create data directories
            data_dirs = [
                os.path.join(os.path.dirname(__file__), "..", "data", "documents"),
                os.path.join(os.path.dirname(__file__), "..", "data", "project-kb"),
                os.path.join(os.path.dirname(__file__), "..", "data", "cache"),
                os.path.join(os.path.dirname(__file__), "..", "logs"),
            ]
            for d in data_dirs:
                os.makedirs(d, exist_ok=True)
            print("  Data directories: OK")
            results["collections"] = True
        except Exception as e:
            print(f"  ERROR: {e}")
            results["collections"] = False
    else:
        print("  Skipped (RAG core failed)")
        results["collections"] = False
    print()

    # --- Summary ---
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    ok_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  {ok_count}/{total} checks passed")
    print()

    critical = ["chromadb", "embeddings"]
    critical_ok = all(results.get(k, False) for k in critical)
    if critical_ok:
        print("  System is ready for local RAG queries.")
        if results.get("ollama"):
            print("  Ollama is running - local LLM generation available.")
        else:
            print("  WARNING: Ollama not running - local answers won't generate.")
        if results.get("claude") or results.get("perplexity"):
            print("  Web search APIs are configured.")
        else:
            print("  NOTE: No web APIs configured - only local mode available.")
    else:
        print("  CRITICAL: ChromaDB or embeddings failed.")
        print("  Fix these before running the system.")

    print()
    print("Next steps:")
    print("  1. Fix any NOT SET / NOT RUNNING items above")
    print("  2. Index documents: python scripts/index_documents.py --dir data/documents")
    print("  3. Start backend:   cd backend && python main.py")
    print("  4. Start frontend:  cd frontend && streamlit run app.py")

    if rag:
        await rag.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
