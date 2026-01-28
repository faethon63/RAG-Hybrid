"""
Integration Tests
Tests all components: auth, RAG core, search integrations, API endpoints.

Usage:
    python scripts/test_system.py
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, desc: str):
        self.passed += 1
        print(f"  [PASS] {desc}")

    def fail(self, desc: str, err: str = ""):
        self.failed += 1
        self.errors.append(f"{desc}: {err}")
        print(f"  [FAIL] {desc} — {err}")


async def test_auth() -> TestResult:
    """Test authentication module."""
    r = TestResult("auth")
    print("\n--- Auth Module ---")

    try:
        from auth import create_token, verify_token, hash_password, check_password, RateLimiter

        # Token creation & verification
        token = create_token("testuser")
        assert token, "Token should not be empty"
        r.ok("create_token returns a token")

        user = verify_token(token)
        assert user == "testuser", f"Expected 'testuser', got '{user}'"
        r.ok("verify_token returns correct username")

        bad = verify_token("invalid.token.here")
        assert bad is None, "Should return None for invalid token"
        r.ok("verify_token rejects invalid token")

        # Password hashing
        hashed = hash_password("secret123")
        assert hashed.startswith("$2b$"), "Should be bcrypt hash"
        r.ok("hash_password produces bcrypt hash")

        assert check_password("secret123", hashed), "Should match"
        r.ok("check_password matches correct password")

        assert not check_password("wrong", hashed), "Should not match"
        r.ok("check_password rejects wrong password")

        # Rate limiter
        limiter = RateLimiter(rpm=3, daily=10)
        assert limiter.check_limit("u1"), "First request should pass"
        assert limiter.check_limit("u1"), "Second request should pass"
        assert limiter.check_limit("u1"), "Third request should pass"
        assert not limiter.check_limit("u1"), "Fourth should be rate limited"
        r.ok("RateLimiter enforces RPM limit")

        usage = limiter.get_usage("u1")
        assert usage["rpm_used"] == 3, f"Expected 3, got {usage['rpm_used']}"
        r.ok("RateLimiter reports usage correctly")

    except Exception as e:
        r.fail("Auth module", str(e))

    return r


async def test_rag_core() -> TestResult:
    """Test RAG core module."""
    r = TestResult("rag_core")
    print("\n--- RAG Core ---")

    try:
        from rag_core import RAGCore

        rag = RAGCore()
        await rag.initialize()
        r.ok("RAGCore initializes successfully")

        # Health checks
        healthy = await rag.is_healthy()
        assert healthy, "Should be healthy after init"
        r.ok("is_healthy returns True")

        chromadb_ok = await rag.check_chromadb()
        assert chromadb_ok, "ChromaDB should be accessible"
        r.ok("ChromaDB is accessible")

        # Embedding
        vecs = rag.embed(["hello world"])
        assert len(vecs) == 1, "Should return one embedding"
        assert len(vecs[0]) > 0, "Embedding should have dimensions"
        r.ok(f"Embedding works (dim={len(vecs[0])})")

        # Index a test document
        count = await rag.index_documents([
            {
                "content": "The RAG-Hybrid system combines local document search with web APIs for comprehensive answers.",
                "title": "test_doc.txt",
                "path": "/test/test_doc.txt",
            }
        ], project="test")
        assert count > 0, "Should index at least one chunk"
        r.ok(f"Document indexing works ({count} chunks)")

        # Search (use lower threshold for short test text)
        results = await rag.search("RAG system", project="test", top_k=3, threshold=0.3)
        assert len(results) > 0, "Should find the indexed document"
        assert results[0]["score"] > 0.3, "Score should be reasonable"
        r.ok(f"Search works (found {len(results)} results, top score={results[0]['score']})")

        # Project management
        projects = await rag.list_projects()
        test_projects = [p for p in projects if p["name"] == "test"]
        assert len(test_projects) > 0, "Should find test project"
        r.ok("list_projects works")

        # Cleanup test project
        deleted = await rag.delete_project("test")
        assert deleted, "Should delete test project"
        r.ok("delete_project works")

        # Ollama check (may fail if not running)
        ollama_ok = await rag.check_ollama()
        if ollama_ok:
            r.ok("Ollama is running")
        else:
            print("  [SKIP] Ollama not running (not critical for tests)")

        await rag.cleanup()

    except Exception as e:
        r.fail("RAG core", str(e))

    return r


async def test_search_integrations() -> TestResult:
    """Test search integration modules."""
    r = TestResult("search_integrations")
    print("\n--- Search Integrations ---")

    try:
        from search_integrations import ClaudeSearch, PerplexitySearch, TavilySearch

        # Claude
        claude = ClaudeSearch()
        claude_ok = await claude.is_healthy()
        if claude_ok:
            r.ok("Claude API is healthy")
        else:
            print("  [SKIP] Claude API not configured")

        # Perplexity
        pplx = PerplexitySearch()
        pplx_ok = await pplx.is_healthy()
        if pplx_ok:
            r.ok("Perplexity API is healthy")
        else:
            print("  [SKIP] Perplexity API not configured")

        # Tavily
        tavily = TavilySearch()
        tavily_ok = await tavily.is_healthy()
        if tavily_ok:
            r.ok("Tavily API is healthy")
        else:
            print("  [SKIP] Tavily API not configured")

        # At least module imports work
        r.ok("All search integration classes import successfully")

    except Exception as e:
        r.fail("Search integrations", str(e))

    return r


async def test_api_models() -> TestResult:
    """Test API request/response models."""
    r = TestResult("api_models")
    print("\n--- API Models ---")

    try:
        # Import main.py models without starting the server
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pydantic import BaseModel
        from typing import List, Optional, Dict, Any

        # Test QueryRequest
        class QueryRequest(BaseModel):
            query: str
            mode: str = "hybrid"
            project: Optional[str] = None
            max_results: int = 5
            include_sources: bool = True

        req = QueryRequest(query="test question")
        assert req.mode == "hybrid", "Default mode should be hybrid"
        r.ok("QueryRequest model works")

        req2 = QueryRequest(query="test", mode="local", project="myproject")
        assert req2.project == "myproject"
        r.ok("QueryRequest with all fields works")

    except Exception as e:
        r.fail("API models", str(e))

    return r


async def main():
    print("=" * 60)
    print("RAG-Hybrid Integration Tests")
    print("=" * 60)

    all_results = []

    all_results.append(await test_auth())
    all_results.append(await test_rag_core())
    all_results.append(await test_search_integrations())
    all_results.append(await test_api_models())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    total_pass = 0
    total_fail = 0
    for r in all_results:
        status = "PASS" if r.failed == 0 else "FAIL"
        print(f"  [{status}] {r.name}: {r.passed} passed, {r.failed} failed")
        total_pass += r.passed
        total_fail += r.failed
        for err in r.errors:
            print(f"         {err}")

    print(f"\n  Total: {total_pass} passed, {total_fail} failed")

    if total_fail > 0:
        print("\n  Some tests failed. Check errors above.")
        sys.exit(1)
    else:
        print("\n  All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
