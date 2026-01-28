"""
RAG-Hybrid System - Main Backend API
FastAPI server that orchestrates all RAG components
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Ensure backend directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load environment (try config/.env then project-root .env)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from rag_core import RAGCore
from search_integrations import ClaudeSearch, PerplexitySearch, TavilySearch
from auth import verify_token, authenticate_user, RateLimiter

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Initialize FastAPI
app = FastAPI(
    title="TogetherOS RAG System",
    description="Hybrid RAG system with Claude, Perplexity, and local knowledge",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
rate_limiter = RateLimiter()

# Initialize RAG components
rag_core = RAGCore()
claude_search = ClaudeSearch()
perplexity_search = PerplexitySearch()
tavily_search = TavilySearch()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # "local", "web", "research", "hybrid"
    project: Optional[str] = None
    max_results: int = 5
    include_sources: bool = True
    

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
    """Health check endpoint"""
    services = {
        "local_rag": await rag_core.is_healthy(),
        "claude_api": await claude_search.is_healthy(),
        "perplexity_api": await perplexity_search.is_healthy(),
        "ollama": await rag_core.check_ollama(),
        "chromadb": await rag_core.check_chromadb()
    }
    
    return HealthResponse(
        status="healthy" if all(services.values()) else "degraded",
        services=services,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Main query endpoint - orchestrates search across all sources
    
    Modes:
    - local: Search only local documents (fast, private)
    - web: Web search only (Claude/Perplexity)
    - research: Deep research with Perplexity
    - hybrid: Combine local + web (default)
    """
    start_time = datetime.utcnow()
    
    # Verify authentication
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Rate limiting
    if not rate_limiter.check_limit(user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    try:
        # Route query based on mode
        if request.mode == "local":
            result = await query_local(request)
        elif request.mode == "web":
            result = await query_web(request)
        elif request.mode == "research":
            result = await query_research(request)
        else:  # hybrid
            result = await query_hybrid(request)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            mode=request.mode,
            sources=result["sources"],
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            confidence=result.get("confidence")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


async def query_local(request: QueryRequest) -> Dict[str, Any]:
    """Search local documents only"""
    results = await rag_core.search(
        query=request.query,
        project=request.project,
        top_k=request.max_results
    )
    
    # Generate answer with Ollama
    context = "\n\n".join([r["content"] for r in results])
    answer = await rag_core.generate_answer(request.query, context)
    
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
        "answer": answer,
        "sources": sources,
        "confidence": calculate_confidence(results)
    }


async def query_web(request: QueryRequest) -> Dict[str, Any]:
    """Search web only (Claude or Perplexity)"""
    # Try Claude first (included in Pro subscription)
    try:
        result = await claude_search.search(request.query)
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
            ]
        }
    except Exception as e:
        # Fallback to Perplexity
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
            ]
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
                snippet=s["snippet"]
            )
            for s in result["citations"]
        ]
    }


async def query_hybrid(request: QueryRequest) -> Dict[str, Any]:
    """Hybrid search: local + web"""
    # Run searches in parallel
    local_task = query_local(request)
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
    combined_answer = await combine_answers(
        local_result["answer"],
        web_result["answer"],
        request.query
    )
    
    combined_sources = local_result["sources"] + web_result["sources"]
    
    return {
        "answer": combined_answer,
        "sources": combined_sources,
        "confidence": (local_result.get("confidence", 0.5) + 0.5) / 2
    }


async def combine_answers(local_answer: str, web_answer: str, query: str) -> str:
    """Intelligently combine local and web answers"""
    # Use Claude to synthesize
    prompt = f"""Given the query: "{query}"

Local knowledge says:
{local_answer}

Web search says:
{web_answer}

Provide a comprehensive answer that synthesizes both sources, noting any agreements or contradictions."""

    return await claude_search.generate(prompt)


def calculate_confidence(results: List[Dict]) -> float:
    """Calculate confidence score from search results"""
    if not results:
        return 0.0
    avg_score = sum(r["score"] for r in results) / len(results)
    return min(avg_score, 1.0)


@app.post("/api/v1/index")
async def index_documents(
    request: IndexRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Index new documents into the RAG system"""
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
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
async def list_projects(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """List available project knowledge bases"""
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    projects = await rag_core.list_projects()
    
    return {
        "projects": projects,
        "count": len(projects),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Starting RAG-Hybrid System...")
    
    # Initialize RAG core
    await rag_core.initialize()
    
    # Check all services
    health = await health_check()
    print(f"System health: {health.status}")
    print(f"  Services: {health.services}")

    print("RAG System ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down RAG System...")
    await rag_core.cleanup()


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
