"""
Bulk Document Indexer
Loads documents from a directory into ChromaDB via the RAG core.

Usage:
    python scripts/index_documents.py [--dir PATH] [--project NAME] [--ext txt,md,pdf]
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from rag_core import RAGCore


SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".html", ".rst"}


def read_file(path: Path) -> str:
    """Read a file's text content."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  [skip] Cannot read {path}: {e}")
        return ""


def collect_documents(directory: str, extensions: set) -> list:
    """Recursively collect documents from a directory."""
    docs = []
    root = Path(directory).resolve()

    if not root.is_dir():
        print(f"Error: {directory} is not a directory")
        return docs

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        # Skip hidden/system files
        if any(part.startswith(".") for part in path.parts):
            continue

        content = read_file(path)
        if content.strip():
            docs.append({
                "content": content,
                "title": path.name,
                "path": str(path),
                "metadata": {
                    "extension": path.suffix,
                    "size_bytes": path.stat().st_size,
                },
            })

    return docs


async def main():
    parser = argparse.ArgumentParser(description="Bulk-index documents into ChromaDB")
    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "documents"),
        help="Directory to index (default: data/documents)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name (creates separate collection)",
    )
    parser.add_argument(
        "--ext",
        default=",".join(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
        help="Comma-separated file extensions to include",
    )
    args = parser.parse_args()

    extensions = {"." + e.strip().lstrip(".") for e in args.ext.split(",")}

    print(f"Scanning: {os.path.abspath(args.dir)}")
    print(f"Extensions: {extensions}")
    print(f"Project: {args.project or '(default)'}")
    print()

    docs = collect_documents(args.dir, extensions)
    print(f"Found {len(docs)} documents")

    if not docs:
        print("Nothing to index.")
        return

    rag = RAGCore()
    await rag.initialize()

    print("Indexing...")
    count = await rag.index_documents(docs, project=args.project)
    print(f"Done. Indexed {count} chunks.")

    await rag.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
