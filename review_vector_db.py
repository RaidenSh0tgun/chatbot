# scripts/review_vector_db.py
"""
Interactive vector DB review tool (Chroma + OllamaEmbeddings).

What it does:
- Loads your existing Chroma DB from ./chroma_db
- Uses the same embedding model you used in main.py (mxbai-embed-large)
- Lets you type a query in the terminal
- Prints top-k retrieved chunks with metadata and content preview
- Optionally exports the latest results to a JSON file for later review

Run:
  python scripts/review_vector_db.py

Notes:
- Make sure Ollama is running (if your embeddings require it).
- Update persist_directory / collection_name / embedding model if your setup differs.
"""

import os
import json
import textwrap
from datetime import datetime

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ---------- CONFIG (match main.py) ----------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rutgers_corpus"
EMBED_MODEL = "mxbai-embed-large"

DEFAULT_K = 10
PREVIEW_CHARS = 750  # how much content to show per chunk


def fmt(text: str, width: int = 100) -> str:
    """Pretty-wrap text for terminal printing."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False))


def doc_to_dict(doc, rank: int) -> dict:
    md = doc.metadata or {}
    content = doc.page_content or ""
    return {
        "rank": rank,
        "source_url": md.get("source_url"),
        "source_file": md.get("source_file"),
        "metadata": md,  # keep full metadata
        "content": content,
    }


def print_doc(doc, rank: int) -> None:
    md = doc.metadata or {}
    content = doc.page_content or ""
    preview = content[:PREVIEW_CHARS]

    print("=" * 110)
    print(f"Rank: {rank}")
    print(f"source_url : {md.get('source_url')}")
    print(f"source_file: {md.get('source_file')}")
    # Print a few extra metadata fields if they exist
    for key in ("title", "chunk_id", "page", "section", "doc_id"):
        if key in md:
            print(f"{key:10}: {md.get(key)}")
    print("-" * 110)
    print(fmt(preview, width=110))
    if len(content) > PREVIEW_CHARS:
        print("\n... [truncated]")
    print()


def export_results(query: str, docs, out_dir: str = "local_review") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"vector_review_{ts}.json")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "k": len(docs),
        "results": [doc_to_dict(d, i) for i, d in enumerate(docs)],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def main():
    print("Connecting to Chroma vector DB...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # You can use db.as_retriever(...) OR similarity_search. similarity_search is straightforward.
    print(f"✅ Loaded collection '{COLLECTION_NAME}' from {PERSIST_DIR}\n")

    k = DEFAULT_K

    while True:
        q = input(f"\nEnter query (or ':q' to quit, ':k 12' to set k={k}, ':help'): ").strip()
        if not q:
            continue
        if q in (":q", ":quit", "quit", "exit"):
            break
        if q.startswith(":k"):
            parts = q.split()
            if len(parts) == 2 and parts[1].isdigit():
                k = int(parts[1])
                print(f"Set k = {k}")
            else:
                print("Usage: :k 12")
            continue
        if q == ":help":
            print(
                "\nCommands:\n"
                "  :q / :quit     Quit\n"
                "  :k N           Set top-k results\n"
                "  :help          Show this help\n"
                "\nTip: queries should be concise, high-signal keywords.\n"
            )
            continue

        try:
            docs = db.similarity_search(q, k=k)
        except Exception as e:
            print(f"\n❌ Retrieval error: {repr(e)}")
            continue

        if not docs:
            print("\n(No results)")
            continue

        print(f"\nTop {len(docs)} results for: {q}\n")
        for i, d in enumerate(docs):
            print_doc(d, i)

        cmd = input("Export these results to JSON? (y/N): ").strip().lower()
        if cmd == "y":
            out_path = export_results(q, docs)
            print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
