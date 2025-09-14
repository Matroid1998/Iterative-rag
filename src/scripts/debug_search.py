#!/usr/bin/env python3
"""
Quick retrieval debug: query the Chroma index directly and print top hits.

Usage examples (from repo root):
  python3 -m src.scripts.debug_search --query "Arsenic (As) is a toxic and highly mobile contaminant, which forms dangerous water." --k 8

From src/:  python3 scripts/debug_search.py --query "..." --k 8
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from protocols.embedding_config import EmbedderConfig
from repo.embeddings.embeddeing_models import HFEmbedder
from repo.index.chroma_index import ChromaTextIndex
from repo.retrievers.multi_retriever import MultiCollectionRetriever
import chromadb

# Default query (edit here to change without CLI args)
DEFAULT_QUERY = (
    "Arsenic (As) is a toxic and highly mobile contaminant, which forms dangerous water."
)


def _build_multi_retriever(persist_path: str, prefer: str | None = None) -> tuple[MultiCollectionRetriever, str, int, int]:
    """Construct a multi-collection retriever over all collections in the store.
    Returns (retriever, chosen_collection_for_count, total_chunks_in_chosen).
    """
    client = chromadb.PersistentClient(path=persist_path)
    cols = client.list_collections()
    names = [c.name for c in cols]
    if not names:
        raise RuntimeError(f"No collections found at {persist_path}")
    embedder = HFEmbedder(EmbedderConfig(device="cpu"))
    indexes = []
    chosen = None
    chosen_count = 0
    total_count = 0
    for name in names:
        idx = ChromaTextIndex(
            persist_path=persist_path,
            collection_name=name,
            embedder=embedder,
            distance="cosine",
        )
        cnt = idx.count()
        total_count += int(cnt or 0)
        indexes.append((idx, name))
        if prefer and name == prefer:
            chosen = name
            chosen_count = cnt
    if chosen is None:
        # pick the largest
        sizes = [(ChromaTextIndex(persist_path, n, embedder), n) for n in names]
        sizes = [(idx.count(), n) for idx, n in sizes]
        sizes.sort(reverse=True)
        chosen_count, chosen = (sizes[0] if sizes else (0, names[0]))
    retriever = MultiCollectionRetriever(indexes, distance_space="cosine", oversample_dense=2)
    return retriever, chosen, chosen_count, total_count


def _pick_non_empty_collection(persist_path: str) -> str | None:
    """Scan Chroma store and return the name of a non-empty collection, if any.
    Prefer chem_corpus or chemrxiv_graph when present.
    """
    try:
        client = chromadb.PersistentClient(path=persist_path)
        cols = client.list_collections()
        # Build (name, count) tuples; count() requires attaching an embedding fn, so use try/except
        scored = []
        for c in cols:
            try:
                cnt = c.count()
            except Exception:
                cnt = 0
            scored.append((c.name, cnt))
        if not scored:
            return None
        # Prefer known names with data
        prefs = ["chem_corpus", "chemrxiv_graph"]
        for p in prefs:
            for n, cnt in scored:
                if n == p and cnt > 0:
                    return n
        # Else pick the largest
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored[0][1] > 0 else None
    except Exception:
        return None


def _print_hits(hits: List[Dict[str, Any]], *, max_text: int = 320) -> None:
    if not hits:
        print("No hits.")
        return
    for i, h in enumerate(hits, 1):
        text = (h.get("text") or "").strip().replace("\n", " ")
        if len(text) > max_text:
            text = text[:max_text] + " ..."
        meta = h.get("metadata") or {}
        title = meta.get("title")
        src = meta.get("source")
        coll = meta.get("_collection")
        print(f"[{i}] id={h.get('id')} score={h.get('score'):.4f} src={src or '-'} collection={coll or '-'}")
        if title:
            print(f"    title: {title}")
        print(f"    text:  {text}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query Chroma index directly and print hits.")
    ap.add_argument("--query", required=False, default=None, help="Query text (optional)")
    ap.add_argument("--k", type=int, default=8, help="Top-K results to show")
    # Default to src/chroma_store so it works when running from src/
    default_persist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_store"))
    ap.add_argument(
        "--persist",
        default=default_persist,
        help="Chroma persist directory (default: src/chroma_store)",
    )
    ap.add_argument(
        "--collection",
        default="chem_corpus",
        help="Chroma collection name (default: chem_corpus)",
    )
    args = ap.parse_args()

    # If the provided path doesn't exist, try a couple of sensible fallbacks
    persist_path = args.persist
    if not os.path.exists(persist_path):
        alt1 = os.path.abspath(os.path.join(os.getcwd(), "chroma_store"))
        alt2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chroma_store"))
        for alt in (alt1, alt2):
            if os.path.exists(alt):
                print(f"[info] Persist path not found: {persist_path}. Using fallback: {alt}")
                persist_path = alt
                break

    retriever, col_name, n, total = _build_multi_retriever(persist_path, prefer=args.collection)
    # Report how many collections exist and their names
    try:
        client = chromadb.PersistentClient(path=persist_path)
        names = [c.name for c in client.list_collections()]
        print(f"Collections at '{persist_path}': {len(names)} -> {', '.join(names)}")
    except Exception:
        print(f"Collections at '{persist_path}': (unable to enumerate)")
    print(f"Using starting collection '{col_name}' with {n} chunks; total across all collections: {total}")
    if total <= 0:
        print("No chunks across any collection. Ingest documents first (see src/scripts/index_data.py).")
        return

    query = args.query or DEFAULT_QUERY
    print(f"\nQuery: {query}")
    hits = retriever.retrieve(query, k=args.k)
    print(f"Top-{args.k} hits:")
    _print_hits(hits)


if __name__ == "__main__":
    main()
