#!/usr/bin/env python3
"""
Index ChemRxiv graph data into Chroma.

Goals:
- Avoid writing gigantic PubChem paragraphs to disk: ingest them straight from
  the JSON into Chroma with sensible chunking and metadata.
- Also ingest the existing text files under `docs/chemrxiv_graph_v2_texts` into
  the same collection.

Dependencies:
  pip install chromadb sentence-transformers

Examples:
  # Index everything using defaults
  python scripts/index_data.py \
    --persist data/chroma_store \
    --collection chemrxiv_graph

  # Only PubChem from JSON, word chunks 200/40
  python scripts/index_data.py --only pubchem --words-per-chunk 200 --words-overlap 40

  # Only the existing text files folder
  python scripts/index_data.py --only docs --docs-root docs/chemrxiv_graph_v2_texts
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Repo components
from protocols.embedding_config import EmbedderConfig
from repo.embeddings.embeddeing_models import HFEmbedder
from repo.index.chroma_index import ChromaTextIndex
from repo.utils.chunking import chunk_document
from repo.utils.normalize import normalize_text
from repo.utils.io import iter_text_folder


# ------------------------------- helpers -------------------------------------

def extract_pubchem_items(json_path: str) -> Iterable[Dict[str, Any]]:
    """
    Stream PubChem items from the chemrxiv graph JSON.
    Yields records of the form {"doc_id","text","title"?,"metadata"} where:
      - doc_id: stable id based on node name and hash
      - text: the raw PubChem paragraph
      - metadata: {source: 'pubchem', node: <node>, key: 'pubchem', name?: <extracted Name>}
    Skips empty/whitespace-only entries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        return

    def _extract_name(s: str) -> Optional[str]:
        m = re.search(r"(?im)^\s*Name\s*:\s*(.+?)\s*$", s)
        return m.group(1).strip() if m else None

    for node_name, node_obj in nodes.items():
        if not isinstance(node_obj, dict):
            continue
        desc = node_obj.get("description")
        if not isinstance(desc, dict):
            continue
        pub = desc.get("pubchem")
        if not isinstance(pub, list):
            continue
        for idx, item in enumerate(pub):
            text = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                # Support both string and {description,text}
                d = str(item.get("description", "")).strip()
                t = str(item.get("text", "")).strip()
                if d and t:
                    text = f"Description:\n{d}\n\nText:\n{t}\n"
                elif d:
                    text = d
                elif t:
                    text = t
            if not text or not str(text).strip():
                continue

            name = _extract_name(text)
            # Build a stable ID. Prefer Name if present, else fallback to index.
            if name:
                doc_id = f"pubchem::{node_name}::{name}::{idx:04d}"
            else:
                doc_id = f"pubchem::{node_name}::item_{idx:04d}"
            meta = {"source": "pubchem", "node": node_name, "key": "pubchem"}
            if name:
                meta["name"] = name
            yield {"doc_id": doc_id, "text": text, "title": name or None, "metadata": meta}


def iter_docs_from_folder(root: str) -> Iterable[Dict[str, Any]]:
    """Iterate .txt/.md/.html docs from a folder and enrich metadata with source.
    Adds metadata['source'] inferred from the first path component if present.
    """
    root = os.path.abspath(root)
    root_path = os.path.normpath(root)
    plen = len(root_path.rstrip(os.sep)) + 1
    for doc in iter_text_folder(root, recursive=True, include_html=True, strip_html=False):
        rel = doc.get("doc_id", "")
        rel_os = rel.replace("/", os.sep)
        # Infer top-level folder name as source if available
        source = None
        parts = rel_os.split(os.sep)
        if len(parts) >= 1:
            head = parts[0].lower()
            if head in {"pubchem", "wikipedia", "chemrxiv"}:
                source = head
        meta = dict(doc.get("metadata") or {})
        if source and "source" not in meta:
            meta["source"] = source
        doc["metadata"] = meta
        yield doc


def add_docs_streaming(
    index: ChromaTextIndex,
    docs: Iterable[Dict[str, Any]],
    *,
    normalize: bool = True,
    words_per_chunk: int = 220,
    words_overlap: int = 50,
    tokenizer: Optional[Any] = None,
    tokens_per_chunk: int = 512,
    tokens_overlap: int = 64,
    chunk_strategy: str = "auto",
    batch_limit_chunks: int = 4096,
) -> int:
    """Stream docs -> chunk -> batch upserts into Chroma. Returns chunks added."""
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []
    total = 0

    def flush():
        nonlocal texts, metas, ids, total
        if texts:
            index.add_documents(texts=texts, metadatas=metas, ids=ids)
            total += len(texts)
            texts, metas, ids = [], [], []

    for d in docs:
        text = d.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        if normalize:
            text = normalize_text(text)
        t, m, i = chunk_document(
            doc_id=str(d.get("doc_id")),
            text=text,
            title=d.get("title"),
            base_metadata=d.get("metadata") or {},
            strategy=chunk_strategy,
            words_per_chunk=words_per_chunk,
            words_overlap=words_overlap,
            tokenizer=tokenizer,
            tokens_per_chunk=tokens_per_chunk,
            tokens_overlap=tokens_overlap,
        )
        texts.extend(t)
        metas.extend(m)
        ids.extend(i)
        if len(texts) >= batch_limit_chunks:
            flush()
    flush()
    return total


# ------------------------------- CLI -----------------------------------------

def build_index(persist_path: str, collection_name: str) -> ChromaTextIndex:
    embedder = HFEmbedder(EmbedderConfig())
    return ChromaTextIndex(
        persist_path=persist_path,
        collection_name=collection_name,
        embedder=embedder,
        distance="cosine",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Index ChemRxiv graph data into Chroma")
    ap.add_argument("--persist", default=os.path.join("data", "chroma_store"), help="Chroma persist directory")
    ap.add_argument("--collection", default="chemrxiv_graph", help="Chroma collection name")
    ap.add_argument("--json", default=os.path.join("docs", "chemrxiv_graph_v2.json"), help="Path to chemrxiv graph JSON")
    ap.add_argument("--docs-root", default=os.path.join("docs", "chemrxiv_graph_v2_texts"), help="Root folder of extracted text files to ingest")
    ap.add_argument("--only", choices=["all", "pubchem", "docs"], default="all", help="Limit to a subset to ingest")

    # Chunking
    ap.add_argument("--chunk-strategy", choices=["auto", "words", "tokens"], default="auto")
    ap.add_argument("--words-per-chunk", type=int, default=220)
    ap.add_argument("--words-overlap", type=int, default=50)
    ap.add_argument("--tokens-per-chunk", type=int, default=512)
    ap.add_argument("--tokens-overlap", type=int, default=64)
    ap.add_argument("--no-normalize", action="store_true", help="Disable text normalization before chunking")
    ap.add_argument("--batch-limit-chunks", type=int, default=4096, help="Flush to Chroma every N chunks")

    args = ap.parse_args(list(argv) if argv is not None else None)

    index = build_index(args.persist, args.collection)
    total_chunks = 0

    def ck(docs: Iterable[Dict[str, Any]]) -> int:
        return add_docs_streaming(
            index=index,
            docs=docs,
            normalize=not args.no_normalize,
            words_per_chunk=args.words_per_chunk,
            words_overlap=args.words_overlap,
            tokens_per_chunk=args.tokens_per_chunk,
            tokens_overlap=args.tokens_overlap,
            chunk_strategy=args.chunk_strategy,
            batch_limit_chunks=args.batch_limit_chunks,
        )

    if args.only in {"all", "pubchem"}:
        print(f"Indexing PubChem from JSON: {args.json}")
        total_chunks += ck(extract_pubchem_items(args.json))

    if args.only in {"all", "docs"}:
        if os.path.isdir(args.docs_root):
            print(f"Indexing text files from: {args.docs_root}")
            total_chunks += ck(iter_docs_from_folder(args.docs_root))
        else:
            print(f"Warning: docs root not found: {args.docs_root}")

    print(f"Done. Total chunks inserted: {total_chunks}. Collection '{args.collection}' now has {index.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

