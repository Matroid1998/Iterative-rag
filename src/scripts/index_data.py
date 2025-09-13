#!/usr/bin/env python3
"""
Index ChemRxiv data into Chroma.

Goals:
- Ingest the already-extracted text files under `docs/chemrxiv_graph_v2_texts`.
- Also index the Hugging Face dataset BASF-AI/ChemRxiv-Paragraphs on the fly
  (streamed) without saving it to disk.

Dependencies:
  pip install chromadb sentence-transformers datasets tqdm

Examples:
  # Index everything using defaults (docs folder + HF dataset streamed)
  # Persisting to ./chroma_store and collection name 'chemrxiv_graph'
  python scripts/index_data.py \
    --persist chroma_store \
    --collection chemrxiv_graph

  # Only the existing text files folder
  python scripts/index_data.py --only docs --docs-root docs/chemrxiv_graph_v2_texts

  # Only the HF dataset (cc-by), limit per split
  python scripts/index_data.py --only hf --hf-config cc-by --hf-max-per-split 20000 --hf-streaming
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

# Repo components
from protocols.embedding_config import EmbedderConfig
from repo.embeddings.embeddeing_models import HFEmbedder
from repo.index.chroma_index import ChromaTextIndex
from repo.utils.chunking import chunk_document
from repo.utils.normalize import normalize_text
from repo.utils.io import iter_text_folder

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class tqdm:  # minimal no-op fallback
        def __init__(self, *args, **kwargs):
            pass
        def update(self, n: int = 1):
            pass
        def close(self):
            pass


# ------------------------------- helpers -------------------------------------


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


# ----- Hugging Face dataset (streamed) ---------------------------------------

def _hf_detect_text(example: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
    if preferred and isinstance(example.get(preferred), str):
        return example.get(preferred)
    for c in ("text", "paragraph", "paragraph_text", "content", "body", "raw_text", "snippet"):
        t = example.get(c)
        if isinstance(t, str):
            return t
    return None


def _hf_detect_doc_id(example: Dict[str, Any]) -> Optional[str]:
    for c in ("doc_id", "document_id", "paper_id", "article_id", "chemrxiv_id", "source_id", "document", "paper", "id"):
        v = example.get(c)
        if isinstance(v, (str, int)):
            return str(v)
    return None


def _hf_detect_para_id(example: Dict[str, Any]) -> Optional[str]:
    for c in ("paragraph_id", "para_id", "segment_id", "chunk_id", "id", "index"):
        v = example.get(c)
        if isinstance(v, (str, int)):
            return str(v)
    return None


def iter_hf_dataset(
    dataset_name: str,
    config: str,
    *,
    streaming: bool = True,
    text_col: Optional[str] = None,
    max_per_split: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Stream BASF-AI/ChemRxiv-Paragraphs and yield docs compatible with our chunker.
    Requires `pip install datasets`.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("The 'datasets' package is required. Install with: pip install datasets") from e

    ds = load_dataset(dataset_name, config, streaming=streaming) if streaming else load_dataset(dataset_name, config)

    # Iterate over all available splits
    if hasattr(ds, "keys"):
        split_items = ((k, ds[k]) for k in ds.keys())
    else:
        split_items = (("train", ds),)

    for split_name, split in split_items:
        count = 0
        for ex in split:
            text = _hf_detect_text(ex, text_col)
            if not isinstance(text, str) or not text.strip():
                continue
            doc_id = _hf_detect_doc_id(ex) or f"hf::{split_name}"
            para_id = _hf_detect_para_id(ex)
            meta = {
                "source": "chemrxiv_hf",
                "dataset": dataset_name,
                "config": config,
                "split": split_name,
            }
            title = None
            if isinstance(ex.get("title"), str):
                title = ex["title"].strip() or None
            yield {
                "doc_id": f"{doc_id}::{para_id}" if para_id else doc_id,
                "text": text,
                "title": title,
                "metadata": meta,
            }
            count += 1
            if max_per_split is not None and count >= max_per_split:
                break


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
    pbar_docs: Optional[Any] = None,
    pbar_chunks: Optional[Any] = None,
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
        if pbar_docs is not None:
            pbar_docs.update(1)
        if pbar_chunks is not None and t:
            pbar_chunks.update(len(t))
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


def _count_docs_in_folder(root: str) -> int:
    exts = {".txt", ".md", ".markdown", ".html", ".htm"}
    root_path = Path(root)
    if not root_path.exists():
        return 0
    return sum(1 for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Index ChemRxiv graph data into Chroma")
    ap.add_argument("--persist", default="chroma_store", help="Chroma persist directory")
    ap.add_argument("--collection", default="chemrxiv_graph", help="Chroma collection name")
    ap.add_argument("--docs-root", default=os.path.join("docs", "chemrxiv_graph_v2_texts"), help="Root folder of extracted text files to ingest")
    ap.add_argument("--only", choices=["all", "docs", "hf"], default="all", help="Which sources to index")

    # Hugging Face dataset options
    ap.add_argument("--hf-dataset", default="BASF-AI/ChemRxiv-Paragraphs", help="HF dataset name")
    ap.add_argument("--hf-config", default="cc-by", choices=["cc-by", "cc-by-nc"], help="HF dataset config (license)")
    ap.add_argument("--hf-streaming", action="store_true", help="Stream examples instead of full download")
    ap.add_argument("--hf-text-col", default=None, help="Explicit text column name (optional)")
    ap.add_argument("--hf-max-per-split", type=int, default=None, help="Limit HF examples per split (debug)")

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

    def ck(docs: Iterable[Dict[str, Any]], pbar_docs: Optional[Any] = None, pbar_chunks: Optional[Any] = None) -> int:
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
            pbar_docs=pbar_docs,
            pbar_chunks=pbar_chunks,
        )

    if args.only in {"all", "docs"}:
        if os.path.isdir(args.docs_root):
            total_files = _count_docs_in_folder(args.docs_root)
            print(f"Indexing text files from: {args.docs_root} (files: {total_files})")
            p_docs = tqdm(total=total_files or None, desc="docs", unit="file", dynamic_ncols=True)
            p_chunks = tqdm(total=None, desc="chunks", unit="chunk", dynamic_ncols=True)
            try:
                total_chunks += ck(iter_docs_from_folder(args.docs_root), p_docs, p_chunks)
            finally:
                p_docs.close(); p_chunks.close()
        else:
            print(f"Warning: docs root not found: {args.docs_root}")

    if args.only in {"all", "hf"}:
        print(f"Indexing HF dataset: {args.hf_dataset} ({args.hf_config})")
        # For streaming, we don't know totals; show running counters.
        p_docs = tqdm(total=None, desc="hf docs", unit="doc", dynamic_ncols=True)
        p_chunks = tqdm(total=None, desc="hf chunks", unit="chunk", dynamic_ncols=True)
        try:
            total_chunks += ck(
                iter_hf_dataset(
                    dataset_name=args.hf_dataset,
                    config=args.hf_config,
                    streaming=args.hf_streaming,
                    text_col=args.hf_text_col,
                    max_per_split=args.hf_max_per_split,
                ),
                p_docs,
                p_chunks,
            )
        finally:
            p_docs.close(); p_chunks.close()

    print(f"Done. Total chunks inserted: {total_chunks}. Collection '{args.collection}' now has {index.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
