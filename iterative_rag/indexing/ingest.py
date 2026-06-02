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
  irag-index \
    --persist chroma_store \
    --collection chemrxiv_graph

  # Only the existing text files folder
  irag-index --only docs --docs-root docs/chemrxiv_graph_v2_texts

  # Only the HF dataset (cc-by), limit per split
  irag-index --only hf --hf-config cc-by --hf-max-per-split 20000 --hf-streaming
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

# Repo components
from iterative_rag.indexing.embedding_config import EmbedderConfig
from iterative_rag.indexing.embeddings import HFEmbedder
from iterative_rag.indexing.chroma_index import ChromaTextIndex
from iterative_rag.chunking.chunking import chunk_document
from iterative_rag.chunking.normalize import normalize_text
from iterative_rag.chunking.io import iter_text_folder, guess_read_text

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


def iter_single_text_file(path: str, *, source: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """Yield a single text file as a document for ingestion."""
    abs_path = os.path.abspath(path)
    text = guess_read_text(abs_path)
    stem = Path(abs_path).stem
    meta = {"source": source or stem, "path": abs_path}
    yield {
        "doc_id": stem,
        "text": text,
        "title": stem,
        "metadata": meta,
    }


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


def _hf_detect_para_id(example: Dict[str, Any], avoid_same_as: Optional[str] = None) -> Optional[str]:
    """Detect a paragraph-level ID, avoiding reuse of the same field/value as doc_id.
    This helps prevent cases where both detectors select the same column (e.g., "id"),
    which can lead to duplicated identifiers like "X::X" across rows.
    """
    candidates = ("paragraph_id", "para_id", "segment_id", "chunk_id", "index", "id")
    for c in candidates:
        v = example.get(c)
        if isinstance(v, (str, int)):
            s = str(v)
            if avoid_same_as is None or s != avoid_same_as:
                return s
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
            # Avoid selecting the exact same field/value for paragraph id
            para_id = _hf_detect_para_id(ex, avoid_same_as=doc_id)
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
    # Track IDs within the current batch to prevent duplicates in a single upsert
    batch_seen_ids: set[str] = set()
    total = 0

    def flush():
        nonlocal texts, metas, ids, total, batch_seen_ids
        if texts:
            # Extra safety: ensure uniqueness just before upsert
            if len(ids) != len(set(ids)):
                unique_texts: List[str] = []
                unique_metas: List[Dict[str, Any]] = []
                unique_ids: List[str] = []
                seen: set[str] = set()
                for t, m, i in zip(texts, metas, ids):
                    if i in seen:
                        continue
                    seen.add(i)
                    unique_texts.append(t)
                    unique_metas.append(m)
                    unique_ids.append(i)
                texts_to_add, metas_to_add, ids_to_add = unique_texts, unique_metas, unique_ids
            else:
                texts_to_add, metas_to_add, ids_to_add = texts, metas, ids
            index.add_documents(texts=texts_to_add, metadatas=metas_to_add, ids=ids_to_add)
            total += len(texts_to_add)
            # Progress reflects actual upsert/embedding work completed
            if pbar_chunks is not None and texts_to_add:
                pbar_chunks.update(len(texts_to_add))
            texts, metas, ids = [], [], []
            batch_seen_ids.clear()

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
        # Deduplicate within the batch to avoid Chroma DuplicateIDError
        for tt, mm, ii in zip(t, m, i):
            if ii in batch_seen_ids:
                continue
            batch_seen_ids.add(ii)
            texts.append(tt)
            metas.append(mm)
            ids.append(ii)
            # Flush as soon as we reach the batch limit to avoid giant single upserts
            # for very large single documents. This also advances the progress bar.
            if len(texts) >= batch_limit_chunks:
                flush()
        if pbar_docs is not None:
            pbar_docs.update(1)
    flush()
    return total


# ------------------------------- CLI -----------------------------------------

def build_index(
    persist_path: str,
    collection_name: str,
    device: str = "cpu",
    model_name: Optional[str] = None,
    batch_size: int = EmbedderConfig.batch_size,
) -> ChromaTextIndex:
    embed_cfg = EmbedderConfig(
        device=device,
        model_name=model_name or EmbedderConfig.model_name,
        batch_size=batch_size,
    )
    embedder = HFEmbedder(embed_cfg)
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
    ap.add_argument("--only", choices=["all", "docs", "hf", "file"], default="all", help="Which sources to index")
    ap.add_argument("--device", default="cpu", help="Device to run embeddings on (cpu, cuda, mps, etc.)")
    ap.add_argument("--model-name", default=EmbedderConfig.model_name, help="SentenceTransformer model name")
    ap.add_argument("--batch-size", type=int, default=EmbedderConfig.batch_size, help="Embedding batch size for SentenceTransformers")
    ap.add_argument("--text-file", default=None, help="Single text file to ingest (optional)")

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

    index = build_index(args.persist, args.collection, args.device, args.model_name, args.batch_size)
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

    if args.text_file:
        if args.only not in {"all", "file"}:
            print(f"Skipping --text-file ({args.text_file}) because --only={args.only}")
        else:
            if not os.path.isfile(args.text_file):
                print(f"Warning: text file not found: {args.text_file}")
            else:
                print(f"Indexing single text file: {args.text_file}")
                # Estimate chunk count to provide a meaningful progress bar.
                try:
                    _txt_for_est = guess_read_text(args.text_file)
                    # Approximate word count similar to chunker (whitespace tokens)
                    est_words = len((_txt_for_est.split()))
                    step = max(1, int(args.words_per_chunk) - int(args.words_overlap))
                    if est_words <= 0:
                        est_chunks = 0
                    elif est_words <= int(args.words_per_chunk):
                        est_chunks = 1
                    else:
                        est_chunks = 1 + (max(0, est_words - int(args.words_per_chunk)) + step - 1) // step
                    print(f"Estimated words: {est_words}; estimated chunks: {est_chunks}")
                except Exception:
                    est_chunks = None  # Fallback: unknown total

                p_docs = tqdm(total=1, desc="file", unit="file", dynamic_ncols=True)
                p_chunks = tqdm(total=est_chunks, desc="chunks", unit="chunk", dynamic_ncols=True)
                try:
                    total_chunks += ck(iter_single_text_file(args.text_file), p_docs, p_chunks)
                finally:
                    p_docs.close(); p_chunks.close()

    print(f"Done. Total chunks inserted: {total_chunks}. Collection '{args.collection}' now has {index.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
