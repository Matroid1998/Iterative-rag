#!/usr/bin/env python3
"""Endpoint 1 — chunk documents and build the Chroma vector store.

Reads raw text documents (the ``chemrxiv``/``pubchem``/``wikipedia`` corpus folders),
normalizes + chunks them, embeds the chunks with a SentenceTransformer model, and
upserts them into a persistent Chroma collection that the iterative RAG system queries.

Examples
--------
    # Index the shipped corpus into ./chroma_store (collection 'chemrxiv_graph')
    irag-index

    # Rebuild the corpus text files from the graph JSON first, then index
    irag-index --from-graph

    # Index just one sub-folder, capped to N documents (quick smoke test)
    irag-index --docs-root data/docs/chemrxiv_graph_v2_texts/wikipedia --collection smoke --limit 50
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from iterative_rag import config


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk documents and build the Chroma index.")
    ap.add_argument("--docs-root", default=str(config.CORPUS_DIR),
                    help="Root folder of .txt/.md/.html documents to ingest.")
    ap.add_argument("--persist", default=str(config.CHROMA_DIR), help="Chroma persist directory.")
    ap.add_argument("--collection", default=config.DEFAULT_COLLECTION, help="Chroma collection name.")
    ap.add_argument("--model-name", default=config.DEFAULT_EMBED_MODEL, help="SentenceTransformer embedding model.")
    ap.add_argument("--device", default=config.DEFAULT_DEVICE, help="Embedding device (cpu, cuda, mps, ...).")
    ap.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    ap.add_argument("--words-per-chunk", type=int, default=220)
    ap.add_argument("--words-overlap", type=int, default=50)
    ap.add_argument("--chunk-strategy", choices=["auto", "words", "tokens"], default="auto")
    ap.add_argument("--no-normalize", action="store_true", help="Disable text normalization before chunking.")
    ap.add_argument("--limit", type=int, default=None, help="Max number of documents to ingest (debug).")
    ap.add_argument("--from-graph", action="store_true",
                    help="Rebuild the corpus text files from the graph JSON before indexing.")
    ap.add_argument("--graph-json", default=str(config.GRAPH_JSON), help="Path to chemrxiv_graph_v2.json (with --from-graph).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    # Imported here so `--help` works without the heavy ML stack installed.
    from iterative_rag.indexing.ingest import build_index, iter_docs_from_folder, add_docs_streaming

    if args.from_graph:
        from iterative_rag.chunking.corpus_builder import build_corpus
        print(f"Rebuilding corpus from {args.graph_json} -> {args.docs_root}")
        build_corpus(args.graph_json, args.docs_root, overwrite=False, verbose=False)

    if not os.path.isdir(args.docs_root):
        raise SystemExit(f"docs root not found: {args.docs_root}")

    index = build_index(
        persist_path=args.persist,
        collection_name=args.collection,
        device=args.device,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    docs = iter_docs_from_folder(args.docs_root)
    if args.limit:
        import itertools
        docs = itertools.islice(docs, args.limit)

    print(f"Indexing documents from {args.docs_root} into collection '{args.collection}' ...")
    total = add_docs_streaming(
        index=index,
        docs=docs,
        normalize=not args.no_normalize,
        words_per_chunk=args.words_per_chunk,
        words_overlap=args.words_overlap,
        chunk_strategy=args.chunk_strategy,
    )
    print(f"Done. Inserted {total} chunks. Collection '{args.collection}' now holds {index.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
