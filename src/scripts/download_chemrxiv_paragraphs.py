#!/usr/bin/env python3
"""
Download BASF-AI/ChemRxiv-Paragraphs from Hugging Face and export .txt files.

Each example (paragraph) is written as a standalone .txt file organized as:

  <out>/<split>/<doc_id>/<para_id>.txt

Where `doc_id` and `para_id` are inferred from common column names, or
fall back to content hashes if missing. Empty/whitespace-only paragraphs are
skipped. Designed to be memory-friendly and able to use streaming mode.

Usage examples:
  python scripts/download_chemrxiv_paragraphs.py \
    --config cc-by \
    --out docs/chemrxiv_paragraphs_texts \
    --streaming

  python scripts/download_chemrxiv_paragraphs.py \
    --config cc-by-nc --text-col paragraph --max-per-split 1000

Requirements:
  pip install datasets
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from typing import Any, Dict, Iterable, Optional, Tuple


DATASET_NAME = "BASF-AI/ChemRxiv-Paragraphs"


def sanitize_name(name: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip().strip(".")
    # Windows reserved names
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    if not cleaned:
        cleaned = "_"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha1_hex(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def detect_text(example: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in example:
        t = example.get(preferred)
        if isinstance(t, str):
            return t
    for c in ("text", "paragraph", "paragraph_text", "content", "body", "raw_text", "snippet"):
        t = example.get(c)
        if isinstance(t, str):
            return t
    return None


def detect_doc_id(example: Dict[str, Any], fallback_text: str) -> str:
    for c in (
        "doc_id",
        "document_id",
        "paper_id",
        "article_id",
        "chemrxiv_id",
        "source_id",
        "document",
        "paper",
        "id",
    ):
        v = example.get(c)
        if isinstance(v, (str, int)):
            return sanitize_name(str(v))
    return f"doc_{sha1_hex(fallback_text, 8)}"


def detect_para_id(example: Dict[str, Any], fallback_text: str) -> str:
    for c in ("paragraph_id", "para_id", "segment_id", "chunk_id", "id", "index"):
        v = example.get(c)
        if isinstance(v, (str, int)):
            return sanitize_name(str(v))
    return f"p_{sha1_hex(fallback_text, 8)}"


def write_paragraph(base_out: str, split: str, example: Dict[str, Any], text_col: Optional[str], overwrite: bool, include_meta: bool, quiet: bool) -> int:
    text = detect_text(example, text_col)
    if not isinstance(text, str) or not text.strip():
        return 0
    doc_id = detect_doc_id(example, text)
    para_id = detect_para_id(example, text)
    out_dir = os.path.join(base_out, sanitize_name(split), doc_id)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{para_id}.txt")

    # If file exists with same content, skip
    if os.path.exists(out_path) and not overwrite:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                if f.read() == text:
                    return 0
        except Exception:
            pass

    payload = text
    if include_meta:
        # Minimal metadata header, then a blank line and the paragraph text
        meta_lines = []
        for k in (
            "title",
            "section",
            "doi",
            "url",
            "year",
            "authors",
        ):
            v = example.get(k)
            if v is not None:
                meta_lines.append(f"{k}: {v}")
        if meta_lines:
            payload = "\n".join(meta_lines) + "\n\n" + text

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)

    if not quiet:
        print(f"Wrote: {os.path.relpath(out_path, start=os.getcwd())}")
    return 1


def iter_splits(ds):
    # Supports both DatasetDict and IterableDatasetDict
    if hasattr(ds, "keys"):
        for k in ds.keys():
            yield k, ds[k]
    else:
        # Single dataset or iterable
        yield "train", ds


def export_dataset(out: str, config: str, text_col: Optional[str], streaming: bool, overwrite: bool, include_meta: bool, max_per_split: Optional[int], quiet: bool) -> None:
    try:
        from datasets import load_dataset
    except Exception:
        print("Error: The 'datasets' package is required. Install with: pip install datasets", file=sys.stderr)
        raise

    ensure_dir(out)

    if streaming:
        ds = load_dataset(DATASET_NAME, config, streaming=True)
    else:
        ds = load_dataset(DATASET_NAME, config)

    total = 0
    for split_name, split in iter_splits(ds):
        count = 0
        it: Iterable[Dict[str, Any]] = split
        for ex in it:
            count += write_paragraph(out, split_name, ex, text_col, overwrite, include_meta, quiet)
            if max_per_split is not None and count >= max_per_split:
                break
        total += count
        if not quiet:
            print(f"Split '{split_name}': wrote {count} files")

    if not quiet:
        print(f"Done. Total files written: {total}. Base directory: {os.path.abspath(out)}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"Download {DATASET_NAME} and export paragraphs as .txt files")
    p.add_argument("--out", default=os.path.join("docs", "chemrxiv_paragraphs_texts"), help="Output base directory (default: src/docs/chemrxiv_paragraphs_texts)")
    p.add_argument("--config", default="cc-by", choices=["cc-by", "cc-by-nc"], help="Dataset config to use (license variant)")
    p.add_argument("--text-col", default=None, help="Explicit text column name (optional)")
    p.add_argument("--streaming", action="store_true", help="Use streaming mode to reduce memory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite files when they already exist")
    p.add_argument("--include-meta", action="store_true", help="Prepend lightweight metadata header to files")
    p.add_argument("--max-per-split", type=int, default=None, help="Limit number of files per split (debug)")
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = p.parse_args(list(argv) if argv is not None else None)

    export_dataset(
        out=args.out,
        config=args.config,
        text_col=args.text_col,
        streaming=args.streaming,
        overwrite=args.overwrite,
        include_meta=args.include_meta,
        max_per_split=args.max_per_split,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
