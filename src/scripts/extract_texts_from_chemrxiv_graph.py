#!/usr/bin/env python3
"""
Extract text content from chemrxiv_graph_v2.json into three root folders.

Behavior (as requested):
- Creates exactly three top-level folders under `--out`:
  `pubchem`, `wikipedia`, and `chemrxiv`.
- For each node's `description` entries:
  - `pubchem`: each list item is written to a file in `pubchem/`.
    If the item contains a line like `Name: <value>`, the file name is based on
    that name (sanitized); otherwise a stable hash-based name is used.
  - `wikipedia`: each list item is written to a unique file in `wikipedia/`.
  - Keys that look like filenames (e.g., `*.pdf`): their items are combined and
    written to `chemrxiv/<same_basename>.txt` with description + text content.
  - Any other keys: items are treated as generic text and written to
    `wikipedia/` with unique names.
- If an item has no usable content (no text nor description), nothing is written.

Usage:
  python scripts/extract_texts_from_chemrxiv_graph.py \
    --json docs/chemrxiv_graph_v2.json \
    --out docs/chemrxiv_graph_v2_texts

Notes:
- Default JSON and output locations can be overridden via CLI flags.
- Re-runs are idempotent for content-based files; existing files are skipped
  unless `--overwrite` is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def is_probable_filename(key: str) -> bool:
    """Return True if the key looks like a filename (has an extension like .pdf)."""
    return bool(re.search(r"\.[A-Za-z0-9]{1,6}$", key))


def sanitize_name(name: str) -> str:
    """Sanitize a name for filesystem safety across platforms.

    Allows alphanumerics, dash, underscore, dot; replaces other chars with underscore.
    """
    # Windows reserved names & trimming trailing dots/spaces
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip().strip(".")
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned or "_"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def short_hash(content: str, n: int = 12) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:n]


def content_hash_name(content: str, prefix: str = "sha1_") -> str:
    return f"{prefix}{short_hash(content)}.txt"


def write_text(path: str, text: str, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        return
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def format_item_content(item: Union[str, Dict[str, Any]]) -> str:
    """Format a single list item (string or object) into text content."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        desc = str(item.get("description", "")).strip()
        body = str(item.get("text", "")).strip()
        if desc and body:
            return f"Description:\n{desc}\n\nText:\n{body}\n"
        elif desc:
            return f"Description:\n{desc}\n"
        elif body:
            return body
        else:
            # No usable content – return empty to allow skipping
            return ""
    return str(item)


def combine_items_content(items: Iterable[Union[str, Dict[str, Any]]]) -> str:
    parts: List[str] = []
    for it in items:
        c = format_item_content(it)
        if c.strip():
            parts.append(c)
    return "\n\n---\n\n".join(parts)


def extract_pubchem_name(text: str) -> Optional[str]:
    """Try to extract a `Name: <value>` line for PubChem entries."""
    # Case-insensitive, anchored at line start, capture the remainder of that line
    m = re.search(r"(?im)^\s*Name\s*:\s*(.+?)\s*$", text)
    if m:
        name = m.group(1).strip()
        # Avoid absurdly long file names
        return name[:120]
    return None


def pick_unique_path(base_dir: str, preferred_name: str, content: str) -> Tuple[str, bool]:
    """Return a unique path within `base_dir` for `preferred_name` and whether to write.

    - If `preferred_name.txt` does not exist, use it.
    - If it exists with the same content, skip (return should_write=False).
    - If it exists with different content, append a hash suffix to avoid clobbering.
    """
    stem = sanitize_name(preferred_name)
    base = os.path.join(base_dir, f"{stem}.txt")
    if not os.path.exists(base):
        return base, True
    # Compare contents; if identical, skip writing
    try:
        with open(base, "r", encoding="utf-8") as f:
            existing = f.read()
        if existing == content:
            return base, False
    except Exception:
        pass
    # Different content – disambiguate with hash suffix
    alt = os.path.join(base_dir, f"{stem}__{short_hash(content, 8)}.txt")
    return alt, True


def process_node(description: Dict[str, Any], out_pubchem: str, out_wikipedia: str, out_chemrxiv: str, overwrite: bool = False, verbose: bool = True) -> None:
    for key, value in description.items():
        if not isinstance(value, list):
            continue

        if key.lower() == "pubchem":
            for item in value:
                content = format_item_content(item)
                if not content.strip():
                    continue
                name = extract_pubchem_name(content)
                if name:
                    out_path, should_write = pick_unique_path(out_pubchem, name, content)
                else:
                    fname = content_hash_name(content, prefix="pubchem_")
                    out_path = os.path.join(out_pubchem, fname)
                    should_write = True
                if should_write or overwrite:
                    write_text(out_path, content, overwrite=overwrite)
                    if verbose:
                        print(f"Wrote: {os.path.relpath(out_path, start=os.getcwd())}")

        elif key.lower() == "wikipedia":
            for item in value:
                content = format_item_content(item)
                if not content.strip():
                    continue
                fname = content_hash_name(content, prefix="wiki_")
                out_path = os.path.join(out_wikipedia, fname)
                write_text(out_path, content, overwrite=overwrite)
                if verbose:
                    print(f"Wrote: {os.path.relpath(out_path, start=os.getcwd())}")

        elif is_probable_filename(key):
            base = os.path.splitext(key)[0]
            content = combine_items_content(value)
            if not content.strip():
                continue
            out_path, should_write = pick_unique_path(out_chemrxiv, base, content)
            if should_write or overwrite:
                write_text(out_path, content, overwrite=overwrite)
                if verbose:
                    print(f"Wrote: {os.path.relpath(out_path, start=os.getcwd())}")

        else:
            # Unknown key: treat like generic text -> wikipedia bucket
            for item in value:
                content = format_item_content(item)
                if not content.strip():
                    continue
                fname = content_hash_name(content, prefix="wiki_")
                out_path = os.path.join(out_wikipedia, fname)
                write_text(out_path, content, overwrite=overwrite)
                if verbose:
                    print(f"Wrote: {os.path.relpath(out_path, start=os.getcwd())}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract texts from chemrxiv graph JSON.")
    parser.add_argument(
        "--json",
        default=os.path.join("docs", "chemrxiv_graph_v2.json"),
        help="Path to chemrxiv_graph_v2.json",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("docs", "chemrxiv_graph_v2_texts"),
        help="Output directory for extracted .txt files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output",
    )

    args = parser.parse_args(argv)

    json_path = args.json
    out_dir = args.out
    overwrite = args.overwrite
    verbose = not args.quiet

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 1

    ensure_dir(out_dir)
    out_pubchem = os.path.join(out_dir, "pubchem")
    out_wikipedia = os.path.join(out_dir, "wikipedia")
    out_chemrxiv = os.path.join(out_dir, "chemrxiv")
    ensure_dir(out_pubchem)
    ensure_dir(out_wikipedia)
    ensure_dir(out_chemrxiv)

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            return 2

    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        print("Error: JSON is missing 'nodes' object.", file=sys.stderr)
        return 3

    total_nodes = 0
    for _, node_obj in nodes.items():
        if not isinstance(node_obj, dict):
            continue
        desc = node_obj.get("description")
        if not isinstance(desc, dict):
            continue
        total_nodes += 1
        process_node(desc, out_pubchem, out_wikipedia, out_chemrxiv, overwrite=overwrite, verbose=verbose)

    if verbose:
        print(f"Done. Processed {total_nodes} node(s). Output at: {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
