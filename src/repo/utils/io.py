# repo/utils/io.py
"""
Lightweight I/O helpers for loading corpora and writing artifacts.

Design goals:
- stdlib-only (no hard deps). Optional helpers auto-detect extras if installed.
- Friendly to our RAG pipeline: load JSONL corpora, scan text folders, and write JSONL.
- Safe file writes (atomic), reasonable encoding handling, simple HTML stripping (optional).

Common formats:
- JSONL/NDJSON lines with objects like: {"doc_id": "...", "text": "...", "title": "...", "metadata": {...}}
- Folders of .txt / .md / .html files (HTML stripping is optional, best-effort).

Tip: Keep heavy parsing (PDF, DOCX) out of here—do it upstream and pass plain text.
"""

from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
import contextlib
import csv
import io
import json
import os
import pathlib
import re
import shutil
import tempfile
import time
import unicodedata
import hashlib


# --------------------------- path & filesystem --------------------------------

PathLike = Union[str, os.PathLike]

def ensure_dir(path: PathLike) -> None:
    """Create directory if missing (parents ok)."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text to file: write to temp file then replace().
    Prevents partial writes if the process crashes mid-write.
    """
    path = pathlib.Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as tmp:
        tmp.write(text)
        tmp_path = pathlib.Path(tmp.name)
    os.replace(tmp_path, path)

def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Atomic write for bytes."""
    path = pathlib.Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp_path = pathlib.Path(tmp.name)
    os.replace(tmp_path, path)

def file_metadata(path: PathLike) -> Dict[str, Any]:
    """Return basic metadata (size, mtime) for a file path."""
    p = pathlib.Path(path)
    st = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "stem": p.stem,
        "suffix": p.suffix.lower(),
        "size_bytes": int(st.st_size),
        "mtime": int(st.st_mtime),
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }


# --------------------------- encodings & text ---------------------------------

_DEFAULT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

def guess_read_text(path: PathLike, encodings: Tuple[str, ...] = _DEFAULT_ENCODINGS) -> str:
    """
    Read text trying a few common encodings. Last resort: 'utf-8' with errors='replace'.
    Keeps things dependency-light (no chardet).
    """
    p = pathlib.Path(path)
    for enc in encodings:
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def sanitize_newlines(s: str) -> str:
    """Normalize line endings to Unix newlines."""
    return s.replace("\r\n", "\n").replace("\r", "\n")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
def strip_html_simple(html: str) -> str:
    """
    Best-effort HTML → text (very simple). For robust HTML parsing, use BeautifulSoup upstream.
    """
    # Remove script/style blocks
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", html)
    # Replace <br> and block tags with newlines
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|h[1-6]|li|ul|ol|section|article|header|footer|tr)>", "\n", html)
    # Strip remaining tags
    text = _HTML_TAG_RE.sub("", html)
    # Unescape a few entities
    text = (text
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'"))
    # Collapse whitespace
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def slugify(s: str, max_len: Optional[int] = 120) -> str:
    """
    Safe-ish file/id slug: NFKD normalize, strip accents, keep alnum/_/-
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9_\-]+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


# ------------------------------ JSONL / NDJSON --------------------------------

def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Load a whole JSONL file into memory (small/medium files)."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def iter_jsonl(path: PathLike) -> Generator[Dict[str, Any], None, None]:
    """Stream a JSONL file line-by-line (generator)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]], append: bool = False) -> None:
    """Write/append records to JSONL, one JSON object per line (UTF-8)."""
    mode = "a" if append else "w"
    ensure_dir(pathlib.Path(path).parent)
    with open(path, mode, encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

@contextlib.contextmanager
def jsonl_writer(path: PathLike, append: bool = False):
    """Context manager for streaming JSONL writes."""
    mode = "a" if append else "w"
    ensure_dir(pathlib.Path(path).parent)
    f = open(path, mode, encoding="utf-8")
    try:
        yield lambda obj: f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        f.close()


# ------------------------------ CSV (stdlib) ----------------------------------

def read_csv_dicts(path: PathLike, delimiter: str = ",") -> List[Dict[str, str]]:
    """Read CSV as a list of dicts (stdlib only)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))

def iter_csv_dicts(path: PathLike, delimiter: str = ",") -> Generator[Dict[str, str], None, None]:
    """Stream CSV rows as dicts."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield row


# ------------------------------ corpus loaders --------------------------------

_TEXT_EXTS = (".txt", ".md", ".markdown")
_HTML_EXTS = (".html", ".htm")

def iter_text_folder(
    root: PathLike,
    *,
    recursive: bool = True,
    include_html: bool = True,
    strip_html: bool = False,
    follow_symlinks: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield documents from a folder. For each file, produce:
      {"doc_id": <relative path>, "text": "...", "title": <stem>, "metadata": {...}}

    - include_html: include .html/.htm files
    - strip_html: apply best-effort tag stripping to HTML content
    """
    root_path = pathlib.Path(root)
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for p in root_path.glob(pattern):
        if not p.is_file():
            continue
        if not follow_symlinks and p.is_symlink():
            continue

        ext = p.suffix.lower()
        if ext in _TEXT_EXTS or (include_html and ext in _HTML_EXTS):
            try:
                raw = guess_read_text(p)
                if ext in _HTML_EXTS and strip_html:
                    text = strip_html_simple(raw)
                else:
                    text = sanitize_newlines(raw)
            except Exception:
                # Skip unreadable files but continue scanning
                continue

            rel = str(p.relative_to(root_path)) if p.is_relative_to(root_path) else str(p)
            meta = file_metadata(p)
            doc = {
                "doc_id": rel.replace(os.sep, "/"),
                "text": text,
                "title": p.stem,
                "metadata": meta,
            }
            yield doc

def load_corpus_from_jsonl(
    path: PathLike,
    *,
    required_keys: Tuple[str, ...] = ("doc_id", "text"),
) -> List[Dict[str, Any]]:
    """
    Load a JSONL corpus and validate minimal keys.
    Returns a list of {"doc_id","text","title"?,"metadata"?}.
    """
    items = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for obj in items:
        for k in required_keys:
            if k not in obj:
                raise ValueError(f"JSONL record missing required key '{k}': {obj}")
        doc = {
            "doc_id": str(obj["doc_id"]),
            "text": str(obj["text"]),
            "title": str(obj.get("title")) if obj.get("title") is not None else None,
            "metadata": obj.get("metadata") or {},
        }
        out.append(doc)
    return out


# ------------------------------- misc helpers ---------------------------------

def sha1_hex(data: Union[str, bytes]) -> str:
    """SHA1 hex of a string or bytes (useful for stable IDs)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()

def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID from doc_id + index + short text hash."""
    h = sha1_hex(text)[:10]
    return f"{doc_id}::chunk_{chunk_index:05d}_{h}"

def save_texts_as_jsonl(
    path: PathLike,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> None:
    """
    Convenience: write parallel lists (texts, metadatas, ids) to JSONL.
    Useful after chunking to snapshot the dataset.
    """
    if metadatas is None:
        metadatas = [{} for _ in texts]
    if ids is None:
        ids = [str(i) for i in range(len(texts))]
    if not (len(texts) == len(metadatas) == len(ids)):
        raise ValueError("texts, metadatas, and ids must have the same length.")
    records = []
    for t, m, i in zip(texts, metadatas, ids):
        records.append({"id": i, "text": t, "metadata": m})
    write_jsonl(path, records, append=False)
