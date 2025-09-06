

from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import math
import re


# ----------------------------- public API -------------------------------------

def chunk_document(
    *,
    doc_id: str,
    text: str,
    title: Optional[str] = None,
    base_metadata: Optional[Dict[str, Any]] = None,
    strategy: str = "auto",               # "auto" | "words" | "tokens"
    # word-based params
    words_per_chunk: int = 200,
    words_overlap: int = 40,
    # token-based params (if tokenizer is provided)
    tokenizer: Optional[Any] = None,      # e.g., transformers.AutoTokenizer.from_pretrained(...)
    tokens_per_chunk: int = 512,
    tokens_overlap: int = 64,
    # housekeeping
    normalize_whitespace: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Split a single document into chunks and return (texts, metadatas, ids).
    Suitable for feeding directly into a vector index (e.g., Chroma.add_documents).

    - If `tokenizer` is provided and strategy in {"auto","tokens"} -> token-based chunking.
    - Else -> word-based sliding window.

    Each metadata contains:
        {
          "doc_id": str,
          "title": Optional[str],
          "chunk_index": int,
          "start_char": Optional[int],   # only for word-based; token-based uses -1
          "end_char": Optional[int],
          "num_words": int,
          "num_tokens": Optional[int],   # if token-based, else None
        }
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    base_meta = dict(base_metadata or {})
    if normalize_whitespace:
        text = _normalize_ws(text)

    if strategy not in {"auto", "words", "tokens"}:
        raise ValueError("strategy must be one of: 'auto', 'words', 'tokens'")

    use_tokens = tokenizer is not None and strategy in {"auto", "tokens"}

    if use_tokens:
        chunks = _chunk_by_tokens(
            text=text,
            tokenizer=tokenizer,
            max_tokens=max(8, int(tokens_per_chunk)),
            overlap=max(0, int(tokens_overlap)),
        )
    else:
        chunks = _chunk_by_words(
            text=text,
            max_words=max(8, int(words_per_chunk)),
            overlap=max(0, int(words_overlap)),
        )

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for i, ch in enumerate(chunks):
        ch_text = ch["text"]
        ch_meta = {
            "doc_id": doc_id,
            "title": title,
            "chunk_index": i,
            "start_char": ch.get("start_char", -1),
            "end_char": ch.get("end_char", -1),
            "num_words": ch.get("num_words", 0),
            "num_tokens": ch.get("num_tokens", None),
        }
        # merge base metadata (base keys do not override chunk keys)
        merged = {**base_meta, **{k: v for k, v in ch_meta.items() if v is not None}}
        cid = _stable_chunk_id(doc_id, i, ch_text)
        texts.append(ch_text)
        metadatas.append(merged)
        ids.append(cid)

    return texts, metadatas, ids


def chunk_corpus(
    docs: Iterable[Dict[str, Any]],
    **kwargs: Any,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Chunk a corpus of documents.
    `docs` items should have keys: {"doc_id", "text"} and optional {"title","metadata"}.

    kwargs are forwarded to `chunk_document(...)`, so you can set strategy/overlaps once.

    Returns concatenated (texts, metadatas, ids) across all docs, in order.
    """
    all_texts: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for d in docs:
        doc_id = str(d["doc_id"])
        text = d["text"]
        title = d.get("title")
        base_metadata = d.get("metadata")
        t, m, i = chunk_document(
            doc_id=doc_id,
            text=text,
            title=title,
            base_metadata=base_metadata,
            **kwargs,
        )
        all_texts.extend(t)
        all_meta.extend(m)
        all_ids.extend(i)

    return all_texts, all_meta, all_ids


# --------------------------- word-based chunking ------------------------------

_WORD_RE = re.compile(r"\S+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _chunk_by_words(text: str, max_words: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Sliding window over *words* with character-accurate start/end spans.
    Uses a greedy word list; overlap words are duplicated between chunks.
    """
    if overlap >= max_words:
        raise ValueError("words_overlap must be < words_per_chunk")

    words = [m.group(0) for m in _WORD_RE.finditer(text)]
    if not words:
        return []

    # Build index from word index -> (start_char, end_char)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for w in words:
        start = text.find(w, pos)
        end = start + len(w)
        spans.append((start, end))
        pos = end

    chunks: List[Dict[str, Any]] = []
    step = max(1, max_words - overlap)
    for start_idx in range(0, len(words), step):
        end_idx = min(len(words), start_idx + max_words)
        if start_idx >= end_idx:
            break
        start_char = spans[start_idx][0]
        end_char = spans[end_idx - 1][1]
        ch_text = text[start_char:end_char]
        chunks.append({
            "text": ch_text,
            "start_char": start_char,
            "end_char": end_char,
            "num_words": end_idx - start_idx,
            "num_tokens": None,
        })
        if end_idx == len(words):
            break

    return _postprocess_tail_merge(chunks, min_len_chars=200)


# -------------------------- token-based chunking ------------------------------

def _chunk_by_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """
    Token-based chunking via a provided tokenizer (e.g., HF AutoTokenizer).
    We chunk on token IDs, then decode each window back to text.
    NOTE: Decoding loses exact original character spans; we set start/end to -1.
    """
    if overlap >= max_tokens:
        raise ValueError("tokens_overlap must be < tokens_per_chunk")

    # Encode without special tokens; keep it simple and consistent
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []

    chunks: List[Dict[str, Any]] = []
    step = max(1, max_tokens - overlap)

    for start in range(0, len(ids), step):
        end = min(len(ids), start + max_tokens)
        if start >= end:
            break
        sub_ids = ids[start:end]
        ch_text = tokenizer.decode(sub_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # Trim whitespace artifacts from decoding
        ch_text = _normalize_ws(ch_text)
        if ch_text:
            chunks.append({
                "text": ch_text,
                "start_char": -1,
                "end_char": -1,
                "num_words": len(ch_text.split()),
                "num_tokens": len(sub_ids),
            })
        if end == len(ids):
            break

    return _postprocess_tail_merge(chunks, min_len_chars=200)


# ------------------------------- utilities -----------------------------------

def _normalize_ws(s: str) -> str:
    # Collapse multiple whitespace into single spaces; preserve newlines lightly.
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)   # cap consecutive newlines at 2
    return s.strip()

def _stable_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    # Stable but unique ID using doc_id + index + short hash of content
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}::chunk_{chunk_index:05d}_{h}"

def _postprocess_tail_merge(chunks: List[Dict[str, Any]], min_len_chars: int = 160) -> List[Dict[str, Any]]:
    """
    If the last chunk is too short, merge it into the previous one
    to avoid tiny trailing chunks that hurt retrieval.
    """
    if len(chunks) >= 2 and len(chunks[-1]["text"]) < min_len_chars:
        prev = chunks[-2]
        last = chunks[-1]
        merged_text = (prev["text"].rstrip() + " " + last["text"].lstrip()).strip()
        prev["text"] = merged_text
        prev["num_words"] = len(merged_text.split())
        if prev.get("num_tokens") is not None and last.get("num_tokens") is not None:
            prev["num_tokens"] = prev["num_tokens"] + last["num_tokens"]
        # adjust spans if character-based
        if prev.get("start_char", -1) >= 0 and last.get("end_char", -1) >= 0:
            prev["end_char"] = last["end_char"]
        chunks.pop()
    return chunks
