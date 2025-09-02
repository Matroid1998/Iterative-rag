# repo/utils/normalize.py
"""
Normalization utilities for unstructured text preprocessing.

Goals:
- Make inputs consistent for indexing & retrieval.
- Stay dependency-light (stdlib only).
- Be safe for multilingual corpora (English, French, Persian, etc.).

Key features:
- Unicode normalization (NFKC by default)
- Remove zero-width/control characters & BOM
- Normalize whitespace (configurable newline handling)
- Normalize quotes/dashes/ellipses
- Optional accent stripping & ASCII fallback
- Small helpers for titles, identifiers, and line-level cleanup
"""

from typing import Iterable, List, Optional
import re
import unicodedata

# ----------------------------- constants --------------------------------------

# Zero-width & directional marks commonly found in text
_ZERO_WIDTH_CHARS = [
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # BOM
    "\u00ad",  # SOFT HYPHEN
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # directional embedding/override
    "\u2066", "\u2067", "\u2068", "\u2069",  # directional isolates
]

# Map fancy quotes/dashes/ellipsis to ASCII-friendly forms
_QUOTES_MAP = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'",
    "′": "'", "″": '"',  # prime symbols often used as quotes
}
_DASHES_MAP = {
    "—": "-",  # em dash
    "–": "-",  # en dash
    "−": "-",  # minus sign
    "‐": "-",  # hyphen
    "‒": "-",  # figure dash
}
_ELLIPSIS_MAP = {"…": "..."}

# Non-breaking spaces and friends → regular space
_SPACE_LIKE_MAP = {
    "\u00a0": " ",  # NBSP
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ",
    "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ",
    "\u2008": " ", "\u2009": " ", "\u200a": " ",
}

# Precompiled regexes
_RE_MULTISPACE = re.compile(r"[ \t\r\f\v]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")  # cap to at most 2
_RE_TRAIL_WS = re.compile(r"[ \t]+$", flags=re.MULTILINE)
_RE_LEAD_WS = re.compile(r"^[ \t]+", flags=re.MULTILINE)

# --------------------------- core normalizers ---------------------------------


def normalize_unicode(s: str, form: str = "NFKC", remove_control: bool = True) -> str:
    """
    Normalize Unicode and optionally strip control/zero-width chars.
    Recommended form: NFKC for search/retrieval consistency.
    """
    if not isinstance(s, str):
        raise TypeError("normalize_unicode expects a string")
    s = unicodedata.normalize(form, s)
    if remove_control:
        # Drop Cc (control) category except \n and \t, and explicit zero-width list
        cleaned = []
        zw = set(_ZERO_WIDTH_CHARS)
        for ch in s:
            if ch in zw:
                continue
            cat = unicodedata.category(ch)
            if cat == "Cc" and ch not in ("\n", "\t"):
                continue
            cleaned.append(ch)
        s = "".join(cleaned)
    return s


def normalize_punctuation(s: str, fix_quotes: bool = True, fix_dashes: bool = True, fix_ellipsis: bool = True) -> str:
    """
    Normalize typographic punctuation to consistent ASCII-ish forms.
    - curly quotes → straight quotes
    - em/en/minus dashes → hyphen-minus
    - ellipsis → "..."
    """
    if fix_quotes:
        for k, v in _QUOTES_MAP.items():
            s = s.replace(k, v)
    if fix_dashes:
        for k, v in _DASHES_MAP.items():
            s = s.replace(k, v)
    if fix_ellipsis:
        for k, v in _ELLIPSIS_MAP.items():
            s = s.replace(k, v)
    for k, v in _SPACE_LIKE_MAP.items():
        s = s.replace(k, v)
    return s


def normalize_whitespace(s: str, keep_newlines: bool = True, strip_lines: bool = True, collapse_to_single: bool = True) -> str:
    """
    Normalize whitespace:
    - Convert exotic spaces to normal spaces (already handled in punctuation pass).
    - Collapse runs of spaces/tabs.
    - Trim leading/trailing spaces on lines.
    - Control newline density (keep at most two consecutive newlines).
    """
    # Collapse horizontal whitespace runs
    s = _RE_MULTISPACE.sub(" ", s)

    if strip_lines:
        s = _RE_LEAD_WS.sub("", s)
        s = _RE_TRAIL_WS.sub("", s)

    if keep_newlines:
        # Cap blank lines to at most 2
        s = _RE_MULTINEWLINE.sub("\n\n", s)
        # Strip outer whitespace but keep internal newlines
        s = s.strip()
    else:
        # Replace newlines with spaces then collapse again
        s = s.replace("\n", " ").strip()
        s = _RE_MULTISPACE.sub(" ", s)

    if collapse_to_single:
        # Just to be sure—collapse any new horizontal runs again
        s = _RE_MULTISPACE.sub(" ", s)

    return s


def strip_accents(s: str) -> str:
    """
    Remove diacritics/accents while preserving base characters.
    Useful for ASCII-only systems or coarse matching.
    """
    norm = unicodedata.normalize("NFD", s)
    out = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", out)


def to_ascii(s: str) -> str:
    """
    Best-effort ASCII fallback: strip accents and drop non-ASCII.
    """
    s = strip_accents(s)
    return s.encode("ascii", "ignore").decode("ascii")


# ----------------------------- pipeline ---------------------------------------


def normalize_text(
    s: str,
    *,
    unicode_form: str = "NFKC",
    remove_control: bool = True,
    fix_quotes: bool = True,
    fix_dashes: bool = True,
    fix_ellipsis: bool = True,
    keep_newlines: bool = True,
    strip_lines: bool = True,
    collapse_to_single: bool = True,
    lowercase: bool = False,
    strip_accents_flag: bool = False,
    ascii_only: bool = False,
) -> str:
    """
    End-to-end normalization pipeline.
    Configure via flags; safe defaults for retrieval.

    Typical settings:
    - For indexing: keep_newlines=True, lowercase=False (case can carry meaning), ascii_only=False
    - For identifiers/keys: lowercase=True, ascii_only=True

    NOTE: `ascii_only=True` implies `strip_accents_flag=True`.
    """
    if not isinstance(s, str):
        raise TypeError("normalize_text expects a string")

    s = normalize_unicode(s, form=unicode_form, remove_control=remove_control)
    s = normalize_punctuation(s, fix_quotes=fix_quotes, fix_dashes=fix_dashes, fix_ellipsis=fix_ellipsis)
    s = normalize_whitespace(s, keep_newlines=keep_newlines, strip_lines=strip_lines, collapse_to_single=collapse_to_single)

    if lowercase:
        s = s.lower()

    if ascii_only:
        s = to_ascii(s)
    elif strip_accents_flag:
        s = strip_accents(s)

    return s


# ----------------------------- line helpers -----------------------------------


def normalize_lines(lines: Iterable[str], **kwargs) -> List[str]:
    """
    Apply `normalize_text` to an iterable of lines, returning a list.
    Useful when pre-cleaning CSV/TSV rows before joining.
    """
    return [normalize_text(line, **kwargs) for line in lines]


def compact_paragraphs(s: str) -> str:
    """
    Slightly more aggressive whitespace compaction:
    - normalize_text with keep_newlines=True
    - collapse multiple blank paragraphs to a single blank line
    """
    s = normalize_text(s, keep_newlines=True)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_title(s: str, max_len: Optional[int] = None) -> str:
    """
    Normalize a short title/heading:
    - full normalization (no lowercase)
    - collapse inner spaces
    - optional max length with ellipsis
    """
    s = normalize_text(s, keep_newlines=False, lowercase=False)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip(" -–—")
    if max_len is not None and len(s) > max_len:
        s = s[: max(0, max_len - 1)].rstrip() + "…"
    return s
