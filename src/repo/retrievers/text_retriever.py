from typing import Any, Dict, List, Optional
import hashlib
import math

from protocols.retrieval_interfaces import LexicalBackend, VectorTextIndex

class TextRetriever:
    """
    Hybrid-ready text retriever over a vector index (e.g., Chroma).
    - Dense side: uses `index.query(query_text, k, where, where_document)` where
      the result is a list of dicts: {"id", "score" (distance), "text", "metadata"}.
      NOTE: `score` from the index is a *distance* (lower is better).
    - Lexical side (optional): any backend implementing `LexicalBackend`.

    This class rescales dense distances → similarities, optionally fuses with
    lexical scores (min-max normalization), and returns a single ranked list.
    """

    def __init__(
        self,
        index: Any,
        *,
        distance_space: str = "cosine",   # "cosine" | "l2" | "ip"
        lexical: Optional[LexicalBackend] = None,
        dense_weight: float = 0.7,        # weight for dense in hybrid fusion
        oversample_dense: int = 2,        # fetch ~k*oversample_dense from dense side before re-ranking
        oversample_lexical: int = 2,      # same idea for lexical
    ) -> None:
        self.index = index
        self.distance_space = distance_space.lower()
        self.lexical = lexical
        self.alpha = float(dense_weight)
        self.oversample_dense = max(1, int(oversample_dense))
        self.oversample_lexical = max(1, int(oversample_lexical))

    # --------------------------- public API -----------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for `query`.
        Returns a list of dicts with keys:
            id: str
            text: str
            metadata: dict
            score_dense: float | None          (higher = better)
            score_lexical: float | None        (higher = better)
            score: float                       (final fused score; higher = better)
            source: "dense" | "lexical" | "hybrid"
        """
        # --- Dense candidates (distances -> similarities) ---
        n_dense = k * self.oversample_dense
        dense_hits = self.index.query(
            query_text=query,
            k=n_dense,
            where=where,
            where_document=where_document,
        )
        for h in dense_hits:
            h["score_dense"] = self._distance_to_similarity(h.get("score", None))
            h["score_lexical"] = None
            h["source"] = "dense"
            # Ensure a stable string id
            h["id"] = _ensure_id(h)

        # --- Lexical candidates (optional) ---
        lexical_hits: List[Dict[str, Any]] = []
        if self.lexical is not None:
            n_lex = k * self.oversample_lexical
            lexical_hits = self.lexical.search(
                query=query, k=n_lex, where=where, where_document=where_document
            )
            for h in lexical_hits:
                # Conform fields
                h.setdefault("metadata", {})
                h.setdefault("text", "")
                h["score_lexical"] = float(h.get("score", 0.0))
                h["score_dense"] = None
                h["source"] = "lexical"
                h["id"] = _ensure_id(h)

        # --- Merge & normalize ------------------------------------------------
        merged: Dict[str, Dict[str, Any]] = {}
        for h in dense_hits:
            merged[h["id"]] = {
                "id": h["id"],
                "text": h.get("text", ""),
                "metadata": h.get("metadata", {}),
                "score_dense": h["score_dense"],
                "score_lexical": None,
                "source": "dense",
            }
        for h in lexical_hits:
            m = merged.get(h["id"])
            if m is None:
                merged[h["id"]] = {
                    "id": h["id"],
                    "text": h.get("text", ""),
                    "metadata": h.get("metadata", {}),
                    "score_dense": None,
                    "score_lexical": h["score_lexical"],
                    "source": "lexical",
                }
            else:
                # combine fields; prefer non-empty text/metadata if missing
                if not m.get("text"):
                    m["text"] = h.get("text", "")
                if not m.get("metadata"):
                    m["metadata"] = h.get("metadata", {})
                m["score_lexical"] = h["score_lexical"]
                m["source"] = "hybrid"

        # Collect for normalization
        items = list(merged.values())
        dense_vals = [x["score_dense"] for x in items if x["score_dense"] is not None]
        lex_vals = [x["score_lexical"] for x in items if x["score_lexical"] is not None]

        dense_min, dense_max = _min_max(dense_vals)
        lex_min, lex_max = _min_max(lex_vals)

        for x in items:
            nd = _min_max_norm(x["score_dense"], dense_min, dense_max)
            nl = _min_max_norm(x["score_lexical"], lex_min, lex_max)
            if nd is None and nl is None:
                # Should not happen (no scores present), default to 0
                fused = 0.0
            elif nd is None:
                fused = nl
            elif nl is None:
                fused = nd
            else:
                fused = self.alpha * nd + (1.0 - self.alpha) * nl
            x["score"] = float(fused) if fused is not None else 0.0

        # Sort by fused score desc and trim
        items.sort(key=lambda r: r["score"], reverse=True)
        return items[:k]

    # --------------------------- helpers --------------------------------------

    def _distance_to_similarity(self, dist: Optional[float]) -> Optional[float]:
        """Convert an index 'distance' into a 'similarity' where higher is better."""
        if dist is None:
            return None
        d = float(dist)
        if self.distance_space == "cosine":
            # Chroma cosine distance ≈ 1 - cosine_similarity
            return 1.0 - d
        if self.distance_space == "l2":
            # Smaller L2 distance is better → map to a decreasing function
            return -d
        if self.distance_space == "ip":
            # Inner product: higher IP is better already
            return d
        # Default conservative fallback
        return -d


# --- small utilities ----------------------------------------------------------

def _ensure_id(hit: Dict[str, Any]) -> str:
    """
    Guarantee a deterministic string ID for a hit.
    Prefer provided 'id'; else hash the text+metadata.
    """
    if "id" in hit and hit["id"] is not None:
        return str(hit["id"])
    text = str(hit.get("text", ""))
    meta = str(hit.get("metadata", ""))
    h = hashlib.sha1((text + "||" + meta).encode("utf-8")).hexdigest()
    return h

def _min_max(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    vmin = min(values)
    vmax = max(values)
    return vmin, vmax

def _min_max_norm(v: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> Optional[float]:
    if v is None or vmin is None or vmax is None:
        return None
    if math.isclose(vmax, vmin):
        return 1.0  # all identical; treat as max
    return (v - vmin) / (vmax - vmin)