from typing import Any, Dict, List, Optional, Tuple
import math


class MultiCollectionRetriever:
    """
    Dense-only retriever that queries multiple vector indexes (collections)
    and returns a single merged ranked list.

    Each index must implement `.query(query_text, k, where, where_document)` and
    return hits with keys: {"id","score"(distance),"text","metadata"}.
    """

    def __init__(
        self,
        indexes: List[Tuple[Any, str]],  # (index, collection_name)
        *,
        distance_space: str = "cosine",  # "cosine" | "l2" | "ip"
        oversample_dense: int = 2,
    ) -> None:
        self.indexes = list(indexes)
        self.distance_space = distance_space.lower()
        self.oversample_dense = max(1, int(oversample_dense))

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        n = k * self.oversample_dense
        items: List[Dict[str, Any]] = []

        for index, col_name in self.indexes:
            hits = index.query(
                query_text=query,
                k=n,
                where=where,
                where_document=where_document,
            )
            for h in hits:
                # Convert distance to similarity (higher is better)
                sim = self._distance_to_similarity(h.get("score"))
                meta = h.get("metadata") or {}
                if isinstance(meta, dict):
                    meta = dict(meta)
                    meta.setdefault("_collection", col_name)
                items.append({
                    "id": str(h.get("id")),
                    "text": h.get("text", ""),
                    "metadata": meta,
                    "score_dense": sim,
                    "source": "dense",
                })

        # Min-max normalize across all collected items to produce final fused score
        dense_vals = [x["score_dense"] for x in items if x["score_dense"] is not None]
        vmin, vmax = _min_max(dense_vals)
        for x in items:
            nd = _min_max_norm(x["score_dense"], vmin, vmax)
            x["score"] = float(nd) if nd is not None else 0.0

        items.sort(key=lambda r: r["score"], reverse=True)
        return items[:k]

    # --------------------------- helpers ----------------------------------
    def _distance_to_similarity(self, dist: Optional[float]) -> Optional[float]:
        if dist is None:
            return None
        d = float(dist)
        if self.distance_space == "cosine":
            return 1.0 - d
        if self.distance_space == "l2":
            return -d
        if self.distance_space == "ip":
            return d
        return -d


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
        return 1.0
    return (v - vmin) / (vmax - vmin)

