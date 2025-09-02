from typing import Any, Dict, List, Optional, Protocol

Hit = Dict[str, Any]  # keep it simple for now; you can replace with a dataclass later.

class LexicalBackend(Protocol):
    def search(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Hit]:
        """
        Returns hits with keys:
          - "id": str
          - "score": float          (higher is better)
          - "text": str
          - "metadata": dict (optional)
        """
        ...

class VectorTextIndex(Protocol):
    def query(
        self,
        query_text: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Hit]:
        """
        Returns hits with keys:
          - "id": str
          - "score": float          (distance; lower is better for cosine/L2)
          - "text": str
          - "metadata": dict (optional)
        """
        ...
