# pip install chromadb

from typing import List, Dict, Optional, Any
import uuid

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Metadata

from repo.embeddings.embeddeing_models import HFEmbedder


class STEmbeddingFunction(EmbeddingFunction):
    """Adapter so Chroma can call our SentenceTransformers embedder."""
    def __init__(self, embedder: HFEmbedder, prompt_name: Optional[str] = None):
        self.embedder = embedder
        self.prompt_name = prompt_name

    def __call__(self, inputs: Documents) -> Embeddings:
        vecs = self.embedder.encode(inputs, prompt_name=self.prompt_name)
        return vecs.tolist()


class ChromaTextIndex:
    """
    Lightweight text index on top of ChromaDB.
    - Stores text chunks as documents with metadata.
    - Uses your HFEmbedder to produce embeddings.
    - Cosine similarity by default (works with normalized vectors).
    """

    def __init__(
        self,
        persist_path: str,
        collection_name: str,
        embedder: HFEmbedder,
        distance: str = "cosine",   # "cosine" | "l2" | "ip"
        prompt_name: Optional[str] = None,
        collection_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        persist_path: directory for Chroma (DuckDB/Parquet files).
        collection_name: unique name for your corpus.
        embedder: instance of HFEmbedder.
        distance: metric to use; keep "cosine" if you normalize embeddings.
        prompt_name: optional ST prompt name to use for both add/query.
        """
        self.client = chromadb.PersistentClient(path=persist_path)

        # HNSW distance is configured via collection metadata in Chroma.
        meta = collection_metadata.copy() if collection_metadata else {}
        meta.setdefault("hnsw:space", distance)

        self.embedding_fn = STEmbeddingFunction(embedder, prompt_name=prompt_name)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata=meta,
        )

    # ---------- Write ops ----------

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Adds (or upserts) text chunks.
        - texts: list of raw chunk strings
        - metadatas: optional list of dicts (same length)
        - ids: optional stable string IDs; auto-generated if None
        Returns the IDs used.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Chroma will call our embedding function automatically
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
        return ids

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()

    # ---------- Read ops ----------

    def query(
        self,
        query_text: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of hits with: id, score (distance), text, metadata.
        You can filter with `where` (on metadata) and `where_document` (on text).
        """
        res = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where,
            where_document=where_document,
            include=["distances", "metadatas", "documents"],
        )
        # Chroma returns lists per query; we sent 1 query -> take [0]
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for i, id_ in enumerate(ids):
            hits.append({
                "id": id_,
                "score": float(dists[i]) if i < len(dists) else None,
                "text": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else None,
            })
        return hits

    def query_batch(
        self,
        queries: List[str],
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Batch query convenience."""
        res = self.collection.query(
            query_texts=queries,
            n_results=k,
            where=where,
            where_document=where_document,
            include=["distances", "metadatas", "documents"],
        )
        out: List[List[Dict[str, Any]]] = []
        for qi in range(len(queries)):
            ids = res.get("ids", [[]])[qi]
            dists = res.get("distances", [[]])[qi]
            docs = res.get("documents", [[]])[qi]
            metas = res.get("metadatas", [[]])[qi]
            hits: List[Dict[str, Any]] = []
            for i, id_ in enumerate(ids):
                hits.append({
                    "id": id_,
                    "score": float(dists[i]) if i < len(dists) else None,
                    "text": docs[i] if i < len(docs) else None,
                    "metadata": metas[i] if i < len(metas) else None,
                })
            out.append(hits)
        return out