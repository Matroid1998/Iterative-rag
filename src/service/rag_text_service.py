"""
RAG Text Service
- Binds: Embedder + Vector Index (Chroma) + TextRetriever + Orchestrator
- Adds convenience methods to ingest corpora (folders / JSONL) with chunking & normalization
- Focuses on unstructured text (KG support can be added later)

Dependencies in this repo:
  protocols.embedding_config.EmbedderConfig
  repo.embeddings.hf_embedder.HFEmbedder
  repo.index.chroma_index.ChromaTextIndex
  repo.retrievers.text_retriever.TextRetriever
  repo.planning.planner_iface.{Planner, JSONPlanner, RuleBasedPlanner}
  service.rag.orchestrator.Orchestrator
  repo.utils.{chunking, normalize, io}

No external LLM client is required to run: by default we use RuleBasedPlanner.
Swap in JSONPlanner with your own LLMClient to make the planner model-driven.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

# Protocols / configs
from protocols.embedding_config import EmbedderConfig

# Repo components
from repo.embeddings.embeddeing_models import HFEmbedder
from repo.index.chroma_index import ChromaTextIndex
from repo.retrievers.text_retriever import TextRetriever

# Planning + orchestrator
from repo.planning.planner_iface import Planner, JSONPlanner, RuleBasedPlanner
from config.app_config import build_json_planner_from_settings
from service.orchestrator import Orchestrator

# Utilities
from repo.utils.chunking import chunk_document, chunk_corpus
from repo.utils.normalize import normalize_text
from repo.utils.io import (
    iter_text_folder,
    load_corpus_from_jsonl,
    save_texts_as_jsonl,
)


class RagTextService:
    """
    High-level entry point for text-only iterative RAG.

    Typical usage:
        svc = RagTextService(persist_path="./chroma_store", collection_name="chem_corpus")
        svc.add_from_folder("./docs", strip_html=True)
        result = svc.answer("Which compound is aromatic?")
        print(result["answer"])
    """

    def __init__(
        self,
        *,
        persist_path: str,
        collection_name: str,
        embedder_cfg: Optional[EmbedderConfig] = None,
        planner: Optional[Planner] = None,
        composer: Optional[Any] = None,          # callable(question, evidence)->dict with {"answer":...}
        distance: str = "cosine",                # "cosine" | "l2" | "ip"
        lexical_backend: Optional[Any] = None,   # optional BM25/OpenSearch adapter (Protocol-compatible)
        retriever_dense_weight: float = 0.7,
        max_steps: int = 6,
    ) -> None:
        # 1) Embedder
        self.embedder_cfg = embedder_cfg or EmbedderConfig()
        self.embedder = HFEmbedder(self.embedder_cfg)

        # 2) Vector index (Chroma)
        self.index = ChromaTextIndex(
            persist_path=persist_path,
            collection_name=collection_name,
            embedder=self.embedder,
            distance=distance,
        )

        # 3) Retriever (hybrid-ready; lexical optional)
        self.retriever = TextRetriever(
            index=self.index,
            distance_space=distance.lower(),
            lexical=lexical_backend,
            dense_weight=retriever_dense_weight,
        )

        # 4) Planner
        if planner is not None:
            self.planner: Planner = planner
        else:
            # Try to build an LLM-backed JSON planner from config; fallback to rule-based
            _p = build_json_planner_from_settings()
            self.planner = _p if _p is not None else RuleBasedPlanner(default_k=8)

        # 5) Orchestrator
        self.orchestrator = Orchestrator(
            planner=self.planner,
            text_retriever=self.retriever,
            compose_answer=composer,   # may be None → uses orchestrator default
            max_steps=max_steps,
            dedupe=True,
            keep_top_evidence=None,
        )

    # -------------------------------------------------------------------------
    # Ingestion helpers (folder / JSONL / in-memory)

    def add_from_folder(
        self,
        root: str,
        *,
        recursive: bool = True,
        include_html: bool = True,
        strip_html: bool = False,
        normalize: bool = True,
        # chunking (word-based defaults)
        words_per_chunk: int = 220,
        words_overlap: int = 50,
        # token-based (if you pass a tokenizer later via kwargs)
        tokenizer: Optional[Any] = None,
        tokens_per_chunk: int = 512,
        tokens_overlap: int = 64,
        chunk_strategy: str = "auto",  # "auto" | "words" | "tokens"
    ) -> List[str]:
        """
        Scan a folder for .txt/.md/.html, normalize & chunk, then add to the index.
        Returns list of chunk IDs inserted.
        """
        docs = list(
            iter_text_folder(
                root,
                recursive=recursive,
                include_html=include_html,
                strip_html=strip_html,
            )
        )

        # Optional text normalization
        if normalize:
            for d in docs:
                d["text"] = normalize_text(d["text"])

        return self._add_docs(
            docs=docs,
            chunk_kwargs=dict(
                strategy=chunk_strategy,
                words_per_chunk=words_per_chunk,
                words_overlap=words_overlap,
                tokenizer=tokenizer,
                tokens_per_chunk=tokens_per_chunk,
                tokens_overlap=tokens_overlap,
            ),
        )

    def add_from_jsonl(
        self,
        path: str,
        *,
        assume_chunked: bool = False,
        text_key: str = "text",
        id_key: str = "id",
        metadata_key: str = "metadata",
        normalize: bool = False,
        # chunking defaults if not pre-chunked
        words_per_chunk: int = 220,
        words_overlap: int = 50,
        tokenizer: Optional[Any] = None,
        tokens_per_chunk: int = 512,
        tokens_overlap: int = 64,
        chunk_strategy: str = "auto",
    ) -> List[str]:
        """
        Ingest JSONL.
        - If assume_chunked=True: expects records like {"id","text","metadata"} and inserts directly.
        - Else: expects {"doc_id","text","title"?,"metadata"?} and performs chunking.
        Returns list of chunk IDs inserted.
        """
        ids_added: List[str] = []

        if assume_chunked:
            from repo.utils.io import iter_jsonl
            batch_texts: List[str] = []
            batch_metas: List[Dict[str, Any]] = []
            batch_ids: List[str] = []

            for rec in iter_jsonl(path):
                txt = rec.get(text_key, "")
                if normalize:
                    txt = normalize_text(txt)
                batch_texts.append(txt)
                batch_metas.append(rec.get(metadata_key) or {})
                batch_ids.append(str(rec.get(id_key)))
                if len(batch_texts) >= 2048:
                    ids_added.extend(self.index.add_documents(batch_texts, batch_metas, batch_ids))
                    batch_texts, batch_metas, batch_ids = [], [], []

            if batch_texts:
                ids_added.extend(self.index.add_documents(batch_texts, batch_metas, batch_ids))
            return ids_added

        # Not pre-chunked → load doc-level records then chunk
        docs = load_corpus_from_jsonl(path)
        if normalize:
            for d in docs:
                d["text"] = normalize_text(d["text"])

        return self._add_docs(
            docs=docs,
            chunk_kwargs=dict(
                strategy=chunk_strategy,
                words_per_chunk=words_per_chunk,
                words_overlap=words_overlap,
                tokenizer=tokenizer,
                tokens_per_chunk=tokens_per_chunk,
                tokens_overlap=tokens_overlap,
            ),
        )

    def add_raw_documents(
        self,
        docs: Iterable[Dict[str, Any]],
        *,
        normalize: bool = True,
        words_per_chunk: int = 220,
        words_overlap: int = 50,
        tokenizer: Optional[Any] = None,
        tokens_per_chunk: int = 512,
        tokens_overlap: int = 64,
        chunk_strategy: str = "auto",
    ) -> List[str]:
        """
        Ingest an iterable of {"doc_id","text","title"?,"metadata"?} in memory.
        Returns list of chunk IDs inserted.
        """
        docs = list(docs)
        if normalize:
            for d in docs:
                d["text"] = normalize_text(d["text"])

        return self._add_docs(
            docs=docs,
            chunk_kwargs=dict(
                strategy=chunk_strategy,
                words_per_chunk=words_per_chunk,
                words_overlap=words_overlap,
                tokenizer=tokenizer,
                tokens_per_chunk=tokens_per_chunk,
                tokens_overlap=tokens_overlap,
            ),
        )

    # -------------------------------------------------------------------------
    # Retrieval & QA

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense (or hybrid) retrieval over the index; returns hit dicts."""
        return self.retriever.retrieve(
            query=query,
            k=k,
            where=where,
            where_document=where_document,
        )

    def answer(
        self,
        question: str,
        *,
        initial_state: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        k_default: int = 8,
        trace: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full iterative loop and return:
          {"question","answer","evidence","citations","actions_trace","steps","stop_reason","elapsed_sec"}
        """
        return self.orchestrator.run(
            question=question,
            initial_state=initial_state,
            filters=filters,
            k_default=k_default,
            trace=trace,
        )

    # -------------------------------------------------------------------------
    # Admin

    def count(self) -> int:
        """Number of chunks in the collection."""
        return self.index.count()

    # -------------------------------------------------------------------------
    # Internal helpers

    def _add_docs(self, docs: List[Dict[str, Any]], *, chunk_kwargs: Dict[str, Any]) -> List[str]:
        """Chunk documents, push into index, return list of chunk IDs inserted."""
        texts: List[str]
        metas: List[Dict[str, Any]]
        ids: List[str]
        texts, metas, ids = chunk_corpus(docs, **chunk_kwargs)
        return self.index.add_documents(texts=texts, metadatas=metas, ids=ids)
