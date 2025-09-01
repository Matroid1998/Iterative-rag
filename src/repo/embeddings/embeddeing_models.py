# repo/embeddings/hf_embedder.py
from typing import Iterable, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from protocols.embedding_config import EmbedderConfig 


class HFEmbedder:
    """Thin wrapper around SentenceTransformer for dense embeddings (defaults to BASF-AI/ChEmbed)."""

    def __init__(self, cfg: EmbedderConfig = EmbedderConfig()):
        self.cfg = cfg

        try:
            self.model = SentenceTransformer(
                cfg.model_name,
                device=cfg.device,
                prompts=cfg.prompts,
                default_prompt_name=cfg.default_prompt_name,
                trust_remote_code=cfg.trust_remote_code,
                model_kwargs=(cfg.model_kwargs or {}),
                tokenizer_kwargs=(cfg.tokenizer_kwargs or {}),
            )
        except TypeError:
            self.model = SentenceTransformer(
                cfg.model_name,
                device=cfg.device,
                trust_remote_code=cfg.trust_remote_code,
            )

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Iterable[str], prompt_name: Optional[str] = None) -> np.ndarray:
        """Encode an iterable of strings into a (N, dim) NumPy array."""
        texts = list(texts)
        try:
            return self.model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                normalize_embeddings=self.cfg.normalize,
                show_progress_bar=self.cfg.show_progress_bar,
                prompt_name=prompt_name,
                convert_to_numpy=True,
            )
        except TypeError:
            return self.model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                normalize_embeddings=self.cfg.normalize,
                show_progress_bar=self.cfg.show_progress_bar,
                convert_to_numpy=True,
            )

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self.encode(queries, prompt_name="query")

    def encode_documents(self, docs: Iterable[str]) -> np.ndarray:
        return self.encode(docs, prompt_name="passage")
