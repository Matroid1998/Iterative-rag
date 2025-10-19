from typing import Iterable, Optional
import numpy as np
import torch
import os

from sentence_transformers import SentenceTransformer
from protocols.embedding_config import EmbedderConfig 


class HFEmbedder:
    """Thin wrapper around SentenceTransformer for dense embeddings (defaults to BASF-AI/ChEmbed)."""

    def __init__(self, cfg: EmbedderConfig = EmbedderConfig()):
        self.cfg = cfg

        # Prepare model_kwargs to avoid PyTorch meta tensor issues
        model_kwargs = cfg.model_kwargs or {}
        # Explicitly disable device_map and low_cpu_mem_usage to prevent meta tensor initialization
        model_kwargs["device_map"] = None
        model_kwargs["low_cpu_mem_usage"] = False  # This can cause meta tensor issues
        
        # Force CPU device to avoid CUDA OOM and meta tensor issues
        target_device = "cpu"
        
        # Set environment variable to prevent accelerate from using meta device
        os.environ["ACCELERATE_USE_CPU"] = "1"
        
        try:
            # Load model with explicit backend and device settings
            self.model = SentenceTransformer(
                cfg.model_name,
                device=target_device,
                backend="torch",  # Explicitly use torch backend
                prompts=cfg.prompts,
                default_prompt_name=cfg.default_prompt_name,
                trust_remote_code=cfg.trust_remote_code,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=(cfg.tokenizer_kwargs or {}),
            )
        except TypeError:
            # Fallback for older sentence-transformers versions without backend parameter
            try:
                self.model = SentenceTransformer(
                    cfg.model_name,
                    device=target_device,
                    prompts=cfg.prompts,
                    default_prompt_name=cfg.default_prompt_name,
                    trust_remote_code=cfg.trust_remote_code,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=(cfg.tokenizer_kwargs or {}),
                )
            except:
                # Minimal fallback
                self.model = SentenceTransformer(
                    cfg.model_name,
                    device=target_device,
                    trust_remote_code=cfg.trust_remote_code,
                )

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        dimension = self.model.get_sentence_embedding_dimension()
        return dimension if dimension is not None else 0

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
