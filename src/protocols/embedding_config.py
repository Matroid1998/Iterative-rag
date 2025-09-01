from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class EmbedderConfig:
    """Config for SentenceTransformer-based embedders."""
    model_name: str = "BASF-AI/ChEmbed"
    device: Optional[str] = None                
    normalize: bool = True                      
    batch_size: int = 32
    show_progress_bar: bool = False
    prompts: Optional[Dict[str, str]] = None
    default_prompt_name: Optional[str] = None
    trust_remote_code: bool = True
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
