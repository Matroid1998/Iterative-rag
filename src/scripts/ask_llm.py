import os
from typing import Any

from protocols.embedding_config import EmbedderConfig
from service.rag_text_service import RagTextService


def main() -> None:
    persist_path = os.path.abspath("./chroma_store")
    collection_name = "chem_corpus"

    # Use CPU
    embed_cfg = EmbedderConfig(device="cpu")

    svc = RagTextService(
        persist_path=persist_path,
        collection_name=collection_name,
        embedder_cfg=embed_cfg,
        planner=None,  # let service pick JSON planner from config if available
        composer=None,
        max_steps=6,
    )

    # Two-hop question: requires both updated sample files
    # - chemistry_1.txt → aromatic compound (e.g., Naphthalene)
    # - chemistry_2.txt → pKa of formic acid (≈ 3.75)
    question = "Which compound mentioned is aromatic, and what is the pKa of formic acid?"

    print("Asking:", question)
    result = svc.answer(question, k_default=8, trace=True)
    print("\nAnswer:")
    print(result.get("answer"))

    print("\nCitations:")
    for c in (result.get("citations") or []):
        print(" -", c)

    print("\nActions trace:")
    for a in (result.get("actions_trace") or []):
        print(" -", a)


if __name__ == "__main__":
    main()
