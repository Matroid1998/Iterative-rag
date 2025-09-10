import os
from typing import Any
import os
print(os.getcwd())
from protocols.embedding_config import EmbedderConfig
from service.rag_text_service import RagTextService


def main() -> None:
    # Configure CPU embedding (set device="cpu")
    embed_cfg = EmbedderConfig(device="cpu")

    persist_path = os.path.abspath("./chroma_store")
    collection_name = "chem_corpus"

    svc = RagTextService(
        persist_path=persist_path,
        collection_name=collection_name,
        embedder_cfg=embed_cfg,
        planner=None,           # keep default RuleBasedPlanner for simple smoke tests
        composer=None,
        distance="cosine",
        lexical_backend=None,
        retriever_dense_weight=0.7,
        max_steps=6,
    )

    docs_dir = os.path.abspath("./sample_docs")
    if not os.path.isdir(docs_dir):
        raise SystemExit(f"sample_docs folder not found: {docs_dir}")

    print(f"Indexing texts from: {docs_dir}")
    ids = svc.add_from_folder(
        docs_dir,
        recursive=False,
        include_html=False,
        strip_html=False,
        normalize=True,
        words_per_chunk=220,
        words_overlap=50,
        chunk_strategy="words",
    )
    print(f"Inserted {len(ids)} chunks into collection '{collection_name}' at {persist_path}")


if __name__ == "__main__":
    main()

