"""
plot2_improved_token_distribution.py

Plot 2: Token length distribution of questions that were answered
incorrectly in Gold Context but correctly in Iterative RAG (≥1 model),
compared to the full question distribution.

Shows whether "improved" questions tend to have shorter or longer
gold contexts than the overall set.

Run from the src/ directory:
    python gold_context_analysis/plot2_improved_token_distribution.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.parent
QA_JSON      = SRC_DIR / "docs" / "chemrxiv_qa.json"
GC_DIR       = SRC_DIR / "response-jsonl-with-context"
RAG_DIR      = SRC_DIR / "responses_reverified"
PLOT_DIR     = Path(__file__).parent / "plots"
TOKENIZER_NAME = "bert-base-uncased"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_gold_contexts(qa_path: Path) -> dict[str, str]:
    with open(qa_path) as f:
        data = json.load(f)
    return {
        item["q"].strip(): item["path"][0]["text"]
        for item in data
        if item.get("q") and item.get("path")
    }


def load_correctness(directory: Path, question_key_path: str) -> dict[str, dict[str, bool]]:
    """Return {question: {model_stem: is_correct}}."""
    results: dict[str, dict[str, bool]] = {}
    keys = question_key_path.split(".")
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                node = rec
                for k in keys:
                    node = node.get(k, {}) if isinstance(node, dict) else {}
                q = (node or "").strip() if isinstance(node, str) else ""
                if q:
                    results.setdefault(q, {})[jsonl_file.stem] = bool(rec.get("is_correct", False))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading gold contexts …")
    gold_contexts = load_gold_contexts(QA_JSON)

    print("Loading Gold Context responses …")
    gc_results = load_correctness(GC_DIR, "raw.question")

    print("Loading Iterative RAG responses …")
    rag_results = load_correctness(RAG_DIR, "raw_response.question")

    # Build token count lists
    all_tokens:      list[int] = []
    improved_tokens: list[int] = []

    for question, gc_dict in gc_results.items():
        ctx = gold_contexts.get(question)
        if ctx is None:
            continue
        n = len(tokenizer(ctx, add_special_tokens=False)["input_ids"])
        all_tokens.append(n)

        rag_dict = rag_results.get(question, {})
        common = set(gc_dict) & set(rag_dict)
        # improved: at least one model went wrong→correct
        if any(not gc_dict[m] and rag_dict[m] for m in common):
            improved_tokens.append(n)

    all_tokens      = np.array(all_tokens)
    improved_tokens = np.array(improved_tokens)

    print(f"  All questions:      {len(all_tokens)}")
    print(f"  Improved questions: {len(improved_tokens)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    bins = np.arange(0, all_tokens.max() + 30, 30)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Draw filled green (improved) first, then the step outline on top
    ax.hist(
        improved_tokens, bins=bins,
        density=True,
        alpha=0.60, color="#059669",
        label=(
            f"Wrong in GC → Correct in RAG  "
            f"(n={len(improved_tokens)},  median={int(np.median(improved_tokens))} tok)"
        ),
    )
    ax.hist(
        all_tokens, bins=bins,
        density=True,
        histtype="step",          # outline only — always visible on top
        linewidth=2.2,
        color="#374151",
        label=f"All questions  (n={len(all_tokens)},  median={int(np.median(all_tokens))} tok)",
    )

    # Vertical median lines
    ax.axvline(np.median(all_tokens), color="#374151", linestyle="--",
               linewidth=1.5, alpha=0.8)
    ax.axvline(np.median(improved_tokens), color="#047857", linestyle="--",
               linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Gold Context Token Count (BERT tokenizer)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Token Length Distribution: All Questions vs. RAG-Improved Questions\n"
        "(questions wrong in Gold Context but correct in Iterative RAG, ≥1 model)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = PLOT_DIR / "plot2_improved_token_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
