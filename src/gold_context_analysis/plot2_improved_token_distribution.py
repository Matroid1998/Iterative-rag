"""
plot2_improved_token_distribution.py

Plot 2: Token length distribution of questions that were answered
correctly in Gold Context (≥1 model) vs. questions that were wrong in
Gold Context but corrected by Iterative RAG.

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
        item["q"].strip(): "\n".join(p.get("text", "") for p in item["path"])
        for item in data
        if item.get("q") and item.get("path")
    }


def normalize_model_key(stem: str) -> str:
    """Produce a canonical model key from a filename stem."""
    s = stem
    for suffix in ("_reverified", "-reasoning"):
        s = s.replace(suffix, "")
    if s.startswith("responses_"):
        s = s[len("responses_"):]
    return s


def load_correctness(directory: Path, question_key_path: str) -> dict[str, dict[str, bool]]:
    """Return {question: {normalized_model_key: is_correct}}."""
    results: dict[str, dict[str, bool]] = {}
    keys = question_key_path.split(".")
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        model_key = normalize_model_key(jsonl_file.stem)
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
                    results.setdefault(q, {})[model_key] = bool(rec.get("is_correct", False))
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

    # Build token count lists — one entry per (question, model) pair
    gc_correct_tokens: list[int] = []   # each model that got it correct in GC
    improved_tokens:   list[int] = []   # each model that went wrong→correct

    for question, gc_dict in gc_results.items():
        ctx = gold_contexts.get(question)
        if ctx is None:
            continue
        n = len(tokenizer(ctx, add_special_tokens=False)["input_ids"])

        # Correct in Gold Context: one entry per model that got it right
        for model, correct in gc_dict.items():
            if correct:
                gc_correct_tokens.append(n)

        rag_dict = rag_results.get(question, {})
        # Improved: one entry per model that went wrong in GC → correct in RAG
        for model in set(gc_dict) & set(rag_dict):
            if not gc_dict[model] and rag_dict[model]:
                improved_tokens.append(n)

    gc_correct_tokens = np.array(gc_correct_tokens)
    improved_tokens   = np.array(improved_tokens)

    print(f"  Correct in GC:      {len(gc_correct_tokens)}")
    print(f"  Improved questions: {len(improved_tokens)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    bins = np.arange(0, gc_correct_tokens.max() + 30, 30)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Draw filled green (improved) first, then the step outline on top
    ax.hist(
        improved_tokens, bins=bins,
        density=True,
        alpha=0.60, color="#059669",
        label=(
            f"Wrong in GC → Correct in Iterative RAG  "
        ),
    )
    ax.hist(
        gc_correct_tokens, bins=bins,
        density=True,
        histtype="step",          # outline only — always visible on top
        linewidth=2.2,
        color="#374151",
        label=f"Correct in GC",
    )

    ax.set_xlabel("Gold Context Token Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Token Length Distribution: Correct in GC vs. Iterative RAG-Improved Questions\n"
        "(questions wrong in Gold Context but correct in Iterative RAG)",
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
