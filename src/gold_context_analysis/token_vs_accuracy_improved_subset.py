"""
token_vs_accuracy_improved_subset.py

Same scatter plot as token_vs_accuracy_scatter.py, but filtered to only
the questions that satisfy BOTH conditions (per model):
  - answered INCORRECTLY in Gold Context  (response-jsonl-with-context/)
  - answered CORRECTLY   in Iterative RAG (responses_reverified/)

A question is included in the plot if AT LEAST ONE model shows this
improvement. The y-axis shows the fraction of models that improved on
that question (i.e. wrong→correct), rather than overall accuracy.

Run from the src/ directory:
    python gold_context_analysis/token_vs_accuracy_improved_subset.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.parent
QA_JSON      = SRC_DIR / "docs" / "chemrxiv_qa.json"
GC_DIR       = SRC_DIR / "response-jsonl-with-context"   # gold context
RAG_DIR      = SRC_DIR / "responses_reverified"           # iterative RAG
PLOT_DIR     = Path(__file__).parent / "plots"

TOKENIZER_NAME = "bert-base-uncased"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_gold_contexts(qa_path: Path) -> dict[str, str]:
    """Return {question_text: gold_context_text}."""
    with open(qa_path) as f:
        data = json.load(f)
    mapping: dict[str, str] = {}
    for item in data:
        q    = item.get("q", "").strip()
        path = item.get("path", [])
        if q and path:
            mapping[q] = path[0].get("text", "")
    return mapping


def load_jsonl_correctness(directory: Path, question_key: str) -> dict[str, dict[str, bool]]:
    """
    Return {question_text: {model_stem: is_correct}} for all *.jsonl files
    in `directory`.  `question_key` is the dot-path to the question string,
    e.g. "raw.question" or "raw_response.question".
    """
    results: dict[str, dict[str, bool]] = {}
    keys = question_key.split(".")

    for jsonl_file in sorted(directory.glob("*.jsonl")):
        model_key = jsonl_file.stem
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Navigate nested key path
                node = record
                for k in keys:
                    node = node.get(k, {}) if isinstance(node, dict) else {}
                question   = (node or "").strip() if isinstance(node, str) else ""
                is_correct = bool(record.get("is_correct", False))
                if question:
                    results.setdefault(question, {})[model_key] = is_correct

    return results


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Loading gold contexts …")
    gold_contexts = load_gold_contexts(QA_JSON)

    print(f"Loading Gold Context responses …")
    gc_results = load_jsonl_correctness(GC_DIR, "raw.question")

    print(f"Loading Iterative RAG responses …")
    rag_results = load_jsonl_correctness(RAG_DIR, "raw_response.question")

    # For each question, count how many models show wrong→correct improvement
    token_counts:      list[int]   = []
    improvement_rates: list[float] = []
    skipped = 0

    all_questions = set(gc_results) | set(rag_results)

    for question in sorted(all_questions):
        context = gold_contexts.get(question)
        if context is None:
            skipped += 1
            continue

        gc_row  = gc_results.get(question, {})
        rag_row = rag_results.get(question, {})

        # Models present in both
        common_models = set(
            # match by base model name: strip "_reverified" suffix differences
            # Both dirs share the same stems → intersect directly
        )
        # Build a set of shared model stems using a flexible match:
        # GC stems look like: responses_openai_gpt-4o_reverified
        # RAG stems look like: responses_openai_gpt-4o_reverified (same)
        common_models = set(gc_row.keys()) & set(rag_row.keys())

        if not common_models:
            skipped += 1
            continue

        # Count models where: GC wrong AND RAG correct
        improved = sum(
            1
            for m in common_models
            if not gc_row[m] and rag_row[m]
        )

        if improved == 0:
            continue  # this question shows no improvement in any model → exclude

        n_tokens        = count_tokens(tokenizer, context)
        improvement_pct = improved / len(common_models) * 100

        token_counts.append(n_tokens)
        improvement_rates.append(improvement_pct)

    print(f"  {len(token_counts)} questions with at least one model improving; {skipped} skipped.")

    token_counts      = np.array(token_counts)
    improvement_rates = np.array(improvement_rates)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        token_counts,
        improvement_rates,
        alpha=0.55,
        s=35,
        color="#059669",
        edgecolors="white",
        linewidths=0.4,
    )

    # Trend line
    coeffs = np.polyfit(token_counts, improvement_rates, 1)
    x_line = np.linspace(token_counts.min(), token_counts.max(), 300)
    ax.plot(
        x_line,
        np.polyval(coeffs, x_line),
        color="#DC2626",
        linewidth=1.8,
        linestyle="--",
        label=f"Trend  (slope {coeffs[0]:+.3f} %/token)",
    )

    ax.set_xlabel("Gold Context Token Count (BERT tokenizer)", fontsize=13)
    ax.set_ylabel("Fraction of Models Improved (%)\n(wrong in Gold Context → correct in Iterative RAG)", fontsize=11)
    ax.set_title(
        "Gold Context Length vs. Model Improvement Rate\n"
        "(questions answered incorrectly in Gold Context\n"
        "but correctly in Iterative RAG)",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    stats_text = (
        f"n = {len(token_counts)}\n"
        f"Token range: [{token_counts.min()}–{token_counts.max()}]\n"
        f"Avg improvement rate: {improvement_rates.mean():.1f}%"
    )
    ax.text(
        0.98, 0.04, stats_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#9CA3AF", alpha=0.85),
    )

    plt.tight_layout()
    out_path = PLOT_DIR / "token_count_vs_improvement_rate.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
