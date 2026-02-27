"""
token_vs_accuracy_scatter.py

Produces a scatter plot where:
  x-axis : number of BERT (bert-base-uncased) tokens in the gold context
            (the `text` field from docs/chemrxiv_qa.json)
  y-axis : average accuracy across the 11 models in
            response-jsonl-with-context/ for that question

Run from the src/ directory:
    python gold_context_analysis/token_vs_accuracy_scatter.py
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
RESPONSE_DIR = SRC_DIR / "response-jsonl-with-context"
PLOT_DIR     = Path(__file__).parent / "plots"

TOKENIZER_NAME = "bert-base-uncased"


# ── 1. Load gold context texts keyed by question ──────────────────────────────
def load_gold_contexts(qa_path: Path) -> dict[str, str]:
    """Return {question_text: gold_context_text} from chemrxiv_qa.json."""
    with open(qa_path) as f:
        data = json.load(f)

    mapping: dict[str, str] = {}
    for item in data:
        question = item.get("q", "").strip()
        paths    = item.get("path", [])
        if paths and question:
            # Concatenate text from ALL path entries (multi-hop = multiple passages)
            text = "\n".join(p.get("text", "") for p in paths)
            mapping[question] = text
    return mapping


# ── 2. Load model responses ───────────────────────────────────────────────────
def load_responses(response_dir: Path) -> dict[str, dict[str, bool]]:
    """
    Return {question_text: {model_file_stem: is_correct}} for every
    *.jsonl file in response_dir.
    """
    results: dict[str, dict[str, bool]] = {}

    for jsonl_file in sorted(response_dir.glob("*.jsonl")):
        model_key = jsonl_file.stem
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                question = (
                    record.get("raw", {}).get("question", "") or ""
                ).strip()
                is_correct = bool(record.get("is_correct", False))
                if question:
                    results.setdefault(question, {})[model_key] = is_correct

    return results


# ── 3. Token count ────────────────────────────────────────────────────────────
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Loading gold contexts from {QA_JSON} …")
    gold_contexts = load_gold_contexts(QA_JSON)
    print(f"  {len(gold_contexts)} unique questions found.")

    print(f"Loading model responses from {RESPONSE_DIR} …")
    responses = load_responses(RESPONSE_DIR)
    model_files = {
        m for q_dict in responses.values() for m in q_dict
    }
    print(f"  {len(model_files)} model files found.")

    # Compute per-question (token_count, avg_accuracy)
    token_counts: list[int]   = []
    avg_accuracies: list[float] = []
    skipped = 0

    for question, model_dict in responses.items():
        context = gold_contexts.get(question)
        if context is None:
            skipped += 1
            continue
        n_tokens   = count_tokens(tokenizer, context)
        avg_acc    = sum(model_dict.values()) / len(model_dict)
        token_counts.append(n_tokens)
        avg_accuracies.append(avg_acc * 100)   # percent

    print(f"  {len(token_counts)} questions matched; {skipped} skipped.")

    token_counts   = np.array(token_counts)
    avg_accuracies = np.array(avg_accuracies)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        token_counts,
        avg_accuracies,
        alpha=0.55,
        s=35,
        color="#2563EB",
        edgecolors="white",
        linewidths=0.4,
    )

    # Trend line
    coeffs = np.polyfit(token_counts, avg_accuracies, 1)
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
    ax.set_ylabel("Average Accuracy Across Models (%)", fontsize=13)
    ax.set_title(
        "Gold Context Length vs. Average Model Accuracy\n"
        "(each point = one question, averaged over 11 models)",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Summary stats annotation
    stats_text = (
        f"n = {len(token_counts)}\n"
        f"Token range: [{token_counts.min()}–{token_counts.max()}]\n"
        f"Avg accuracy: {avg_accuracies.mean():.1f}%"
    )
    ax.text(
        0.98, 0.04, stats_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#9CA3AF", alpha=0.85),
    )

    plt.tight_layout()
    out_path = PLOT_DIR / "token_count_vs_avg_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
