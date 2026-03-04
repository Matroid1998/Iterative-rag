"""
plot1_gc_accuracy_by_token_bin.py

Plot 1: Impact of Gold Context length on Gold Context accuracy.

Groups questions into token-count bins and shows the mean accuracy ±1 std
per bin as a bar chart. Directly answers: "as context length grows, does
accuracy go up or down?"

Run from the src/ directory:
    python gold_context_analysis/plot1_gc_accuracy_by_token_bin.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.parent
QA_JSON      = SRC_DIR / "docs" / "chemrxiv_qa.json"
GC_DIR       = SRC_DIR / "response-jsonl-with-context"
PLOT_DIR     = Path(__file__).parent / "plots"
TOKENIZER_NAME = "bert-base-uncased"

# Bin edges (token counts). Adjust to taste.
# Wide range to cover multi-hop questions (up to 4 passages concatenated).
BIN_EDGES = [0, 150, 300, 450, 600, 900, 1200, 2000]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_gold_contexts(qa_path: Path) -> dict[str, str]:
    """Concatenate ALL passage texts in 'path' (multi-hop gold context)."""
    with open(qa_path) as f:
        data = json.load(f)
    return {
        item["q"].strip(): " ".join(p["text"] for p in item["path"] if p.get("text"))
        for item in data
        if item.get("q") and item.get("path")
    }


def load_gc_correctness(gc_dir: Path) -> dict[str, dict[str, bool]]:
    """Return {question: {model_stem: is_correct}}."""
    results: dict[str, dict[str, bool]] = {}
    for jsonl_file in sorted(gc_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (rec.get("raw", {}).get("question") or "").strip()
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

    print("Loading Gold Context model responses …")
    gc_results = load_gc_correctness(GC_DIR)

    # Build (token_count, avg_accuracy_%) pairs
    token_counts   = []
    avg_accuracies = []

    for question, model_dict in gc_results.items():
        ctx = gold_contexts.get(question)
        if ctx is None or not model_dict:
            continue
        n = len(tokenizer(ctx, add_special_tokens=False)["input_ids"])
        acc = np.mean(list(model_dict.values())) * 100
        token_counts.append(n)
        avg_accuracies.append(acc)

    token_counts   = np.array(token_counts)
    avg_accuracies = np.array(avg_accuracies)

    # Bin
    bin_labels = []
    bin_means  = []
    bin_stds   = []
    bin_counts = []

    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (token_counts >= lo) & (token_counts < hi)
        vals = avg_accuracies[mask]
        if len(vals) == 0:
            continue
        bin_labels.append(f"{lo}–{hi}")
        bin_means.append(vals.mean())
        bin_stds.append(vals.std())
        bin_counts.append(len(vals))

    # ── Plot ──────────────────────────────────────────────────────────────────
    x = np.arange(len(bin_labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    # Color palette: light blue (near white) → dark blue based on n
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "n_blue", ["#D6E4FF", "#08306B"]
    )
    norm = mcolors.Normalize(vmin=min(bin_counts), vmax=max(bin_counts))
    bar_colors = [cmap(norm(n)) for n in bin_counts]

    bars = ax.bar(
        x, bin_means,
        yerr=bin_stds,
        capsize=5,
        color=bar_colors,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.6,
        error_kw=dict(ecolor="#1E3A8A", elinewidth=1.5),
    )

    # X-axis: token bin label + n below it
    combined_labels = [f"{lbl}\nn={n}" for lbl, n in zip(bin_labels, bin_counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(combined_labels, fontsize=11)
    ax.set_xlabel("Gold Context Token Count", fontsize=12)
    ax.set_ylabel("Average Accuracy Across Models (%)", fontsize=12)
    ax.set_title(
        "Effect of Gold Context Length on Accuracy\n"
        "(mean ± 1 std per token-count bin, across all questions and 11 models)",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 110)
    ax.axhline(avg_accuracies.mean(), color="#DC2626", linestyle="--",
               linewidth=1.4, label=f"Overall mean: {avg_accuracies.mean():.1f}%")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Colorbar legend for n values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("n (sample size)", fontsize=11)

    plt.tight_layout()
    out = PLOT_DIR / "plot1_gc_accuracy_by_token_bin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
