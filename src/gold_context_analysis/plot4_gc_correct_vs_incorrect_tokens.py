"""
plot4_gc_correct_vs_incorrect_tokens.py

Bar chart comparing mean gold-context token counts for two question groups,
defined per model using Gold Context responses only:

  • "Correct in GC"   : model answered correctly  (is_correct = True)
  • "Incorrect in GC" : model answered incorrectly (is_correct = False)

Output:
  plots/plot4a_per_model_gc_bars.png  – one subplot per model (11 panels)
  plots/plot4b_summary_gc_bars.png    – averaged across models ± std

Run from the src/ directory:
    python gold_context_analysis/plot4_gc_correct_vs_incorrect_tokens.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).parent.parent
QA_JSON  = SRC_DIR / "docs" / "chemrxiv_qa.json"
GC_DIR   = SRC_DIR / "response-jsonl-with-context"
PLOT_DIR = Path(__file__).parent / "plots"
TOKENIZER_NAME = "bert-base-uncased"

MODEL_LABELS = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified":                        "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified": "Claude 3.7 Thinking",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified":           "Claude 3.7 Sonnet",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified":                         "DeepSeek R1",
    "responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified":                    "Llama 3.3 70B",
    "responses_openai_gpt-4o_reverified":                                                  "GPT-4o",
    "responses_openai_gpt-5":                                                              "GPT-5",
    "responses_openrouter_anthropic_claude_sonnet_4_5_reasoning":                          "Claude Sonnet 4.5",
    "responses_openrouter_google__gemini-2.5-pro-reasoning":                               "Gemini 2.5 Pro",
    "responses_openrouter_x-ai__grok-4-fast-reasoning":                                   "Grok 4 Fast",
    "responses_openrouter_z-ai__glm-4.6-reasoning_reverified":                            "GLM-4.6",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_gold_contexts(qa_path: Path) -> dict[str, str]:
    with open(qa_path) as f:
        data = json.load(f)
    return {
        item["q"].strip(): item["path"][0]["text"]
        for item in data
        if item.get("q") and item.get("path")
    }


def load_gc_correctness(gc_dir: Path) -> dict[str, dict[str, bool]]:
    """Return {question: {model_stem: is_correct}}."""
    results: dict[str, dict[str, bool]] = {}
    for jsonl_file in sorted(gc_dir.glob("*.jsonl")):
        stem = jsonl_file.stem
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
                    results.setdefault(q, {})[stem] = bool(rec.get("is_correct", False))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading gold contexts …")
    gold_contexts = load_gold_contexts(QA_JSON)
    token_cache = {
        q: len(tokenizer(ctx, add_special_tokens=False)["input_ids"])
        for q, ctx in gold_contexts.items()
    }

    print("Loading Gold Context responses …")
    gc_results = load_gc_correctness(GC_DIR)

    all_stems = sorted(MODEL_LABELS.keys())
    per_model: dict[str, dict] = {}

    for stem in all_stems:
        correct_toks   = []
        incorrect_toks = []
        for question, model_dict in gc_results.items():
            if stem not in model_dict:
                continue
            tok = token_cache.get(question)
            if tok is None:
                continue
            if model_dict[stem]:
                correct_toks.append(tok)
            else:
                incorrect_toks.append(tok)

        per_model[stem] = {
            "correct_mean":   np.mean(correct_toks)   if correct_toks   else np.nan,
            "correct_n":      len(correct_toks),
            "incorrect_mean": np.mean(incorrect_toks) if incorrect_toks else np.nan,
            "incorrect_n":    len(incorrect_toks),
        }
        label = MODEL_LABELS[stem]
        print(f"  {label:30s}  correct n={len(correct_toks):4d}  incorrect n={len(incorrect_toks):4d}")

    # ── Plot 4a: per-model ────────────────────────────────────────────────────
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharey=True)
    axes_flat = axes.flatten()

    for i, stem in enumerate(all_stems):
        ax    = axes_flat[i]
        stats = per_model[stem]
        label = MODEL_LABELS[stem]

        means  = [stats["correct_mean"], stats["incorrect_mean"]]
        ns     = [stats["correct_n"],    stats["incorrect_n"]]
        colors = ["#2563EB", "#DC2626"]
        xlabs  = ["Correct\nin GC", "Incorrect\nin GC"]

        ax.bar([0, 1], means, color=colors, alpha=0.80,
               edgecolor="white", linewidth=0.5, width=0.55)
        for xi, (mean, n) in enumerate(zip(means, ns)):
            if not np.isnan(mean):
                ax.text(xi, mean + 2, f"{mean:.0f}\n(n={n})",
                        ha="center", va="bottom", fontsize=7, color="#111827")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(xlabs, fontsize=8)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if i % ncols == 0:
            ax.set_ylabel("Avg Token Count", fontsize=8)

    for j in range(len(all_stems), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Gold Context Token Count: Correctly vs. Incorrectly Answered Questions ─ Per Model",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_a = PLOT_DIR / "plot4a_per_model_gc_bars.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_a}")
    plt.close(fig)

    # ── Plot 4b: summary ──────────────────────────────────────────────────────
    correct_means   = np.array([per_model[m]["correct_mean"]   for m in all_stems
                                if not np.isnan(per_model[m]["correct_mean"])])
    incorrect_means = np.array([per_model[m]["incorrect_mean"] for m in all_stems
                                if not np.isnan(per_model[m]["incorrect_mean"])])

    group_means = [correct_means.mean(),   incorrect_means.mean()]
    group_stds  = [correct_means.std(),    incorrect_means.std()]
    xlabs       = ["Correct in GC", "Incorrect in GC"]
    colors      = ["#2563EB", "#DC2626"]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.bar(
        [0, 1], group_means,
        yerr=group_stds, capsize=8,
        color=colors, alpha=0.82,
        edgecolor="white", linewidth=0.6,
        error_kw=dict(elinewidth=2, ecolor="#1a1a2e"),
        width=0.45,
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(xlabs, fontsize=12)
    ax2.set_ylabel("Mean Gold Context Token Count", fontsize=11)
    ax2.set_title(
        "Gold Context Token Count:\nCorrect vs. Incorrect Questions in GC Setup\n"
        "Averaged Across 11 Models (error bars = std across models)",
        fontsize=11,
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_xlim(-0.5, 1.5)

    plt.tight_layout()
    out_b = PLOT_DIR / "plot4b_summary_gc_bars.png"
    fig2.savefig(out_b, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_b}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
