"""
plot3_token_count_by_group.py

For each of the 11 models, questions are classified into two groups:
  • "Improved"     : wrong in Gold Context AND correct in Iterative RAG
  • "Both Correct" : correct in Gold Context AND correct in Iterative RAG

Y-axis in each panel: mean gold-context token count for that group.

Output:
  plots/plot3a_per_model_token_bars.png  – one subplot per model (11 panels)
  plots/plot3b_summary_token_bars.png    – averaged across models ± std

Run from the src/ directory:
    python gold_context_analysis/plot3_token_count_by_group.py
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
RAG_DIR  = SRC_DIR / "responses_reverified"
PLOT_DIR = Path(__file__).parent / "plots"
TOKENIZER_NAME = "bert-base-uncased"

# Friendly short model labels
MODEL_LABELS = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified":           "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified": "Claude 3.7 Thinking",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified":           "Claude 3.7 Sonnet",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified":             "DeepSeek R1",
    "responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified":        "Llama 3.3 70B",
    "responses_openai_gpt-4o_reverified":                                      "GPT-4o",
    "responses_openai_gpt-5_reverified":                                       "GPT-5",
    "responses_openrouter_anthropic_claude_sonnet_4_5_reasoning":              "Claude Sonnet 4.5",
    "responses_openrouter_google__gemini-2.5-pro_reverified":                  "Gemini 2.5 Pro",
    "responses_openrouter_x-ai__grok-4-fast_reverified":                       "Grok 4 Fast",
    "responses_openrouter_z-ai__glm-4.6_reverified":                           "GLM-4.6",
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


def load_correctness(directory: Path, q_key_path: str) -> dict[str, dict[str, bool]]:
    """Return {question: {model_stem: is_correct}}."""
    results: dict[str, dict[str, bool]] = {}
    keys = q_key_path.split(".")
    for jsonl_file in sorted(directory.glob("*.jsonl")):
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
                node = rec
                for k in keys:
                    node = node.get(k, {}) if isinstance(node, dict) else {}
                q = (node or "").strip() if isinstance(node, str) else ""
                if q:
                    results.setdefault(q, {})[stem] = bool(rec.get("is_correct", False))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading gold contexts …")
    gold_contexts = load_gold_contexts(QA_JSON)

    # Pre-compute token counts once
    token_cache: dict[str, int] = {
        q: len(tokenizer(ctx, add_special_tokens=False)["input_ids"])
        for q, ctx in gold_contexts.items()
    }

    print("Loading Gold Context responses …")
    gc_results = load_correctness(GC_DIR, "raw.question")

    print("Loading Iterative RAG responses …")
    rag_results = load_correctness(RAG_DIR, "raw_response.question")

    # Explicit mapping: GC stem → RAG stem (accounts for filename differences)
    STEM_PAIRS = [
        # (gc_stem, rag_stem)
        ("responses_bedrock_mistral.mistral-large-2402-v1:0_reverified",
         "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified"),
        ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified",
         "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified"),
        ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified",
         "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified"),
        ("responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified",
         "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified"),
        ("responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified",
         "responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified"),
        ("responses_openai_gpt-4o_reverified",
         "responses_openai_gpt-4o_reverified"),
        ("responses_openai_gpt-5",                              # GC has no _reverified suffix
         "responses_openai_gpt-5_reverified"),
        ("responses_openrouter_anthropic_claude_sonnet_4_5_reasoning",
         "responses_openrouter_anthropic_claude_sonnet_4_5_reasoning"),
        ("responses_openrouter_google__gemini-2.5-pro-reasoning",  # GC has -reasoning
         "responses_openrouter_google__gemini-2.5-pro_reverified"),
        ("responses_openrouter_x-ai__grok-4-fast-reasoning",       # GC has -reasoning
         "responses_openrouter_x-ai__grok-4-fast_reverified"),
        ("responses_openrouter_z-ai__glm-4.6-reasoning_reverified", # GC has -reasoning
         "responses_openrouter_z-ai__glm-4.6_reverified"),
    ]

    per_model: dict[str, dict[str, float]] = {}   # rag_stem → stats

    for gc_stem, rag_stem in STEM_PAIRS:
        imp_toks = []
        bc_toks  = []
        for question, gc_dict in gc_results.items():
            if gc_stem not in gc_dict:
                continue
            rag_dict = rag_results.get(question, {})
            if rag_stem not in rag_dict:
                continue
            tok = token_cache.get(question)
            if tok is None:
                continue
            gc_ok  = gc_dict[gc_stem]
            rag_ok = rag_dict[rag_stem]
            if not gc_ok and rag_ok:
                imp_toks.append(tok)
            elif gc_ok and rag_ok:
                bc_toks.append(tok)

        per_model[rag_stem] = {
            "improved_mean": np.mean(imp_toks) if imp_toks else np.nan,
            "improved_n":    len(imp_toks),
            "bc_mean":       np.mean(bc_toks)  if bc_toks  else np.nan,
            "bc_n":          len(bc_toks),
        }
        label = MODEL_LABELS.get(rag_stem, rag_stem)
        print(f"  {label:30s}  improved n={len(imp_toks):4d}  bc n={len(bc_toks):4d}")

    all_models = list(per_model.keys())


    # ── Plot 3a: per-model subplots ───────────────────────────────────────────
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharey=True)
    axes_flat = axes.flatten()

    for i, model in enumerate(all_models):
        ax    = axes_flat[i]
        stats = per_model[model]
        label = MODEL_LABELS.get(model, model)

        means  = [stats["improved_mean"], stats["bc_mean"]]
        ns     = [stats["improved_n"],    stats["bc_n"]]
        colors = ["#059669", "#2563EB"]
        xlabs  = ["Improved\n(GC→RAG)", "Both\nCorrect"]

        bars = ax.bar([0, 1], means, color=colors, alpha=0.80,
                      edgecolor="white", linewidth=0.5, width=0.55)
        for xi, (mean, n, color) in enumerate(zip(means, ns, colors)):
            if not np.isnan(mean):
                ax.text(xi, mean + 3, f"{mean:.0f}\n(n={n})",
                        ha="center", va="bottom", fontsize=7, color="#111827")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(xlabs, fontsize=8)
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if i % ncols == 0:
            ax.set_ylabel("Avg Token Count", fontsize=8)

    # Hide unused subplot
    for j in range(len(all_models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Gold Context Token Count by Group ─ Per Model\n"
        "(Improved = wrong in GC, correct in RAG  |  Both Correct = correct in both)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out_a = PLOT_DIR / "plot3a_per_model_token_bars.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_a}")
    plt.close(fig)

    # ── Plot 3b: summary across models ───────────────────────────────────────
    imp_means = np.array([per_model[m]["improved_mean"] for m in all_models
                          if not np.isnan(per_model[m]["improved_mean"])])
    bc_means  = np.array([per_model[m]["bc_mean"]       for m in all_models
                          if not np.isnan(per_model[m]["bc_mean"])])

    group_means = [imp_means.mean(), bc_means.mean()]
    group_stds  = [imp_means.std(),  bc_means.std()]
    group_ns    = [len(imp_means),   len(bc_means)]
    xlabs       = ["Wrong in GC\n→ Correct in RAG", "Correct\nin Both"]
    colors      = ["#059669", "#2563EB"]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    bars = ax2.bar(
        [0, 1], group_means,
        yerr=group_stds, capsize=8,
        color=colors, alpha=0.82,
        edgecolor="white", linewidth=0.6,
        error_kw=dict(elinewidth=2, ecolor="#1a1a2e"),
        width=0.45,
    )
    for xi, color in enumerate(colors):
        pass  # no annotations

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(xlabs, fontsize=12)
    ax2.set_ylabel("Mean Gold Context Token Count", fontsize=11)
    ax2.set_title(
        "Gold Context Token Count by Group\n"
        "Averaged Across 11 Models (error bars = std across models)",
        fontsize=11,
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_xlim(-0.5, 1.5)

    plt.tight_layout()
    out_b = PLOT_DIR / "plot3b_summary_token_bars.png"
    fig2.savefig(out_b, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_b}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
