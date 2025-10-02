# Cross-System RAG Analysis Plots

This directory links coverage-gap, quality-audit, and hallucination judgments to surface end-to-end behaviour. Each plot merges records by identical `(system, question)` keys across the three judgment outputs in `src/rag_analysis/output`.

## Plots

1. **Error Cascade (`1_error_cascade.py`)** – Custom three-stage Sankey illustrating how coverage gaps propagate to query issues and hallucinations for a selected system.
2. **Correctness vs Problem Heatmap (`2_correctness_problem_heatmap.py`)** – Heatmap of failure-mode prevalence among incorrect answers (`has_gap`, `carry_drop`, `late_hit`, `composition_failure`, `miscalibration`).
3. **Efficiency vs Quality (`3_efficiency_quality_tradeoff.py`)** – Scatter of average planner steps vs accuracy by system, marker size proportional to average specificity.
4. **Carry-Drop Accuracy Impact (`4_carry_drop_accuracy.py`)** – Grouped bars showing accuracy for runs with and without anchor carry-drop per system.
5. **Coverage → Hallucination (`5_coverage_to_hallucination.py`)** – Composition-failure rates conditioned on coverage gaps and late hits.
6. **Carry vs Anchoring (`6_carry_vs_anchoring.py`)** – Per-step comparison of coverage carry drops and query anchoring rates.
7. **Planning vs Confidence (`7_planning_vs_confidence.py`)** – Side-by-side bars for per-model logical alignment vs overconfident finalisation rates.

## Usage

```bash
cd /media/torontoai/Iterative-rag/src/rag_analysis/cross_rag_plots
python3 run_all_plots.py
```

Each script also accepts `python3 <script>.py` to generate an individual PNG in this directory. Some scripts expose additional CLI switches (e.g., `1_error_cascade.py --system <name>`).

## Requirements

- Python 3.8+
- `matplotlib`
- `numpy`
- `pandas`
- (optional) `seaborn` for prettier heatmaps

Install dependencies once:
```bash
pip install matplotlib numpy pandas seaborn
```

## Data Dependencies

- Coverage judgments: `/src/rag_analysis/output/*coverage_gap_judgments.jsonl`
- Quality judgments: `/src/rag_analysis/output/*quality_judement.jsonl`
- Hallucination judgments: `/src/rag_analysis/output/*hallucination_judgment.jsonl`
- Model accuracy CSVs: `/src/results/new_results_csv/*.csv`

Plots skip gracefully when the necessary inputs are missing.
