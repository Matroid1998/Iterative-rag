# Quality (Query Audit) Analysis Plots

This folder contains visualization scripts for the query-audit judgments produced by the Iterative RAG evaluation pipeline. The focus is on understanding how query formulation evolves over planner steps, how it correlates with answer correctness, and which failure modes co-occur.

## Available Plots

1. **Query Degradation Over Steps (`1_query_degradation_over_steps.py`)**  
   Multi-panel line chart (one panel per system) tracking the per-step averages of `specificity_score` and `on_topic_score`. Helps reveal whether query quality deteriorates as the planner iterates.

2. **Fusion / Skip Effectiveness (`2_fusion_skip_effectiveness.py`)**  
   Box plots of run accuracy (0/1) for traces that used fusion/skip vs those that did not, grouped by question hop count. Shows whether skipping hops actually helps.

3. **Query Flag Co-occurrence Matrix (`3_query_flag_cooccurrence.py`)**  
   Heatmap of pairwise co-occurrence rates for the main query-quality flags (vague, over-broad, compound, off-topic). Highlights correlated query issues.

4. **Distractor Latch vs Model Performance (`4_distractor_vs_accuracy.py`)**  
   Bar chart of distractor latch rates by system, with a secondary axis overlaying overall model accuracy (ingested from `/src/results/new_results_csv`).

5. **Step Alignment Metrics (`5_step_alignment.py`)**  
   Two-panel chart combining: (a) per-step `is_next_logical_hop` rates by system, and (b) per-model comparison of standard alignment vs exact step=predicted-hop alignment.

6. **Query Flag Composition per Model (`6_query_flags_stacked.py`)**  
   Stacked percentage bars showing how frequently each flag fires for each system.

7. **Score Distributions & Trends (`7_score_distribution_trends.py`)**  
   Box plots of `specificity_score` and `on_topic_score` per system plus an aggregate trend line showing how the average evolves with step depth.

8. **Fusion / Skip Activation by Step (`8_fusion_skip_by_step.py`)**  
   Bar chart of fusion/skip rates per planner step (aggregated across systems) to identify when multi-hop jumps happen.

9. **Stability Indicators (`9_stability_indicators.py`)**  
   Grouped bars showing per-system rates of runs with partial-step contradictions and distractor latch events.

All scripts emit 300‑DPI PNG files into this directory.

## Usage

Run a single plot:
```bash
cd /media/torontoai/Iterative-rag/src/rag_analysis/qual_rag_plots
python 1_query_degradation_over_steps.py
```

Run the whole suite:
```bash
python run_all_plots.py
```

## Requirements

- Python 3.8+
- `matplotlib`
- `numpy`
- `pandas`

Install (once):
```bash
pip install matplotlib numpy pandas
```

## Data Sources

Quality judgments: `/src/rag_analysis/output/*quality_judement.jsonl`  
Coverage judgments (for accuracy join): `/src/rag_analysis/output/*coverage_gap_judgments.jsonl`  
System-level accuracy: `/src/results/new_results_csv/*.csv`

Each script gracefully skips plots if the necessary data is unavailable (e.g., missing coverage file for a system).

## Author

Generated for Iterative RAG quality analysis.
