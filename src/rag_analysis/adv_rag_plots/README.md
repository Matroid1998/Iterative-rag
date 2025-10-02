# Advanced RAG Analysis Plots

This directory collects higher-level visualisations that combine coverage, quality, and hallucination analytics.

## Plots

1. **Stepwise Error Alluvial (`1_stepwise_error_alluvial.py`)** – Alluvial diagram of query-quality labels from steps 1→3.
2. **Model Radar Profile (`2_model_radar_profile.py`)** – Radar chart comparing accuracy, effort, and evidence metrics per system.
3. **Hop Count Effects (`3_hop_count_effects.py`)** – Line plots of miscalibration, late-hit, and composition failure rates by hop count.
4. **Steps vs Retrieval Efficiency (`4_steps_vs_retrieval_efficiency.py`)** – Box plots of planner depth per model with late-hit delta overlay.

## Usage

```bash
cd /media/torontoai/Iterative-rag/src/rag_analysis/adv_rag_plots
python3 <plot_script>.py
```

Each script reads completed JSONL outputs under `src/rag_analysis/output` and writes a PNG next to the script. They require pandas/matplotlib/numpy (and merge utilities from the other plotting packages).

## Requirements

```bash
pip install matplotlib numpy pandas seaborn
```

(seaborn is optional, only used when available.)
