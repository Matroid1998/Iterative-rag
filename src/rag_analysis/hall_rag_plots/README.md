# Hallucination Analysis Plots

This folder houses visualization scripts for the hallucination audit judgments produced by the iterative RAG evaluation pipeline.

## Plots

1. **Miscalibration Direction by Hop Count (`1_miscalibration_by_hop.py`)**  
   Stacked bar chart of miscalibration directions across question hop counts.

2. **Sufficiency vs Coverage Scatter (`2_sufficiency_vs_coverage.py`)**  
   Scatter plot of sufficiency vs coverage coloured by miscalibration direction, point size = unsupported claims.

3. **Unsupported Claims Distribution (`3_unsupported_claims_distribution.py`)**  
   Histograms of unsupported claim counts per model.

4. **Composition Failure Root Causes (`4_composition_failure_root_causes.py`)**  
   Grouped bars showing composition failures that coincide with coverage gaps, carry drops, late hits, and poor query quality.

5. **Composition Failure Rate (`5_composition_failure_rate.py`)**  
   Bar chart of composition failure percentage per model.

6. **Evidence Sufficiency Distribution (`6_sufficiency_distribution.py`)**  
   Histogram of sufficiency estimates with the 0.6 threshold marked.

7. **Miscalibration Mix per Model (`7_miscalibration_mix.py`)**  
   Stacked bars of confidence directions by model, with overall miscalibration rate annotated.

8. **Coverage vs Confidence Scatter (`8_coverage_vs_confidence.py`)**  
   Complementary scatter (coverage on X, sufficiency on Y) coloured by direction.

All scripts write 300‑DPI PNGs into this directory.

## Usage

```bash
cd /media/torontoai/Iterative-rag/src/rag_analysis/hall_rag_plots
python run_all_plots.py  # executes the full suite
```

Run individual plots by substituting the desired script above.

## Requirements

- Python 3.8+
- `matplotlib`
- `numpy`
- `pandas`

Install once:
```bash
pip install matplotlib numpy pandas
```

## Data Sources

- Hallucination judgments: `/src/rag_analysis/output/*hallucination_judgment.jsonl`
- Coverage judgments (for root-cause correlation): `/src/rag_analysis/output/*coverage_gap_judgments.jsonl`
- Query audit judgments (for poor query flags): `/src/rag_analysis/output/*quality_judement.jsonl`
- System-level accuracy: `/src/results/new_results_csv/*.csv`

Each script skips gracefully if required inputs are missing.

