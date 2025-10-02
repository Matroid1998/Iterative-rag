# Advanced Analysis Plots

This directory contains advanced visualization scripts that reveal complex patterns in RAG system performance through multi-dimensional analysis.

## Overview

These plots go beyond basic metrics to explore:
- **Temporal dynamics**: How query quality evolves step-by-step
- **Multi-dimensional profiles**: Comprehensive model comparisons across 7 dimensions
- **Scaling effects**: How task complexity affects failure modes
- **Efficiency analysis**: Relationship between steps taken and retrieval effectiveness

## Plots

### 1. Step-by-Step Error Evolution (`1_step_error_evolution.py`)
**Type**: Alluvial/flow diagram  
**Purpose**: Visualize how query quality categories flow and transform from step 1 → step 2 → step 3

**Categories tracked**:
- `clean`: Well-formed, focused query
- `anchored`: Uses previous context
- `compound`: Multiple sub-questions
- `poor`: Vague or over-broad
- `off_topic`: Not relevant to task
- `done`: No more steps

**Key insights**:
- Identifies common degradation paths (e.g., clean → compound → poor)
- Shows recovery patterns (poor → anchored)
- Reveals stable vs volatile query quality trajectories

### 2. Model Comparison Radar Chart (`2_model_comparison_radar.py`)
**Type**: Multi-axis radar/spider chart  
**Purpose**: Compare models across 7 performance dimensions simultaneously

**Dimensions** (all normalized 0-100%):
1. **Accuracy**: From CSV results files
2. **Specificity**: Query precision (from quality judgments)
3. **On-Topic Rate**: % runs without off-topic queries
4. **Sufficiency**: % runs with sufficiency_score ≥ 0.6
5. **Coverage Rate**: % runs without coverage gaps
6. **Calibration Rate**: % runs with correct confidence (no miscalibration)
7. **Avg Steps (inverted)**: Fewer steps = higher score (efficiency)

**Key insights**:
- Visual model "fingerprints" showing strength/weakness profiles
- Trade-off visualization (e.g., high accuracy but low efficiency)
- Identifies balanced vs specialized models

**Data source**: Merges judgments + `/media/torontoai/Iterative-rag/src/results/new_results_csv/*.csv`

### 3. Hop Count Effects (`3_hop_count_effects.py`)
**Type**: Multi-line plot with sample size bars  
**Purpose**: Show how failure rates scale with task complexity (number of logical hops)

**Metrics tracked by hop count**:
- Miscalibration rate
- Late hit rate (retrieval delays)
- Composition failure rate
- Coverage gap rate

**Key insights**:
- Which failure modes are hop-sensitive vs hop-independent
- Identifies complexity thresholds where failures spike
- Validates whether systems scale gracefully with difficulty

**Analysis includes**: Statistical trends, most sensitive metrics, percentage point changes

### 4. Steps Per Run Distribution (`4_steps_per_run.py`)
**Type**: Histograms with dual-axis overlay (per model)  
**Purpose**: Analyze step count distribution and retrieval efficiency

**Primary axis**: Histogram of step counts per run
- Color-coded: Green (≤2 steps), Orange (3 steps), Red (≥4 steps)
- Shows mode, mean, median

**Secondary axis**: Average retrieval delay by step count
- Delay = `first_hit_step - hop_index` (how late information arrives)
- Shows correlation between steps taken and retrieval timing

**Key insights**:
- Step count modal patterns (do models converge to 2-3 steps?)
- Retrieval efficiency: Does taking more steps mean better timing?
- Model efficiency ranking (combined score: steps + weighted delay)

## Usage

### Run all plots:
```bash
source .venv/bin/activate
python3 src/rag_analysis/advanced_plots/run_all_plots.py
```

### Run individual plot:
```bash
source .venv/bin/activate
python3 src/rag_analysis/advanced_plots/1_step_error_evolution.py
```

## Output

All plots saved as high-resolution PNGs (300 DPI):
- `1_step_error_evolution.png`
- `2_model_comparison_radar.png`
- `3_hop_count_effects.png`
- `4_steps_per_run.png`

## Data Requirements

**Input files**:
- `src/rag_analysis/output/*coverage_gap_judgments.jsonl`
- `src/rag_analysis/output/*quality_judement.jsonl`
- `src/rag_analysis/output/*hallucination_judgment.jsonl`
- `src/results/new_results_csv/*.csv` (for accuracy data)

**Merge strategy**: All three judgment types joined by `(model, question)` tuple

## Dependencies

- Python 3.8+
- matplotlib
- numpy
- Standard library: json, pathlib, collections, csv

Install: Already in project's `.venv` environment

## Technical Notes

### Alluvial Plot Implementation
- Custom flow visualization (not using matplotlib.sankey)
- Proportional height rectangles for each category
- Flow lines connect categories across steps
- Line thickness proportional to transition count

### Radar Chart Normalization
- All metrics scaled to 0-100% range
- Steps inverted: `((max_steps - actual) / max_steps) * 100`
- Ensures consistency: higher values = better performance

### Hop Count Thresholds
- Only includes hop counts with n ≥ 10 samples
- Prevents statistical noise from rare hop counts

### Retrieval Delay Calculation
- Extracted from `late_hit_per_hop.per_hop[]` in coverage judgments
- Averaged across all hops in a run
- Only includes hops that were eventually hit

## Key Questions Answered

1. **Do queries degrade or improve over time?** → Plot 1
2. **Which model is best overall?** → Plot 2 (depends on priorities)
3. **Do complex tasks break the system?** → Plot 3
4. **Is taking more steps wasteful or beneficial?** → Plot 4

## Related Analysis

- **Hallucination plots**: Focus on confidence and unsupported claims
- **Cross-system plots**: Show cascading failures across components
- **Advanced plots** (this directory): Multi-dimensional and temporal dynamics
