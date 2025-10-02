# Coverage Gap Analysis Plots

This directory contains visualization scripts for analyzing coverage gap judgments from the RAG system.

## Overview

These plots analyze three key failure modes identified by the coverage gap judgment system:
1. **Retrieval Coverage Gap**: When the system never retrieves documents needed for a specific hop
2. **Anchor Carry-Drop**: When key entities from previous steps are lost in subsequent queries
3. **Late Hit**: When documents for a hop are retrieved later than they should be

## Plots

### 1. Late Hit Timing Distribution (`1_late_hit_timing_distribution.py`)
**Violin plot** showing the distribution of retrieval delays `(first_hit_step - hop_index)` for each hop.

**Insight**: Reveals how late documents are typically retrieved and whether certain hops (e.g., hop 2) are consistently delayed.

**Output**: `late_hit_timing_distribution.png`

---

### 2. Model Coverage Rates (`2_model_coverage_rates.py`)
**Grouped bar chart** comparing coverage gap and late hit rates across all models.

**Insight**: Identifies which models struggle most with retrieval coverage issues.

**Output**: `model_coverage_rates.png`

---

### 3. Anchor Carry by Step (`3_anchor_carry_by_step.py`)
**Multi-line chart** showing anchor carry-drop rate by step number, with one line per model.

**Insight**: Reveals at which steps models tend to lose track of key entities, and which models are most prone to this issue.

**Output**: `anchor_carry_by_step.png`

---

### 4. Accuracy Linkage (`4_accuracy_linkage.py`)
**Dual bar charts** showing:
- Left: Accuracy rate when each issue type is present
- Right: Prevalence of issues in correct vs incorrect answers

**Insight**: Quantifies how much each coverage issue impacts answer correctness.

**Output**: `accuracy_linkage.png`

---

### 5. Missed Hop Patterns (`5_missed_hop_patterns.py`)
**Stacked bar chart** showing which hops are missed in questions of different complexity (1-hop, 2-hop, etc.).

**Insight**: Reveals whether multi-hop questions systematically miss later hops.

**Output**: `missed_hop_patterns.png`

---

### 6. Anchor Carry Temporal Pattern (`6_anchor_carry_temporal.py`)
**Dual line charts**:
- Top: Aggregated carry-drop rate by step across all models
- Bottom: Breakdown by top 5 models

**Insight**: Shows whether anchor degradation worsens over time and includes trend analysis.

**Output**: `anchor_carry_temporal.png`

---

## Usage

### Run All Plots
```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
python src/rag_analysis/cov_rag_plots/run_all_plots.py
```

### Run Individual Plots
```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
python src/rag_analysis/cov_rag_plots/1_late_hit_timing_distribution.py
python src/rag_analysis/cov_rag_plots/2_model_coverage_rates.py
# ... etc
```

## Requirements

- matplotlib
- numpy
- Python 3.7+

Install dependencies (using the project's virtual environment):
```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
pip install matplotlib numpy
```

## Input Data

All scripts read from: `/media/torontoai/Iterative-rag/src/rag_analysis/output/*coverage_gap_judgments.jsonl`

Expected JSON structure:
```json
{
  "question": "...",
  "is_correct": true/false,
  "parsed_judgment": {
    "retrieval_coverage_gap": {
      "has_gap": true/false,
      "missed_hops": [1, 2, ...]
    },
    "anchor_carry_drop": {
      "any_carry_drop": true/false,
      "per_step": [
        {"step": 1, "carry_drop": false},
        ...
      ]
    },
    "late_hit_per_hop": {
      "any_late_hit": true/false,
      "per_hop": [
        {"hop_index": 1, "first_hit_step": 1, "late_hit": false},
        ...
      ]
    }
  }
}
```

## Output

All plots are saved to the same directory as PNG files with 300 DPI resolution.

## Key Findings

Based on initial analysis across models:
- **Coverage gaps are rare**: Only 2-5% of runs have true retrieval gaps
- **Anchor carry-drop varies significantly**: 2.9% (GPT-5) to 20.1% (Claude-3.7)
- **Late hits are common**: 8-16% of runs across models
- **Hop 1 is most often missed**: In multi-hop questions, retrieval gaps primarily affect hop 1

## Author

Generated for Iterative RAG analysis project.
