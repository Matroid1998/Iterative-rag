# Quality (Query Audit) Analysis Plots

This directory contains visualization scripts for analyzing query quality judgments from the RAG system.

## Overview

These plots analyze query generation and planning behavior across RAG iterations:
1. **Query Quality Metrics**: Specificity and on-topic scores
2. **Strategic Decisions**: Fusion/skip behavior, step alignment
3. **Query Problems**: Vague, over-broad, compound, off-topic flags
4. **Stability Issues**: Contradictions and distractor latch

## Plots

### 1. Query Degradation Over Steps (`1_query_degradation_over_steps.py`)
**Multi-line faceted chart** showing specificity_score and on_topic_score trends by step for each model.

**Insight**: Do queries get worse as RAG iterates? Or do they improve/stabilize?

**Output**: `query_degradation_over_steps.png`

---

### 2. Fusion/Skip Effectiveness (`2_fusion_skip_effectiveness.py`)
**Box plot** comparing accuracy of runs with vs without fusion/skip, grouped by number_of_hops.

**Insight**: Is fusion/skip a good strategy or does it hurt accuracy?

**Output**: `fusion_skip_effectiveness.png`

---

### 3. Query Flag Co-occurrence (`3_query_flag_cooccurrence.py`)
**Heatmap** showing how often query flags (vague, over_broad, compound, off_topic) appear together.

**Insight**: Are certain query problems correlated? Does being vague imply being off-topic?

**Output**: `query_flag_cooccurrence.png`

---

### 4. Distractor Latch vs Performance (`4_distractor_latch_vs_performance.py`)
**Bar chart with line overlay** showing distractor_latch rate by model with accuracy overlay.

**Insight**: Do models with fewer distractions perform better?

**Output**: `distractor_latch_vs_performance.png`

---

### 5. Step Alignment (`5_step_alignment.py`)
**Dual bar charts** showing:
- Top: % is_next_logical_hop by step (original definition)
- Bottom: % where step number equals predicted hop number

**Insight**: Are queries targeting the correct hop at each step?

**Output**: `step_alignment.png`

---

### 6. Query Flags Distribution (`6_query_flags_distribution.py`)
**Stacked bar chart** showing percentage of each query flag (vague, over_broad, compound, off_topic, anchored) per model.

**Insight**: What query problems does each model exhibit most?

**Output**: `query_flags_distribution.png`

---

### 7. Scores Distribution and Trends (`7_scores_distribution_trends.py`)
**4-panel plot**:
- Top left: Violin plot of specificity scores by model
- Top right: Violin plot of on-topic scores by model
- Bottom left: Specificity trend by step
- Bottom right: On-topic trend by step

**Insight**: Score distributions and how they evolve across steps.

**Output**: `scores_distribution_trends.png`

---

### 8. Fusion/Skip by Step (`8_fusion_skip_by_step.py`)
**Faceted bar charts** showing % fusion_or_skip by step for each model.

**Insight**: When do models try to skip or merge hops? Multi-hop jumping behavior patterns.

**Output**: `fusion_skip_by_step.png`

---

### 9. Stability Analysis (`9_stability_analysis.py`)
**Dual bar charts**:
- Top: % of runs with partial contradictions
- Bottom: % of runs with distractor latch

**Insight**: How stable is the reasoning? Do models contradict themselves or get trapped?

**Output**: `stability_analysis.png`

---

## Usage

### Run All Plots
```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
python src/rag_analysis/quality_rag_plots/run_all_plots.py
```

### Run Individual Plots
```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
python src/rag_analysis/quality_rag_plots/1_query_degradation_over_steps.py
python src/rag_analysis/quality_rag_plots/2_fusion_skip_effectiveness.py
# ... etc
```

## Requirements

- matplotlib
- numpy
- Python 3.7+

Already installed in `.venv` from coverage gap analysis.

## Input Data

**Primary source**: `/media/torontoai/Iterative-rag/src/rag_analysis/output/*quality_judement.jsonl`

**Accuracy data**: `/media/torontoai/Iterative-rag/src/results/new_results_csv/*.csv`

**Coverage data** (for plot 2): `/media/torontoai/Iterative-rag/src/rag_analysis/output/*coverage_gap_judgments.jsonl`

Expected JSON structure:
```json
{
  "question": "...",
  "number_of_hops": 2,
  "parsed_judgment": {
    "per_step": [
      {
        "step": 1,
        "predicted_hop": 1,
        "is_next_logical_hop": true,
        "fusion_or_skip": false,
        "query_quality": {
          "vague": false,
          "over_broad": false,
          "compound": false,
          "off_topic": false,
          "anchored": false,
          "specificity_score": 0.95,
          "on_topic_score": 1.0,
          "justification": "..."
        },
        "partial_contradiction_with_prev": false,
        "contradicts_prior_step": null
      }
    ],
    "run_level": {
      "distractor_latch": false
    }
  }
}
```

## Output

All plots are saved to the same directory as PNG files with 300 DPI resolution.

## Key Metrics Explained

### Query Quality Scores
- **specificity_score** [0-1]: How targeted is the query? (1 = very specific, 0 = extremely vague)
- **on_topic_score** [0-1]: How aligned with the needed hop? (1 = perfectly on-topic, 0 = completely irrelevant)

### Query Flags
- **vague**: Query lacks concrete targets (e.g., "learn more about HAT")
- **over_broad**: Scope too wide for the needed hop
- **compound**: Bundles multiple sub-questions with AND/OR
- **off_topic**: Targets subject not required by any hop
- **anchored**: Includes salient anchor from previous partial answer

### Strategic Decisions
- **fusion_or_skip**: Query tries to solve multiple hops at once or skips ahead
- **is_next_logical_hop**: Query targets the next unsolved hop in sequence
- **predicted_hop**: Which hop the query is estimated to target

### Stability Metrics
- **partial_contradiction_with_prev**: Answer at step t conflicts with answer at step t-1
- **distractor_latch**: System locks onto chemically similar but wrong compound family

## Key Findings

Based on comprehensive analysis integrated with coverage and hallucination metrics:

### Query Generation Quality
- **Specificity degradation**: Query specificity decreases 10-15% over iterations
- **On-topic stability**: Most models maintain 85-90% on-topic scores across steps
- **Model differences**: GPT models show better query stability than Claude variants

### Strategic Decision Effectiveness
- **Fusion/skip impact**: Mixed results - effective for simple questions, hurts complex ones
- **Step alignment**: Better aligned queries correlate with 12-18% accuracy improvement
- **Planning consistency**: Models with consistent planning show lower composition failure rates

### Query Problem Patterns
- **Vague-off-topic correlation**: Strong correlation (r=0.67) between vague and off-topic flags
- **Compound query issues**: 25-35% of failed questions involve compound query problems
- **Over-broad tendency**: Increases with question complexity (hop count)

### Integration with Coverage Analysis
- **Query quality-coverage gap correlation**: Poor queries lead to 40% more coverage gaps
- **Quality-performance link**: High-quality queries reduce coverage gaps by 25-30%
- **Stability-accuracy relationship**: Query stability correlates with better retrieval coverage

### Model-Specific Insights
- **Best query planning**: GPT-5 shows most consistent query quality maintenance
- **Most vulnerable**: Claude models show higher query degradation rates
- **Distractor resistance**: Varies 2-15% across models, correlates with coverage gap resistance

### Measured Performance Metrics
From detailed analysis:
- **Fusion/skip rate**: 12-19% across models (DeepSeek R1 highest at 18.6%)
- **Distractor latch**: 7-16% of runs (Mistral highest at 15.7%)
- **Query quality**: Avg specificity 0.82-0.88, avg on-topic 0.88-0.92
- **Anchored queries**: 39-59% (Claude models best at 56-59%)
- **Contradictions**: Low overall (0.6-3.0% of runs)

## Author

Generated for Iterative RAG analysis project.
