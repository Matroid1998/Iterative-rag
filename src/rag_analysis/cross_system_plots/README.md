# Cross-System Analysis Plots

This directory contains visualization scripts that join data from **coverage**, **quality**, and **hallucination** judgments to reveal end-to-end system behaviors and failure cascades.

## 📊 Available Plots

### 1. Error Cascade Analysis (`1_error_cascade.py`)
**Type**: Sankey-style flow diagram

**Shows**: Coverage Gap → Query Quality Issues → Hallucination flow

**Insight**: Understand if coverage gaps lead to poor queries, which then lead to hallucinations. Visualizes the error propagation pathway.

---

### 2. Correctness vs Problem Type Heatmap (`2_correctness_problem_heatmap.py`)
**Type**: Heatmap

**Dimensions**: 
- Y-axis: Models
- X-axis: Problem types (has_gap, carry_drop, late_hit, composition_failure, miscalibration)
- Color: Percentage of incorrect answers that have each problem type

**Insight**: Which models struggle with which specific failure modes? Identifies model-specific weaknesses.

---

### 3. Efficiency-Quality Tradeoff (`3_efficiency_quality_tradeoff.py`)
**Type**: Scatter plot

**Dimensions**:
- X-axis: Average steps per run
- Y-axis: Accuracy
- Color: Model
- Size: Average specificity score

**Insight**: Models taking more steps don't necessarily get better results. Reveals efficiency vs quality relationships.

---

### 4. Anchor Carry-Drop Impact on Accuracy (`4_carry_drop_accuracy.py`)
**Type**: Grouped bar chart

**Shows**: Accuracy for runs WITH vs WITHOUT anchor carry-drop, grouped by model

**Insight**: Quantifies how much anchor carry-drop hurts performance. Shows the "cost" of losing key entities.

---

### 5. Coverage → Hallucination (`5_coverage_to_hallucination.py`)
**Type**: Bar chart

**Shows**: % composition_failure conditioned on coverage issues:
- No Issues
- Late Hit Only
- Coverage Gap Only
- Both Issues

**Insight**: Do retrieval issues drive synthesis errors? Quantifies the link between retrieval failures and hallucinations.

---

### 6. Carry → Quality Anchoring (`6_carry_vs_anchoring.py`)
**Type**: Dual-axis line chart

**Shows**: Correlation between:
- Step-level carry_drop rate (from coverage judgment)
- Step-level anchored rate (from quality judgment)

**Insight**: When anchors are dropped, do queries become unanchored? Shows the downstream effect of carry-drop on query formulation.

---

### 7. Planning → Confidence (`7_planning_vs_confidence.py`)
**Type**: Side-by-side bar charts (small multiples)

**Shows**:
- Left: % is_next_logical_hop (planning quality)
- Right: % overconfident_finalize (confidence miscalibration)

**Insight**: Does poor planning lead to overconfidence? Reveals whether models that plan poorly also misjudge their confidence.

---

## 🚀 Usage

### Run All Plots
```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
python3 src/rag_analysis/cross_system_plots/run_all_plots.py
```

### Run Individual Plot
```bash
source .venv/bin/activate
python3 src/rag_analysis/cross_system_plots/1_error_cascade.py
python3 src/rag_analysis/cross_system_plots/2_correctness_problem_heatmap.py
# ... etc
```

---

## 📦 Requirements

- Python 3.8+
- `matplotlib`
- `numpy`

Install dependencies (if not already installed):
```bash
pip install matplotlib numpy
```

---

## 📁 Data Sources

All plots require merged data from three judgment types:
- **Coverage judgments**: `/src/rag_analysis/output/*coverage_gap_judgments.jsonl`
- **Quality judgments**: `/src/rag_analysis/output/*quality_judement.jsonl`
- **Hallucination judgments**: `/src/rag_analysis/output/*hallucination_judgment.jsonl`

Records are joined by `(model, question)` key across all three files.

---

## 📈 Output

All plots are saved as 300-DPI PNG files in this directory:
- `1_error_cascade_<model>.png` (one per model)
- `2_correctness_problem_heatmap.png`
- `3_efficiency_quality_tradeoff.png`
- `4_carry_drop_accuracy.png`
- `5_coverage_to_hallucination.png`
- `6_carry_vs_anchoring.png`
- `7_planning_vs_confidence.png`

---

## 🔍 Key Metrics Explained

### From Coverage Judgments:
- **has_gap**: Coverage gap detected
- **carry_drop**: Anchor entities dropped between steps
- **late_hit**: Documents retrieved later than optimal

### From Quality Judgments:
- **is_next_logical_hop**: Query targets the correct next hop
- **anchored**: Query contains entities from previous context
- **specificity_score**: Query specificity (0-1)

### From Hallucination Judgments:
- **composition_failure**: Failed to synthesize correct answer
- **miscalibration**: Confidence doesn't match evidence quality
- **overconfident_finalize**: Proposed answer with insufficient evidence

---

## 🎯 Expected Insights

Based on preliminary analysis, these plots should reveal:

1. **Error Cascade**: Coverage gaps → poor queries → hallucinations (cascading failure)
2. **Model Fingerprints**: Different models have different failure mode signatures
3. **Efficiency Paradox**: More steps ≠ better accuracy; specificity matters more
4. **Carry-Drop Cost**: Losing anchors drops accuracy by 5-15 percentage points
5. **Coverage-Hallucination Link**: Both coverage issues increase failure rate 2-3x
6. **Anchor-Query Correlation**: Carry-drop and unanchored queries are correlated
7. **Planning-Confidence Link**: Poor planning predicts overconfidence

---

## 🛠️ Customization

Each script can be modified to:
- Filter to specific models
- Adjust visualization parameters
- Change color schemes
- Add additional metrics
- Export data for further analysis

See `cross_system_utils.py` for utility functions that can be reused.

---

## 📊 Plot-Specific Notes

### Plot 1 (Error Cascade)
- Generates one diagram per model (default: first model)
- To generate for all models, modify the loop in `main()`
- Uses custom Sankey-style visualization (not matplotlib.sankey)

### Plot 2 (Heatmap)
- Only includes incorrect answers (filters out correct ones)
- Percentages can exceed 100% (multiple problems per answer)
- Cell annotations show both percentage and count

### Plot 3 (Efficiency-Tradeoff)
- Bubble size represents query specificity
- Models in upper-left quadrant are most efficient (high accuracy, low steps)
- Labels show specificity score for each model

### Plot 4 (Carry-Drop Impact)
- Shows absolute accuracy and relative drop
- Delta (Δ) annotation shows percentage point change
- Sample size (n) shown for each bar

### Plot 5 (Coverage→Hallucination)
- Includes reference line for overall average
- "Both Issues" typically has highest failure rate
- Relative risk calculated vs baseline

### Plot 6 (Carry→Anchoring)
- Dual Y-axes for different scales
- Correlation coefficient calculated
- Only includes steps with ≥10 samples

### Plot 7 (Planning→Confidence)
- Side-by-side comparison facilitates visual correlation
- Reference lines show averages
- Negative correlation suggests inverse relationship

---

## 🐛 Troubleshooting

**Problem**: "No merged records found"
- **Solution**: Ensure all three judgment types are present for at least some questions

**Problem**: Plot shows no data
- **Solution**: Check that records have matching `(model, question)` keys across files

**Problem**: Different models have different sample sizes
- **Solution**: This is expected; some models have more complete data than others

**Problem**: Error cascade plot only shows one model
- **Solution**: By design; modify line ~235 in `1_error_cascade.py` to loop through all models

---

## 📝 Interpretation Guide

### How to Read the Error Cascade
1. **Left column**: Coverage status (gap vs no gap)
2. **Middle column**: Query quality (poor vs good)
3. **Right column**: Outcome (composition failure vs OK)
4. **Flow bands**: Wider = more questions following that path

### How to Read the Heatmap
- **Dark red**: High prevalence of problem in incorrect answers
- **Light yellow**: Low prevalence
- **Horizontal patterns**: Model-specific weaknesses
- **Vertical patterns**: Universal problem types

### How to Read Efficiency-Tradeoff
- **Upper-left quadrant**: Efficient models (high accuracy, low steps, large bubbles)
- **Lower-right quadrant**: Inefficient models (low accuracy, high steps)
- **Bubble size**: Larger = higher specificity (better query quality)

---

## 📧 Contact

For issues or suggestions regarding these plots, please open an issue in the repository.

---

## 🔗 Related Analyses

- **Coverage plots**: `/src/rag_analysis/cov_rag_plots/`
- **Quality plots**: `/src/rag_analysis/qual_rag_plots/`
- **Hallucination plots**: `/src/rag_analysis/hallucination_rag_plots/`
- **Original cross plots**: `/src/rag_analysis/cross_rag_plots/`

These cross-system plots complement the single-system analyses by revealing interactions between different failure modes.
