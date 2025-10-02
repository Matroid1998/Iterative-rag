# Hallucination Analysis Plots

This directory contains visualization scripts for analyzing hallucination-related metrics in the iterative RAG evaluation pipeline. These plots help identify composition failures, confidence miscalibration, and unsupported claims patterns across different models.

## 📊 Available Plots

### 1. Miscalibration Direction by Hop Count (`1_miscalibration_by_hop.py`)
**Stacked bar chart**: Shows miscalibration direction (overconfident/underconfident/ok) by question complexity (number of hops).

**Insight**: Are models overconfident on simple questions and underconfident on complex ones?

---

### 2. Sufficiency vs Coverage Scatter (`2_sufficiency_vs_coverage.py`)
**Scatter plot**: X=sufficiency_score_est, Y=hop_coverage_est, color=miscalibration direction, size=unsupported_claims.

**Insight**: Can we predict miscalibration from sufficiency and coverage scores? Identifies dangerous quadrants.

---

### 3. Unsupported Claims Distribution (`3_unsupported_claims_distribution.py`)
**Histogram**: Distribution of unsupported claims per run, faceted by model.

**Insight**: Which models make more unsupported claims? Shows faithfulness patterns.

---

### 4. Composition Failure Root Causes (`4_composition_failure_root_causes.py`)
**Grouped bar chart**: Percentage of composition failures that also have coverage_gap, carry_drop, late_hit, or poor_query_quality.

**Insight**: What leads to composition failure? Identifies primary failure modes.

---

### 5. Composition Failure Rate (`5_composition_failure_rate.py`)
**Bar chart**: Percentage of composition failures per model.

**Insight**: Which models have higher composition failure rates?

---

### 6. Evidence Sufficiency Distribution (`6_sufficiency_distribution.py`)
**Histogram + Box plot**: Distribution of sufficiency_score_est with 0.6 threshold line.

**Insight**: How well does the evidence support the answers across the dataset?

---

### 7. Miscalibration Mix per Model (`7_miscalibration_mix.py`)
**Stacked bar chart**: Confidence directions (overconfident/underconfident/ok) per model with overall miscalibration rate.

**Insight**: Which models are overconfident vs underconfident? Shows confidence calibration patterns.

---

### 8. Coverage vs Confidence Scatter (`8_coverage_vs_confidence.py`)
**Scatter plot**: hop_coverage_est vs sufficiency_score_est colored by miscalibration direction with regime annotations.

**Insight**: What combinations of coverage and confidence lead to miscalibration? Visualizes risk zones.

---

## 🚀 Usage

### Run All Plots
```bash
cd /media/torontoai/Iterative-rag/src/rag_analysis/hallucination_rag_plots
python3 run_all_plots.py
```

### Run Individual Plot
```bash
python3 1_miscalibration_by_hop.py
python3 2_sufficiency_vs_coverage.py
# ... etc
```

---

## 📦 Requirements

- Python 3.8+
- `matplotlib`
- `numpy`
- `pandas` (optional, for some analyses)

Install dependencies:
```bash
pip install matplotlib numpy pandas
```

---

## 📁 Data Sources

All plots read from:
- **Hallucination judgments**: `/src/rag_analysis/output/*hallucination_judgment.jsonl`
- **Coverage judgments**: `/src/rag_analysis/output/*coverage_gap_judgments.jsonl` (for plot 4)
- **Quality judgments**: `/src/rag_analysis/output/*quality_judement.jsonl` (for plot 4)

---

## 📈 Output

All plots are saved as 300-DPI PNG files in this directory with descriptive names:
- `1_miscalibration_by_hop.png`
- `2_sufficiency_vs_coverage.png`
- `3_unsupported_claims_distribution.png`
- etc.

---

## 🔍 Key Metrics Explained

### From Hallucination Judgments:

1. **Composition Failure** (`composition_and_faithfulness.composition_failure`):
   - Binary: Did the system fail to properly synthesize the answer?
   
2. **Unsupported Claims** (`composition_and_faithfulness.unsupported_claims`):
   - Per-step array indicating if claims are supported by evidence

3. **Sufficiency Score** (`composition_and_faithfulness.sufficiency_score_est`):
   - 0.0 to 1.0: How well does evidence support the answer?
   - Threshold: 0.6

4. **Hop Coverage Estimate** (`confidence_miscalibration.hop_coverage_est`):
   - 0.0 to 1.0: Estimated coverage of required hops
   
5. **Miscalibration Direction** (`confidence_miscalibration.direction`):
   - `ok`: Properly calibrated
   - `overconfident_finalize`: Proposed answer with insufficient evidence
   - `underconfident_continue`: Continued searching despite having sufficient evidence

6. **Is Miscalibrated** (`confidence_miscalibration.is_miscalibrated`):
   - Binary: Overall miscalibration flag

---

## 🎯 Key Findings (Expected)

Based on preliminary data analysis:

- **~55% miscalibration rate** across models
- **~18.8% composition failure rate**
- **~25.8% unsupported claims** at step level
- Strong correlation between low sufficiency and overconfidence
- Coverage gaps frequently lead to composition failures

Actual findings may vary as more data is collected.

---

## 🛠️ Customization

Each script can be modified to:
- Change color schemes
- Adjust bin sizes for histograms
- Modify threshold values
- Add additional annotations
- Export data to CSV for further analysis

See individual script files for customization options.

---

## 📝 Notes

- Scripts skip gracefully if required input files are missing
- All plots include statistical summaries printed to console
- Plots are designed for publication quality (300 DPI)
- Color schemes are colorblind-friendly where possible

---

## 🐛 Troubleshooting

**Problem**: "No hallucination judgment files found"
- **Solution**: Ensure you've run the hallucination judgment scripts first

**Problem**: Plot 4 (Root Causes) shows no data
- **Solution**: Requires coverage AND quality judgment files in addition to hallucination judgments

**Problem**: Model names look weird
- **Solution**: Edit `hall_plot_utils.py` `normalize_model_name()` function to add your model patterns

---

## 📧 Contact

For issues or suggestions regarding these plots, please open an issue in the repository.
