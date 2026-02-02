# Advanced Analysis Plots - Quick Start

## 🚀 Quick Run

```bash
cd /media/torontoai/Iterative-rag
source .venv/bin/activate
python3 src/rag_analysis/advanced_plots/run_all_plots.py
```

**Output**: 4 PNG files in same directory (300 DPI, ~2 MB total)

---

## 📊 What Each Plot Shows

| Plot | What It Reveals | Key Metric |
|------|----------------|------------|
| **1. Step Evolution** | Query quality trajectory | 41.7% degradation rate |
| **2. Model Radar** | 7-dimensional profile | 80.9% accuracy, 23.4% calibration |
| **3. Hop Effects** | Complexity scaling | +19.2pp miscalibration (1→2 hops) |
| **4. Steps Distribution** | Efficiency analysis | 2.82 avg steps, 1.03 delay |

---

## 🔍 Quick Findings

### Critical Issues
- ⚠️ **Calibration**: 86.7% miscalibrated on 2-hop questions
- ⚠️ **Query degradation**: 41.7% of clean queries become poor
- ⚠️ **Composition failures**: Double for 2-hop (15.4% → 28.4%)

### System Strengths
- ✅ **Coverage**: 100% (no retrieval gaps)
- ✅ **On-topic**: 95.1% stay on task
- ✅ **Accuracy**: 80.9% overall

---

## 📁 Files Generated

```
1_step_error_evolution.png      822 KB   Alluvial diagram
2_model_comparison_radar.png    600 KB   Radar chart
3_hop_count_effects.png         247 KB   Line plots
4_steps_per_run.png             228 KB   Histogram
```

---

## 💡 Top 3 Recommendations

1. **Fix 2-hop calibration** → Use complexity-aware thresholds
2. **Prevent query degradation** → Add quality gates at each step
3. **Reduce retrieval delay** → Implement prefetching

---

## 📖 Full Documentation

- **README.md**: Technical details, data sources, usage
- **KEY_FINDINGS.md**: Detailed analysis, implications, recommendations
- **SUMMARY.md**: Comprehensive overview, integration guide

---

## 🐛 Troubleshooting

**No plots generated?**
```bash
# Check environment
source .venv/bin/activate
python3 -c "import matplotlib; print('✓ OK')"

# Check data files
ls src/rag_analysis/output/*judgment*.jsonl
```

**Want individual plot?**
```bash
python3 src/rag_analysis/advanced_plots/3_hop_count_effects.py
```

---

## 🔄 Next Steps

1. Run plots: `python3 run_all_plots.py`
2. Open PNGs to visualize
3. Read KEY_FINDINGS.md for interpretation
4. Implement top 3 recommendations
5. Re-run analysis to measure improvement
