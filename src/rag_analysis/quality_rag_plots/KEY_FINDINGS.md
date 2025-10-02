# Quality (Query Audit) Analysis - Key Findings

> Analysis of 9,976 query steps across 3,558 runs for 6 models

---

## 🎯 Executive Summary

**Critical Discovery**: Fusion/skip strategies **hurt accuracy by 5-6%** across all hop counts, despite being intended as an optimization. Additionally, all models show significant **query degradation** over iteration steps, with specificity dropping 1.6-3.5% per step.

---

## 📊 Major Findings by Category

### 1️⃣ Query Degradation Over Steps

**Key Insight**: **All models degrade query quality as iterations progress**

| Model | Specificity Drop (per step) | On-Topic Drop (per step) | Overall Trend |
|-------|------------------------------|---------------------------|---------------|
| GPT-5 | **-1.6%** (best) | -3.0% | Slow degradation |
| Claude 3.7 Sonnet | -2.1% | -3.5% | Moderate degradation |
| Claude + Reasoning | -2.4% | -3.3% | Moderate degradation |
| Mistral Large | -3.1% | -3.6% | High degradation |
| GPT-4o | -3.1% | -3.5% | High degradation |
| DeepSeek R1 | **-3.5%** (worst) | -4.2% | High degradation |

**Example (Mistral)**:
- Step 1: 0.887 specificity, 0.943 on-topic
- Step 5: 0.773 specificity, 0.801 on-topic
- **Total loss**: -12.8% specificity, -15.0% on-topic

**Implication**: Query refinement mechanism needs improvement. Later queries become vaguer and less focused.

---

### 2️⃣ Fusion/Skip Effectiveness ⚠️

**Critical Finding**: **Fusion/skip strategies HURT accuracy consistently**

| Question Type | With Fusion/Skip | Without Fusion/Skip | Impact |
|---------------|------------------|---------------------|--------|
| **1-hop** | 83.3% (n=18) | **89.6%** (n=1,394) | **-6.3%** ⚠️ |
| **2-hop** | 79.7% (n=636) | **85.2%** (n=633) | **-5.4%** ⚠️ |

**Analysis**:
- Fusion/skip is used in **11-19%** of runs (DeepSeek R1 highest at 18.6%)
- It's **attempted more often** at later steps (step 5: 6.7-23.9%)
- Despite being a strategic shortcut, it **consistently underperforms** sequential reasoning

**Model-Specific Fusion Rates**:
1. DeepSeek R1: **18.6%** (highest, but accuracy suffers)
2. Claude + Reasoning: 16.9%
3. Claude 3.7: 15.3%
4. GPT-5: 13.3%
5. Mistral: 12.7%
6. GPT-4o: **12.5%** (lowest)

**Recommendation**: Consider **disabling or penalizing** fusion/skip, as it reduces accuracy without benefit.

---

### 3️⃣ Query Flag Co-occurrence

**Key Pattern**: **Query problems cluster together**

**Individual Occurrence Rates**:
- Compound queries: **11.0%** (most common)
- Over-broad queries: **6.0%**
- Off-topic queries: **2.7%**
- Vague queries: **1.1%** (least common)

**Strong Correlations** (conditional probabilities):
- **Vague → Over-broad**: 57.0% (if vague, likely also over-broad)
- **Over-broad → Compound**: 62.0% (if over-broad, likely also compound)
- **Over-broad + Compound co-occur**: 3.7% of all steps

**Interpretation**:
- Vague queries tend to be over-broad (not specific enough)
- Over-broad queries often contain compound concepts (too many ideas)
- Off-topic is relatively independent (2.7% baseline, weak correlations)

**Implication**: Fixing vague queries will also improve over-broad problems. They're not independent issues.

---

### 4️⃣ Distractor Latch vs Performance

**Critical Correlation**: **Distractor rate inversely predicts accuracy** (r = -0.578)

| Model | Distractor Rate | Accuracy | Interpretation |
|-------|-----------------|----------|----------------|
| GPT-5 | **7.4%** (best) | 80.9% | Good selectivity |
| Claude + Reasoning | 9.9% | **84.5%** (best accuracy) | Excellent balance |
| Claude 3.7 Sonnet | **10.6%** | **84.5%** | Excellent balance |
| DeepSeek R1 | 13.0% | 82.3% | Moderate traps |
| GPT-4o | 13.3% | 82.0% | Moderate traps |
| Mistral Large | **15.7%** (worst) | **75.3%** (worst) | High scaffold traps |

**Key Insight**:
- **Mistral** gets distracted 2.1× more often than GPT-5
- **Claude models** achieve best accuracy with low distraction rates
- Distractor latch = "scaffold trap" (locks onto chemically similar but wrong compound family)

**Recommendation**: Distractor latch is a strong predictor of failure. Priority intervention point.

---

### 5️⃣ Step Alignment (Hop Targeting)

**Finding**: **Models rarely target the correct hop** at each step

**Two Metrics**:
1. **Next Logical Hop**: Does the query target the immediate next hop in the chain?
2. **Step = Hop**: Simplified check (step number matches predicted hop)

| Model | Next Logical Hop | Step = Hop | Gap |
|-------|------------------|------------|-----|
| DeepSeek R1 | **76.1%** (best) | 53.0% | 23.1% |
| GPT-5 | 74.3% | **57.4%** (best strict) | 16.9% |
| Mistral Large | 64.2% | 36.1% | 28.1% |
| Claude + Reasoning | 60.7% | 35.5% | 25.2% |
| Claude 3.7 Sonnet | 58.7% | 41.1% | 17.6% |
| GPT-4o | **47.4%** (worst) | **27.9%** (worst strict) | 19.5% |

**Pattern by Step**:
- Step 1: 82-91% aligned (all models agree on starting hop)
- Step 2: 53-73% aligned (divergence begins)
- Step 3+: 22-61% aligned (significant misalignment)

**Interpretation**:
- **GPT-4o** loses alignment fastest (47.4% overall, 22.2% at step 5)
- **DeepSeek R1** maintains best hop-tracking (76.1% overall)
- Models struggle to maintain focus on the correct intermediate target

---

### 6️⃣ Query Flags Distribution (Model Profiles)

**Model-Specific Weaknesses**:

| Model | Vague | Over-Broad | Compound | Off-Topic | **Anchored** |
|-------|-------|------------|----------|-----------|--------------|
| **Claude + Reasoning** | 1.1% | 7.6% | 16.3% | 2.2% | **58.8%** ⭐ |
| Claude 3.7 Sonnet | 0.6% | 6.3% | 11.0% | 2.5% | **56.0%** ⭐ |
| GPT-4o | 1.7% | 6.2% | 7.8% | 2.5% | 52.4% |
| Mistral Large | 1.5% | 5.8% | 9.1% | 4.0% | 50.8% |
| DeepSeek R1 | 0.8% | 5.5% | 12.0% | 2.5% | 42.9% |
| **GPT-5** | **0.4%** ⭐ | **2.9%** ⭐ | 12.4% | 2.6% | **39.3%** |

**Key Profiles**:

1. **Claude Models (with/without reasoning)**:
   - **Highest anchoring** (56-59%) - good grounding in context
   - Moderate compound queries (11-16%)
   - **Best accuracy** (84.5%)

2. **GPT-5**:
   - **Lowest problem rates** across vague/over-broad/off-topic
   - **Lowest anchoring** (39.3%) - more exploratory
   - Good accuracy (80.9%) despite less anchoring

3. **Mistral Large**:
   - **Highest off-topic rate** (4.0%)
   - Moderate issues across all flags
   - Lowest accuracy (75.3%)

4. **DeepSeek R1**:
   - Low problem rates (good)
   - **Highest compound rate** (12.0%) among reasoning models
   - Good accuracy (82.3%)

5. **GPT-4o**:
   - **Highest vague rate** (1.7%)
   - Lowest compound rate (7.8%)
   - Moderate accuracy (82.0%)

**Interpretation**:
- **Anchoring correlates with accuracy**: Claude (58.8% anchored, 84.5% accurate) vs GPT-5 (39.3% anchored, 80.9% accurate)
- **Compound queries**: Not necessarily bad (GPT-5 has 12.4% compound but good accuracy)
- **Off-topic is rare** (2.2-4.0%) but Mistral struggles most

---

### 7️⃣ Score Distributions and Trends

**Overall Score Profiles**:

| Model | Specificity Mean ± Std | On-Topic Mean ± Std |
|-------|------------------------|---------------------|
| **GPT-5** | **0.879 ± 0.117** ⭐ | **0.918 ± 0.155** ⭐ |
| DeepSeek R1 | 0.853 ± 0.133 | 0.900 ± 0.168 |
| Claude 3.7 Sonnet | 0.851 ± 0.124 | 0.899 ± 0.167 |
| Claude + Reasoning | 0.838 ± 0.134 | 0.895 ± 0.160 |
| Mistral Large | 0.837 ± 0.150 | 0.882 ± 0.189 |
| **GPT-4o** | **0.823 ± 0.152** | **0.878 ± 0.176** |

**Key Insights**:
- **GPT-5**: Highest mean scores AND lowest variability (most consistent)
- **GPT-4o**: Lowest scores AND highest variability (least consistent)
- **Reasoning models** (Claude + Reasoning, DeepSeek R1): Middle of the pack

**Trend Analysis** (already covered in Finding #1):
- All models degrade over steps
- GPT-5 degrades slowest (-1.6% per step)
- DeepSeek R1 degrades fastest (-3.5% per step)

---

### 8️⃣ Fusion/Skip by Step (Temporal Patterns)

**When do models attempt fusion/skip?**

| Model | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Overall |
|-------|--------|--------|--------|--------|--------|---------|
| Claude + Reasoning | **19.8%** | 12.5% | 15.4% | 18.8% | 19.4% | 16.9% |
| DeepSeek R1 | **18.9%** | 16.2% | 20.5% | 18.8% | **23.9%** | 18.6% |
| GPT-5 | 16.9% | 8.3% | 8.7% | 8.2% | **6.7%** | 13.3% |
| Claude 3.7 Sonnet | 15.0% | 11.9% | 16.0% | 18.0% | 23.6% | 15.3% |
| GPT-4o | 12.6% | 7.5% | 11.1% | 12.5% | 19.3% | 12.5% |
| Mistral Large | 11.0% | 8.7% | 13.8% | 17.2% | 19.3% | 12.7% |

**Patterns**:
1. **Early fusion** (Step 1): Claude + Reasoning and DeepSeek R1 try to skip early (18-20%)
2. **Late fusion** (Step 5): Most models increase fusion attempts at final step (19-24%)
3. **Consistent low** (GPT-5): Reduces fusion over time (16.9% → 6.7%)

**Interpretation**:
- **Step 1 fusion** (18-20%): Attempting to answer directly without RAG
- **Step 5 spike** (19-24%): Desperation move when iterations aren't helping
- **GPT-5's declining fusion**: Learns that sequential is better

**But remember**: Fusion/skip hurts accuracy by 5-6% (Finding #2)!

---

### 9️⃣ Stability Analysis (Contradictions + Distractors)

**Contradiction Rates** (partial contradiction with previous step):

| Model | Contradiction Rate | Interpretation |
|-------|-------------------|----------------|
| **GPT-5** | **0.8%** ⭐ | Most stable |
| Claude 3.7 Sonnet | 1.3% | Very stable |
| DeepSeek R1 | 1.7% | Stable |
| Claude + Reasoning | 2.0% | Stable |
| Mistral Large | 7.4% | Moderate instability |
| **GPT-4o** | **9.6%** ⚠️ | Highest instability |

**Distractor Latch Rates** (from Finding #4):

| Model | Distractor Rate | Interpretation |
|-------|-----------------|----------------|
| GPT-5 | **7.4%** ⭐ | Best selectivity |
| Claude + Reasoning | 9.9% | Excellent |
| Claude 3.7 Sonnet | 10.6% | Excellent |
| DeepSeek R1 | 13.0% | Moderate |
| GPT-4o | 13.3% | Moderate |
| **Mistral Large** | **15.7%** ⚠️ | High scaffold traps |

**Combined Stability Score** (lower is better):

| Model | Contradictions + Distractors | Overall Stability |
|-------|------------------------------|-------------------|
| **GPT-5** | **8.2%** ⭐ | Most stable |
| Claude 3.7 Sonnet | 11.9% | Very stable |
| Claude + Reasoning | 11.9% | Very stable |
| DeepSeek R1 | 14.7% | Moderate |
| **GPT-4o** | **22.9%** | Unstable |
| **Mistral Large** | **23.1%** ⚠️ | Most unstable |

**Key Insights**:
- **GPT-5**: Best at maintaining consistent reasoning and avoiding distractors
- **Claude models**: Excellent stability (low contradictions, low distractors)
- **GPT-4o**: High contradictions (9.6%) but moderate distractors (13.3%)
- **Mistral**: High distractors (15.7%) but moderate contradictions (7.4%)

---

## 🎯 Model Rankings Summary

### 1. Best Overall Quality: **GPT-5**
- ✅ Highest specificity (0.879) and on-topic (0.918) scores
- ✅ Lowest query problem rates (vague, over-broad, off-topic)
- ✅ Slowest query degradation (-1.6% per step)
- ✅ Lowest contradiction rate (0.8%)
- ✅ Lowest distractor rate (7.4%)
- ✅ Best overall stability (8.2%)
- ⚠️ Lower anchoring (39.3%) - more exploratory
- ⚠️ Moderate accuracy (80.9%)

### 2. Best Accuracy: **Claude 3.7 Sonnet (with/without reasoning)**
- ✅ Highest accuracy (84.5%)
- ✅ High anchoring (56-59%)
- ✅ Low contradictions (1.3-2.0%)
- ✅ Low distractors (9.9-10.6%)
- ✅ Excellent stability (11.9%)
- ⚠️ Moderate query degradation (-2.1 to -2.4% per step)
- ⚠️ High compound queries (11-16%)

### 3. Best Hop Tracking: **DeepSeek R1**
- ✅ Best next logical hop alignment (76.1%)
- ✅ Good score profiles (0.853 specificity)
- ✅ Low problem rates
- ✅ Good accuracy (82.3%)
- ⚠️ Highest query degradation (-3.5% per step)
- ⚠️ Highest fusion rate (18.6%)
- ⚠️ Moderate distractors (13.0%)

### 4. Most Consistent: **GPT-5**
- ✅ Lowest score variability (0.117 std for specificity)
- ✅ Most stable reasoning (0.8% contradictions)
- ✅ Best distractor avoidance (7.4%)
- ✅ Reduces fusion over time (16.9% → 6.7%)

### 5. Worst Performance: **Mistral Large**
- ❌ Lowest accuracy (75.3%)
- ❌ Highest distractor rate (15.7%)
- ❌ High contradictions (7.4%)
- ❌ Worst stability (23.1% combined)
- ❌ Highest off-topic rate (4.0%)
- ❌ High query degradation (-3.1% per step)

### 6. Most Unstable: **GPT-4o**
- ❌ Highest contradiction rate (9.6%)
- ❌ Worst hop alignment (47.4%)
- ❌ Lowest score means (0.823 specificity, 0.878 on-topic)
- ❌ Highest score variability (0.152 std)
- ⚠️ Moderate accuracy (82.0%)
- ⚠️ Moderate distractors (13.3%)

---

## 💡 Actionable Recommendations

### 🚨 Priority 1: Disable/Fix Fusion/Skip
- **Finding**: Fusion/skip reduces accuracy by 5-6%
- **Action**: Remove or heavily penalize fusion/skip strategies
- **Alternative**: Only allow fusion on step 1 for truly simple 1-hop questions

### 🚨 Priority 2: Address Query Degradation
- **Finding**: All models lose 1.6-3.5% specificity per step
- **Action**: Implement query refinement feedback loop
- **Alternative**: Use GPT-5's query formulation to guide other models

### 🚨 Priority 3: Reduce Distractor Latch
- **Finding**: Strong negative correlation with accuracy (r = -0.578)
- **Action**: Improve compound family disambiguation in retrieval
- **Focus**: Mistral Large (15.7% distractor rate)

### 🔧 Priority 4: Improve Hop Alignment
- **Finding**: GPT-4o loses hop tracking (47.4% next logical hop)
- **Action**: Explicit hop-tracking mechanism in prompts
- **Alternative**: Multi-step planning before first query

### 📊 Priority 5: Model-Specific Tuning
- **GPT-5**: Reduce fusion attempts further (already declining trend)
- **Claude models**: Maintain current approach (best accuracy)
- **Mistral**: Address off-topic drift (4.0% rate)
- **GPT-4o**: Stabilize reasoning (9.6% contradiction rate)
- **DeepSeek R1**: Slow down query degradation (-3.5% per step)

---

## 🔗 Cross-Reference with Coverage Gap Analysis

**Complementary Insights**:

| Coverage Gap Findings | Quality Findings | Combined Insight |
|-----------------------|------------------|------------------|
| Coverage gaps reduce accuracy to 34.8% | Distractor latch reduces accuracy (r=-0.578) | Both retrieval AND query issues hurt accuracy |
| Late hits happen 13.6% at hop 1 | Fusion/skip at step 1: 11-20% | Models try shortcuts when retrieval lags |
| Mistral loses 25.7% anchors at step 2 | Mistral has highest distractor rate (15.7%) | Mistral struggles with both anchoring and distractors |
| 2-hop questions: 5.0% gap rate | Fusion/skip hurts 2-hop by 5.4% | Complex questions suffer from both issues |
| Claude has lowest gap rate (2.3%) | Claude has best accuracy (84.5%) | Good retrieval → good outcomes |

**Unified Failure Model**:
1. **Coverage Gap** → Retrieval doesn't find the right document
2. **Distractor Latch** → Retrieval finds wrong but similar document
3. **Query Degradation** → Later queries become vaguer, worsening retrieval
4. **Fusion/Skip** → Model tries shortcut, loses accuracy
5. **Hop Misalignment** → Model targets wrong intermediate, chains break

**Fix Priority**:
1. Disable fusion/skip (immediate 5-6% accuracy gain)
2. Reduce distractor latch (better compound disambiguation)
3. Fix query degradation (maintain specificity over iterations)
4. Improve coverage (reduce gaps and late hits)

---

## 📈 Statistical Summary

- **Total Query Steps Analyzed**: 9,976
- **Total Runs**: 3,558
- **Models**: 6
- **Hop Counts**: 1-5 steps
- **Question Types**: 1-hop (23.7%), 2-hop (76.3%)

**Key Metrics**:
- **Specificity Range**: 0.823-0.879 (mean across models)
- **On-Topic Range**: 0.878-0.918 (mean across models)
- **Query Degradation**: -1.6% to -3.5% per step
- **Fusion/Skip Rate**: 11-19% (varies by model)
- **Fusion/Skip Impact**: -5.4% to -6.3% accuracy
- **Distractor Correlation**: r = -0.578 with accuracy
- **Contradiction Range**: 0.8-9.6% (GPT-5 best, GPT-4o worst)
- **Distractor Range**: 7.4-15.7% (GPT-5 best, Mistral worst)

---

**Generated**: From 9 plot analyses across 6 models
**Plots**: All 9 visualizations available in `/src/rag_analysis/quality_rag_plots/`
**Documentation**: See `README.md` and `IMPLEMENTATION_SUMMARY.md` for details
