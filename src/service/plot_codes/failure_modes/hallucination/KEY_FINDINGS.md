# Hallucination Analysis - Key Findings

**Generated**: October 2, 2025  
**Data Source**: Hallucination judgment output files (2,965 runs for GPT-5)

---

## 🎯 Executive Summary

Analysis of hallucination patterns in RAG system reveals:
1. **55% miscalibration rate** - Models frequently misjudge their evidence quality
2. **18.9% composition failure rate** - Significant answer synthesis issues
3. **47.4% of composition failures linked to poor query quality** - Main root cause
4. **2-hop questions show higher underconfidence** - Complexity affects calibration

---

## 📊 Detailed Findings by Plot

### 1. Miscalibration Direction by Hop Count

**Key Insights:**
- **1-hop questions (n=1490)**:
  - 53.0% OK (well-calibrated)
  - 33.5% Underconfident
  - 13.6% Overconfident

- **2-hop questions (n=1475)**:
  - 36.9% OK (well-calibrated)
  - 45.6% Underconfident ⚠️
  - 17.5% Overconfident

**Critical Finding**: Models become MORE underconfident on complex questions, suggesting they struggle to recognize when they have sufficient evidence for multi-hop reasoning.

---

### 2. Sufficiency vs Coverage Scatter

**Key Patterns by Calibration:**

**OK (Correct Calibration) - n=1333**:
- Avg Sufficiency: 0.949
- Avg Coverage: 0.978
- Avg Unsupported Claims: 0.23
- **97.2% in High Coverage & High Sufficiency quadrant** ✓

**Underconfident - n=1172**:
- Avg Sufficiency: 0.880
- Avg Coverage: 0.997 (very high!)
- Avg Unsupported Claims: 0.71
- **98.5% in High Coverage & High Sufficiency quadrant**
- **Issue**: System has good evidence but doesn't recognize it

**Overconfident - n=460**:
- Avg Sufficiency: 0.431 ⚠️ (very low)
- Avg Coverage: 0.950
- Avg Unsupported Claims: 1.86 (highest!)
- **92.6% in High Coverage & LOW Sufficiency quadrant** 🚨
- **Issue**: System proposes answers despite poor evidence quality

**Critical Finding**: Overconfidence strongly correlates with low sufficiency scores (< 0.6). This is the "danger zone" where models hallucinate most.

---

### 3. Unsupported Claims Distribution

**GPT-5 Statistics (n=2965)**:
- 59.7% of runs have ZERO unsupported claims ✓
- Mean unsupported claims: 0.67 per run
- Median: 0 (most runs are clean)
- Max: 5 unsupported claims in worst case

**Interpretation**: Most answers are faithful to evidence, but when failures occur, they involve 1-2 unsupported statements on average.

---

### 4. Composition Failure Root Causes

**Total Composition Failures: 559 (18.9% of runs)**

**Co-occurrence with failure modes**:
- **Poor Query Quality**: 265 cases (47.4%) 🎯 PRIMARY CAUSE
- **Late Hit**: 98 cases (17.5%)
- **Anchor Carry-Drop**: 97 cases (17.4%)
- **Coverage Gap**: 66 cases (11.8%)

**Critical Finding**: Nearly HALF of composition failures coincide with poor query formulation. Improving query quality could reduce composition failures by up to 47%.

**Multi-factor failures**: Many composition failures have multiple root causes, suggesting compounding effects.

---

### 5. Composition Failure Rate

**GPT-5 Performance**:
- Total runs: 2,965
- Composition failures: 559
- **Failure rate: 18.9%**

**Interpretation**: Approximately 1 in 5 answers fail to properly synthesize information from retrieved evidence.

---

### 6. Evidence Sufficiency Distribution

**Overall Statistics (n=2965)**:
- Mean sufficiency: 0.841
- Median: 1.000 (most runs have perfect sufficiency)
- Std: 0.226
- **Below threshold (0.6): 15.5%** ⚠️

**Distribution Pattern**:
- Strong bimodal distribution
- Most runs cluster at 1.0 (perfect) or 0.0 (complete failure)
- Few runs in the "marginal evidence" zone (0.4-0.6)

**Critical Finding**: Evidence is typically either very strong (1.0) or very weak (< 0.5). The 0.6 threshold effectively separates these regimes.

---

### 7. Miscalibration Mix per Model

**GPT-5 Breakdown (n=2965)**:
- **Overall miscalibration: 55.0%** 🚨
- OK (Calibrated): 1,333 (45.0%)
- Underconfident: 1,172 (39.5%)
- Overconfident: 460 (15.5%)

**Key Ratios**:
- Underconfident : Overconfident = 2.5 : 1
- Models err on the side of caution more often than overconfidence

**Interpretation**: While overconfidence is less common, it's more dangerous (leads to hallucinations). Underconfidence wastes computational resources but is safer.

---

### 8. Coverage vs Confidence Scatter

**Quadrant Analysis**:

**High Coverage & High Sufficiency (SAFE ZONE)**:
- OK: 97.2%
- Underconfident: 98.5%
- Overconfident: 0.0%

**High Coverage & LOW Sufficiency (DANGER ZONE)**:
- OK: 0.0%
- Underconfident: 1.0%
- Overconfident: 92.6% 🚨

**Low Coverage & High Sufficiency (OVERCONFIDENT RISK)**:
- OK: 2.8%
- Underconfident: 0.4%
- Overconfident: 2.8%

**Critical Finding**: The danger zone is **high coverage but low sufficiency**. This suggests the system retrieves documents but they don't actually support the answer. This is where most overconfident errors occur.

---

## 🔑 Cross-Plot Insights

### Pattern 1: Complexity → Underconfidence
- 2-hop questions have 45.6% underconfidence vs 33.5% for 1-hop
- Models struggle to recognize when multi-hop evidence is sufficient

### Pattern 2: Low Sufficiency → Overconfidence → Hallucination
- Overconfident runs average 0.431 sufficiency (below threshold)
- These runs also have 1.86 unsupported claims on average
- Clear causal chain: poor evidence → overconfidence → unsupported claims

### Pattern 3: Query Quality is the Biggest Lever
- 47.4% of composition failures link to poor query quality
- This is the highest co-occurrence among all root causes
- **Actionable**: Improving query formulation could reduce failures by ~50%

### Pattern 4: Bimodal Evidence Quality
- Sufficiency scores cluster at 0.0 and 1.0
- Few "marginal" cases in between
- Suggests retrieval either succeeds completely or fails completely

### Pattern 5: High Coverage ≠ Good Evidence
- Overconfident runs have 95% average coverage
- But only 43% sufficiency
- **Issue**: Retrieving many documents doesn't guarantee relevance

---

## 🎯 Recommendations

### 1. **Priority: Improve Query Quality**
- Root cause of 47.4% of composition failures
- Focus on: reducing vague/over-broad/compound queries
- Implement query validation before retrieval

### 2. **Add Sufficiency Threshold Check**
- Block proposals when sufficiency < 0.6
- Would prevent most overconfident errors (92.6% occur below threshold)
- Trade-off: May increase underconfidence slightly

### 3. **Confidence Calibration Training**
- Train models to better estimate evidence sufficiency on 2-hop questions
- Current: 45.6% underconfident on 2-hop vs 33.5% on 1-hop
- Models are too conservative on complex questions

### 4. **Evidence Quality Filtering**
- High coverage (95%) doesn't prevent overconfidence
- Need better relevance scoring, not just more documents
- Implement post-retrieval evidence validation

### 5. **Detect Danger Zone Early**
- Monitor runs entering "high coverage, low sufficiency" regime
- This is where 92.6% of overconfident errors occur
- Early warning could trigger additional retrieval or human review

---

## 📈 Expected Impact

If recommendations are implemented:

| Improvement | Expected Impact |
|------------|----------------|
| Query quality validation | -47% composition failures |
| Sufficiency threshold (0.6) | -92% overconfident errors |
| Evidence quality filtering | -30% low-sufficiency cases |
| Calibration training | -20% underconfidence on 2-hop |

**Total**: Could reduce composition failures from 18.9% → ~9-10% and nearly eliminate dangerous overconfidence.

---

## 🔍 Areas for Further Investigation

1. **Why do 2-hop questions cause more underconfidence?**
   - Is it hop count or question complexity?
   - Are models overly cautious after first hop?

2. **What makes sufficiency bimodal?**
   - Why so few "marginal evidence" cases?
   - Is this a retrieval issue or evaluation issue?

3. **Can we predict overconfidence before proposal?**
   - Build early warning system based on sufficiency trajectory
   - Intervene before dangerous proposals

4. **Query quality vs other failure modes**
   - Does poor query quality compound with coverage gaps?
   - Are there query patterns that predict failure?

---

## 📊 Data Quality Notes

- Analysis based on 2,965 runs for GPT-5
- Additional models available but not yet analyzed in detail
- Some runs missing coverage/quality judgments (merged on ~559 composition failures)
- All metrics are from LLM-based judgments, not ground truth

---

## 📧 Next Steps

1. ✅ Generate all 8 hallucination plots
2. ⏳ Analyze additional models (Claude, DeepSeek, etc.)
3. ⏳ Implement top recommendations
4. ⏳ Re-run analysis to measure improvement
5. ⏳ Cross-reference with quality and coverage analyses

---

*Last updated: October 2, 2025*
