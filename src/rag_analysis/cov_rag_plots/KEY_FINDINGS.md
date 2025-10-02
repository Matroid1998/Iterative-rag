# Coverage Gap Analysis - Key Findings

**Generated**: October 2, 2025  
**Data Source**: RAG analysis output files from multiple models

---

## Executive Summary

Analysis of 2,628+ RAG runs across 5 models reveals three primary retrieval failure modes:
1. **Coverage Gaps** (2.2-4.9%): System never retrieves needed documents
2. **Anchor Carry-Drop** (2.9-25.7% at step 2): Key entities lost between steps
3. **Late Hits** (8.9-17.6%): Documents retrieved later than optimal

**Critical Finding**: Coverage gaps reduce accuracy to 34.8%, compared to 80%+ when absent.

---

## Detailed Findings by Plot

### 1. Late Hit Timing Distribution

**Key Insights:**
- **Hop 1**: 13.6% of retrievals are delayed (351/2575)
  - Mean delay: 0.18 steps
  - Max delay: 4 steps
  
- **Hop 2**: Only 3.6% are delayed (44/1219)
  - Mean delay: -0.74 steps (typically retrieved EARLY)
  - Max delay: 3 steps

**Interpretation**: Hop 1 documents are more likely to be retrieved late, but hop 2 is often retrieved ahead of schedule, suggesting the system may be jumping ahead or retrieving too broadly.

---

### 2. Model Coverage Rates

**Coverage Gap Rates** (% of runs with missed hops):
| Model | Coverage Gap | Late Hit |
|-------|-------------|----------|
| **GPT-5** | 4.9% ⚠️ | 8.9% ✓ |
| DeepSeek R1 | 4.0% | 14.2% |
| Claude 3.7 Sonnet | 2.3% ✓ | 15.9% |
| Claude 3.7 Sonnet + Reasoning | 2.5% | 17.6% |
| Mistral Large | 2.8% | 16.0% |

**Key Finding**: 
- **GPT-5 has the highest coverage gap rate (4.9%)** but the lowest late hit rate
- **Claude models have the lowest coverage gaps** but higher late hit rates
- Trade-off: Models that miss fewer hops may retrieve them later

---

### 3. Anchor Carry-Drop by Step

**Step 2 Carry-Drop Rates** (most critical step):
| Model | Step 2 Rate | Step 3 Rate | Trend |
|-------|-------------|-------------|-------|
| **Mistral Large** | 25.7% ⚠️⚠️ | 16.7% | High degradation |
| Claude 3.7 Sonnet | 19.7% ⚠️ | 10.2% | Moderate |
| DeepSeek R1 | 18.5% ⚠️ | 9.9% | Moderate |
| Claude 3.7 + Reasoning | 14.5% | 8.0% | Improved |
| **GPT-5** | 8.8% ✓ | 1.2% ✓ | Excellent |

**Critical Insights**:
1. **Mistral loses anchors in 1 out of 4 step 2 queries** - major weakness
2. **GPT-5 maintains anchors best** - only 8.8% drop at step 2
3. **Anchor carry improves after step 2** across all models
4. **Step 2 is the critical failure point** for anchor propagation

---

### 4. Accuracy Linkage to Coverage Issues

**Impact on Correctness**:

| Issue Type | Accuracy When Present | Prevalence in Incorrect | Prevalence in Correct |
|------------|----------------------|------------------------|----------------------|
| **Coverage Gap** | 34.8% ⚠️⚠️ | 28.6% | 4.9% |
| Late Hit | 80.3% | 36.5% | 47.8% |
| Anchor Drop | 80.8% | 35.0% | 47.3% |

**Critical Findings**:
1. **Coverage gaps are devastating**: Only 34.8% accuracy when present
2. **Coverage gaps are 5.8× more common in wrong answers** (28.6% vs 4.9%)
3. **Late hits and anchor drops are less harmful**: Still achieve ~80% accuracy
4. **Priority for fixing**: Coverage gaps >> Anchor drops ≈ Late hits

---

### 5. Missed Hop Patterns by Complexity

**1-hop questions** (n=1,379):
- 2.2% miss the single hop
- Relatively straightforward

**2-hop questions** (n=1,249):
- 2.5% miss hop 1
- 2.5% miss hop 2
- **Overall gap rate: 5.0%** (more than double 1-hop questions)

**Key Finding**: Multi-hop questions are **2.3× more likely to have coverage gaps**. Both hops are equally likely to be missed.

---

### 6. Anchor Carry Temporal Pattern

**Aggregated across all models**:

| Step | Total Queries | Carry-Drop Count | Rate |
|------|--------------|------------------|------|
| 2 | 1,674 | 298 | 17.8% ⚠️ |
| 3 | 867 | 85 | 9.8% |
| 4 | 567 | 51 | 9.0% |
| 5 | 428 | 31 | 7.2% |

**Trend**: **DECREASING** at -3.25% per step

**Interpretation**: 
- **Step 2 is the danger zone**: Nearly 1 in 5 queries lose anchors
- **Anchor carry improves with experience**: Later steps maintain anchors better
- **Systems learn**: Once past step 2, anchor propagation stabilizes

---

## Actionable Recommendations

### Priority 1: Fix Coverage Gaps (High Impact)
- **Target**: Reduce from 2-5% to <1%
- **Focus on**: 2-hop questions (5% gap rate)
- **ROI**: Could improve accuracy from 34.8% to 80%+ on affected runs
- **Models needing most help**: GPT-5 (4.9%), DeepSeek R1 (4.0%)

### Priority 2: Improve Step 2 Anchor Carry (High Frequency)
- **Target**: Reduce step 2 carry-drop from 17.8% to <10%
- **Focus on**: Mistral Large (25.7% drop) and Claude models (14-20% drop)
- **Method**: Explicit anchor injection in query generation
- **Expected benefit**: Reduce wasted retrievals, improve efficiency

### Priority 3: Optimize Late Hit Detection (Efficiency)
- **Target**: Reduce hop 1 late hits from 13.6% to <8%
- **Focus on**: Better initial query formulation
- **Expected benefit**: Faster convergence, fewer steps needed

---

## Model-Specific Recommendations

### GPT-5
- ✓ **Strengths**: Best anchor carry (8.8%), lowest late hits (8.9%)
- ⚠️ **Weakness**: Highest coverage gaps (4.9%)
- **Action**: Improve initial retrieval breadth

### Claude 3.7 Sonnet (with Reasoning)
- ✓ **Strengths**: Low coverage gaps (2.5%), reasoning helps anchor carry (14.5% vs 19.7%)
- ⚠️ **Weakness**: Higher late hits (17.6%)
- **Action**: Earlier retrieval of hop 1 documents

### Mistral Large
- ⚠️ **Weakness**: Poor anchor carry at step 2 (25.7%)
- **Action**: Critical need for anchor preservation mechanism

### DeepSeek R1
- ✓ **Strengths**: Balanced performance
- ⚠️ **Weakness**: Moderate coverage gaps (4.0%) and anchor drop (18.5%)
- **Action**: Improve both coverage and anchor mechanisms

---

## Technical Debt & Future Work

1. **Question complexity analysis**: Investigate why 2-hop questions have 2.3× higher gap rates
2. **Step 2 phenomenon**: Deep dive into why step 2 is the anchor loss danger zone
3. **Early hop 2 retrieval**: Understand why hop 2 is often retrieved early (median -1 step)
4. **Coverage gap root causes**: Analyze the 89 runs with gaps to identify query patterns
5. **Anchor recovery**: Study cases where anchors are lost but answers are still correct (80.8%)

---

## Conclusion

The analysis reveals **coverage gaps as the primary failure mode** (65% accuracy drop), with **step 2 anchor loss** as a secondary efficiency problem (18% occurrence). Fixing coverage gaps should be the immediate priority, targeting 2-hop questions and improving initial retrieval breadth, especially for GPT-5 and DeepSeek R1.

The positive finding is that **late hits and anchor drops are recoverable** (still achieving ~80% accuracy), suggesting the system has some resilience. However, the 25.7% anchor drop rate in Mistral at step 2 indicates significant room for improvement in query generation consistency.
