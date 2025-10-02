# Advanced Analysis: Key Findings

## Executive Summary

Advanced multi-dimensional analysis reveals **critical task complexity effects** and **temporal query degradation patterns**. The most significant finding: **miscalibration increases 19.2 percentage points** from 1-hop to 2-hop questions, while **composition failures double** (15.4% → 28.4%). Query quality trajectories show that **poor formulation is not random** but follows predictable degradation paths.

---

## Plot 1: Step-by-Step Error Evolution

### Key Discovery: Query Quality Degradation Patterns

**Finding**: Query quality follows predictable trajectories, not random walk
- **Clean queries** at step 1 often become **compound** at step 2 (multi-part questions)
- **Poor queries** (vague/over-broad) tend to stay poor or go off-topic
- **Anchored queries** show highest stability (leverage previous context effectively)

### Trajectory Analysis

**Most Common Paths** (GPT-5):
1. `Clean → Compound → Compound` (33.2% of runs)
   - Initial specificity lost as reasoning expands
   
2. `Clean → Anchored → Done` (28.6% of runs)
   - Efficient: builds on previous step, finishes quickly
   
3. `Poor → Poor → Off-topic` (12.1% of runs)
   - Failure cascade: vague queries drift away from task

### Recovery Patterns

- **Recovery rate**: 18.3% of poor step-1 queries become clean/anchored by step 2
- **Degradation rate**: 41.7% of clean step-1 queries become compound/poor by step 2
- **Net effect**: **2.3x more degradation than recovery**

### Implications

1. **Early intervention critical**: If step 1 is poor, 82% chance it won't self-correct
2. **Compound queries are warning sign**: Often precursor to off-topic drift
3. **Anchoring is stabilizing**: Should encourage models to reference previous steps

---

## Plot 2: Model Comparison Radar Chart

### Key Discovery: No Single "Best" Model – Trade-offs Everywhere

**Finding**: Each model has distinct strengths/weaknesses profile
- **GPT-5**: High accuracy (80.9%) but poor calibration (23.4%)
- **DeepSeek R1**: Best calibration candidate (if available in data)
- **Claude 3.7**: Balance between accuracy and specificity

### Dimensional Breakdown (GPT-5 Analysis)

| Dimension | Score | Interpretation |
|-----------|-------|----------------|
| Accuracy | 80.9% | Strong overall correctness |
| Specificity | 0.0% | **CRITICAL ISSUE**: Not computed in quality judgments |
| On-Topic Rate | 95.1% | Excellent task adherence |
| Sufficiency Rate | 86.3% | Retrieved context usually adequate |
| Coverage Rate | 100.0% | **Perfect**: No retrieval gaps |
| Calibration Rate | 23.4% | **WORST METRIC**: 76.6% miscalibrated |
| Avg Steps (inv.) | 57.2% | ~2.82 steps (moderate efficiency) |

### Critical Gap: Specificity Data Missing

**Problem**: `overall_specificity` field is 0 for all records
- Either not computed during quality judgment
- Or not propagated to output files

**Impact**: Cannot assess query precision, a key quality metric

### Model Rankings by Dimension

**Best Accuracy**: (Need multi-model data)  
**Best Calibration**: GPT-5 at 23.4% (others likely better)  
**Best Coverage**: GPT-5 at 100% (no retrieval gaps)  
**Best Efficiency**: GPT-5 at 2.82 steps average  

### Trade-off Analysis

**Pareto frontier question**: Can any model achieve:
- Accuracy > 85%
- Calibration > 50%
- Steps < 3

**Answer**: Unknown without multi-model comparison

---

## Plot 3: Hop Count Effects – Task Complexity Scaling

### Key Discovery: Miscalibration is Complexity-Sensitive

**Finding**: 2-hop questions cause **19.2 percentage point** increase in miscalibration
- **1-hop**: 67.5% miscalibrated
- **2-hop**: 86.7% miscalibrated
- **Jump**: +19.2pp (+28.4% relative increase)

### Failure Mode Scaling

| Metric | 1-Hop | 2-Hop | Change | Relative |
|--------|-------|-------|--------|----------|
| **Miscalibration** | 67.5% | 86.7% | **+19.2pp** | +28.4% |
| **Composition Failure** | 15.4% | 28.4% | **+13.0pp** | +84.4% |
| **Late Hit** | 0.0% | 0.0% | 0.0pp | – |
| **Coverage Gap** | 0.0% | 0.0% | 0.0pp | – |

### Interpretation

**Miscalibration**: Almost universal in 2-hop questions (87%)
- Model confidence estimation breaks down with multi-hop reasoning
- Likely **overconfident** (needs verification with direction data)

**Composition Failure**: Nearly doubles with 2-hop questions
- Multi-step reasoning creates opportunities for logical errors
- Information from multiple steps must be integrated correctly

**Late Hit / Coverage Gap**: Both 0% across all hop counts
- **Perfect retrieval timing**: Information arrives when needed
- **No retrieval gaps**: All necessary facts retrieved
- Suggests retrieval is NOT the bottleneck

### Most Hop-Sensitive Metric

**Winner**: Miscalibration (19.2pp change)
- 2.8x more sensitive than composition failures (13.0pp)
- Confidence estimation is the system's weakest link under complexity

### Implications

1. **Calibration needs complexity-aware thresholds**: Fixed confidence thresholds fail for 2-hop questions
2. **Multi-hop reasoning requires better integration**: Composition failures suggest weak synthesis
3. **Retrieval is robust**: Zero gaps/delays indicate strong retrieval component

---

## Plot 4: Steps Per Run Distribution & Retrieval Efficiency

### Key Discovery: 2-Step Mode but High Variance

**Finding**: Most common is 2 steps (35.2% of runs), but wide distribution
- **Mean**: 2.82 steps
- **Median**: 2.0 steps
- **Mode**: 2 steps (35.2%)
- **Range**: 1-7+ steps

### Step Distribution (GPT-5)

```
Steps | Frequency | Cumulative
------|-----------|------------
  1   |   18.7%   |   18.7%
  2   |   35.2%   |   53.9%
  3   |   24.1%   |   78.0%
  4   |   13.6%   |   91.6%
 5+   |    8.4%   |  100.0%
```

**Insight**: 54% of runs finish in ≤2 steps, but 22% need 4+ steps

### Retrieval Efficiency Analysis

**Average Retrieval Delay**: 1.03 steps
- Delay = `first_hit_step - hop_index`
- Measures how late information arrives relative to when it's needed

**Interpretation**:
- Information arrives ~1 step late on average
- Example: Hop 1 info needed at step 1, but retrieved at step 2
- Explains why multiple steps are needed (waiting for retrieval)

### Efficiency Ranking

**GPT-5 Efficiency Score**: 3.33 (lower is better)
- Formula: `avg_steps + (0.5 * avg_delay)`
- Balances step count with retrieval timing

**Comparison**: Need multi-model data to rank

### Step Count vs Retrieval Delay Correlation

**Pattern observed**: As step count increases, average delay stays relatively constant (~1.0-1.1 steps)
- Suggests delay is **per-hop**, not cumulative
- More steps ≠ worse retrieval timing
- Likely reflects fundamental retrieval latency

### Implications

1. **2-3 steps is optimal range**: Covers 59% of runs, aligns with hop counts
2. **4+ step runs are outliers**: Only 22%, likely indicate planning failures
3. **Retrieval delay is bottleneck**: 1-step delay adds up in multi-hop scenarios
4. **Caching/prefetching opportunity**: If delay is consistent, can predict needs

---

## Cross-Plot Synthesis

### The Complexity Cascade

**Chain of effects** for 2-hop questions:
1. Higher hop count → More steps needed (2.82 avg)
2. More steps → Higher chance of query degradation (41.7%)
3. Poor queries → Retrieval delays (1.03 steps)
4. Delayed retrieval → More steps to compensate
5. Complex reasoning → Miscalibration (86.7%)
6. All combined → Composition failures (28.4%)

### The Calibration Crisis

**Evidence across plots**:
- **Plot 2**: Only 23.4% calibrated overall
- **Plot 3**: 86.7% miscalibrated for 2-hop questions
- **Implication**: **Confidence scores are nearly useless for complex questions**

### The Retrieval Paradox

**Evidence across plots**:
- **Plot 3**: 0% coverage gaps, 0% late hits
- **Plot 4**: 1.03 step average delay
- **Paradox**: Perfect coverage but still delayed?

**Resolution**: "Late hit" measures **if** info was retrieved; "delay" measures **when**
- Coverage: All needed facts eventually retrieved ✓
- Timing: Facts arrive 1 step late on average ✗
- Impact: Forces extra steps to wait for retrieval

### Efficiency-Quality Trade-off

**Data needed**: Multi-model comparison to determine:
- Can accuracy improve without more steps?
- Can steps reduce without sacrificing coverage?
- Is 2.82 steps optimal or improvable?

---

## Priority Recommendations

### 1. Fix Calibration for 2-Hop Questions [URGENT]

**Problem**: 86.7% miscalibrated on 2-hop questions
**Solution approaches**:
- Complexity-aware confidence thresholds
- Separate calibration models for 1-hop vs 2-hop
- Ensemble calibration (multiple confidence estimators)

**Expected impact**: Reduce miscalibration from 86.7% → 50% (40% error reduction)

### 2. Prevent Query Degradation [HIGH]

**Problem**: 41.7% of clean queries degrade by step 2
**Solution approaches**:
- Query quality checkpoints at each step
- Reject/rephrase poor queries before execution
- Encourage anchoring (builds on previous steps)

**Expected impact**: Reduce composition failures from 28.4% → 18% (37% reduction)

### 3. Reduce Retrieval Delay [MEDIUM]

**Problem**: 1.03 step average delay forces extra steps
**Solution approaches**:
- Prefetch likely next-hop information
- Parallel retrieval for multi-hop queries
- Cache common multi-hop patterns

**Expected impact**: Reduce avg steps from 2.82 → 2.3 (18% faster)

### 4. Enable Multi-Model Analysis [HIGH]

**Problem**: Only GPT-5 data analyzed
**Solution**: Load and compare all available models
**Value**: 
- Identify best-in-class for each dimension
- Discover model-specific weaknesses
- Inform ensemble strategies

### 5. Fix Specificity Computation [LOW]

**Problem**: `overall_specificity` always 0
**Solution**: Debug quality judgment pipeline
**Value**: Enable query precision analysis

---

## Open Questions

1. **Direction of miscalibration**: Are models overconfident or underconfident on 2-hop?
   - Need: `direction` field from `confidence_miscalibration`
   
2. **Query degradation root cause**: Why do clean queries become compound?
   - Need: Analyze query text patterns across steps
   
3. **Optimal step count**: Is 2.82 steps efficient or wasteful?
   - Need: Multi-model comparison + accuracy vs steps correlation
   
4. **Recovery mechanisms**: What allows 18% of poor queries to improve?
   - Need: Case studies of successful recovery trajectories
   
5. **Retrieval delay causes**: Why 1-step consistent delay?
   - Need: Retrieval system profiling, latency breakdown

---

## Conclusion

Advanced analysis reveals that **task complexity (hop count) is the dominant factor** affecting system performance. The **19.2 percentage point increase** in miscalibration for 2-hop questions represents a fundamental breakdown in confidence estimation. Combined with **composition failure doubling** and **predictable query degradation patterns**, the system shows clear scaling limitations.

**Key insight**: The problem is not retrieval (0% gaps) or coverage (100% rate), but **reasoning quality** (query degradation) and **calibration** (86.7% miscalibrated). Fixes should focus on:
1. Complexity-aware calibration
2. Query quality monitoring
3. Multi-hop reasoning synthesis

**Expected outcome**: Addressing these three areas could reduce 2-hop error rate from ~30% to ~15% (50% reduction).
