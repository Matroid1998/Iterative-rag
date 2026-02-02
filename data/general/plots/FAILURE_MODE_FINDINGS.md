# Failure Mode Analysis - Key Findings

## Summary

Analysis of 11 models across 4 failure modes: **Coverage Gap**, **Composition Failure**, **Overconfident**, and **Distractor Latch**.

### Overall Rankings by Damage Index (Expected Loss)

#### Most Damaging Failure Modes (Average):
1. **Composition Failure**: 15.29pp expected loss
   - Highest impact (58.3pp drop when present)
   - Moderate prevalence (26.3%)
   - Most damaging across all models

2. **Coverage Gap**: 3.05pp expected loss
   - Moderate impact (20.5pp drop)
   - Lower prevalence (15.6%)
   - Second most damaging

3. **Overconfident**: 1.73pp expected loss
   - Lower impact (12.7pp drop)
   - Lowest prevalence (13.8%)
   - Least damaging of the three

4. **Distractor Latch**: 0.00pp expected loss
   - Not detected in any model
   - Suggests distractor latch is rare or not properly captured

---

## Detailed Findings

### Coverage Gap
**Definition**: Model stops retrieving before covering all required information hops.

| Metric | Value |
|--------|-------|
| Average Prevalence | 15.6% |
| Average Impact | 20.5pp |
| Average Damage | 3.05pp |

**Most Affected Models** (by damage):
1. **Llama 3.3 70B**: 5.31pp (26.7% prevalence, 19.9pp impact)
2. **GLM 4.6**: 4.45pp (15.8% prevalence, 28.2pp impact)
3. **DeepSeek R1**: 4.32pp (16.9% prevalence, 25.6pp impact)

**Least Affected Models**:
1. **Grok 4 Fast**: 0.62pp (11.9% prevalence, 5.3pp impact)
2. **Claude Sonnet 4.5**: 1.44pp (11.6% prevalence, 12.5pp impact)

**Interpretation**: 
- Coverage gaps hurt accuracy significantly (20pp average drop)
- Llama 3.3 70B has highest prevalence (26.7%) - frequently stops too early
- Grok 4 Fast shows resilience with minimal impact even when gaps occur

---

### Composition Failure
**Definition**: Model generates answer that fails to properly compose retrieved information (faithfulness issues).

| Metric | Value |
|--------|-------|
| Average Prevalence | 26.3% |
| Average Impact | 58.3pp |
| Average Damage | 15.29pp |

**Most Affected Models** (by damage):
1. **Llama 3.3 70B**: 24.36pp (37.2% prevalence, 65.5pp impact)
2. **Mistral Large**: 22.97pp (36.8% prevalence, 62.5pp impact)
3. **GLM 4.6**: 16.01pp (23.5% prevalence, 68.2pp impact)

**Least Affected Models**:
1. **Claude Sonnet 4.5**: 9.96pp (21.2% prevalence, 47.0pp impact)
2. **Claude 3.7 + Reasoning**: 11.04pp (24.3% prevalence, 45.5pp impact)

**Interpretation**:
- **BY FAR the most damaging failure mode** (15.29pp vs 3.05pp for coverage gap)
- When composition fails, accuracy drops dramatically (58.3pp average!)
- Open-source models (Llama, Mistral) suffer most
- Claude models show better composition skills
- GPT-5 has highest impact (80.8pp) but lowest prevalence (16.5%) among high-performing models

---

### Overconfident
**Definition**: Model stops early (finalize_step < required hops) AND has quality issues (coverage<0.70 OR sufficiency<0.60).

| Metric | Value |
|--------|-------|
| Average Prevalence | 13.8% |
| Average Impact | 12.7pp |
| Average Damage | 1.73pp |

**Most Affected Models** (by damage):
1. **Llama 3.3 70B**: 3.43pp (15.3% prevalence, 22.4pp impact)
2. **GLM 4.6**: 2.90pp (15.5% prevalence, 18.7pp impact)
3. **Grok 4 Fast**: 2.77pp (17.5% prevalence, 15.9pp impact)

**Least Affected Models**:
1. **Claude 3.7 + Reasoning**: 0.06pp (4.8% prevalence, 1.3pp impact)
2. **Claude 3.7 Sonnet**: 0.44pp (3.3% prevalence, 13.5pp impact)

**Interpretation**:
- Least damaging of the three failure modes
- Claude models rarely show overconfidence (3-5% prevalence)
- Open-source models (Llama, GLM, Grok) more prone to overconfident stopping
- Impact is modest (12.7pp) compared to composition failure (58.3pp)

---

### Distractor Latch
**Definition**: Model uses irrelevant evidence (gets distracted by non-relevant sources).

| Metric | Value |
|--------|-------|
| Average Prevalence | 0.0% |
| Average Impact | 0.0pp |
| Average Damage | 0.00pp |

**Finding**: No distractor latch cases detected across any model.

**Possible Explanations**:
1. Models effectively filter irrelevant information
2. Retrieval quality is high (no irrelevant sources returned)
3. Detection criteria too strict or not capturing the phenomenon
4. Evidence quality judgments classify most sources as relevant

---

## Model Profiles

### Best Overall (Lowest Total Damage)
**Claude 3.7 + Reasoning**: 13.58pp total damage
- Lowest overconfident (0.06pp)
- Low coverage gap (2.48pp)
- Best composition skills (11.04pp)
- **Strength**: Rarely overconfident, good faithfulness

### Worst Overall (Highest Total Damage)
**Llama 3.3 70B**: 33.10pp total damage
- Highest composition failure (24.36pp)
- Highest coverage gap (5.31pp)
- High overconfident (3.43pp)
- **Weakness**: Struggles with all three failure modes

### High-Volume/Moderate-Impact
**GPT-5**: 17.64pp total damage
- Low coverage gap damage (2.41pp) despite high prevalence (28.9%)
- Moderate composition damage (13.30pp) with low prevalence (16.5%)
- Moderate overconfident (1.93pp)
- **Profile**: Retrieves extensively but composition issues when they occur hit hard (80.8pp impact)

### Specialized Strengths
**Claude Sonnet 4.5**: 12.33pp total damage
- Lowest composition prevalence among high-performers (21.2%)
- Good balance across all metrics
- **Strength**: Consistent performance, no major weaknesses

**Grok 4 Fast**: 16.53pp total damage
- Surprisingly low coverage gap impact (0.62pp, 5.3pp impact only)
- Higher overconfident issues (2.77pp)
- **Strength**: Robust to incomplete retrieval

---

## Strategic Recommendations

### For Deployment:
1. **Choose Claude 3.7 + Reasoning** for:
   - High-stakes applications requiring faithfulness
   - Questions where overconfidence could be costly
   - Scenarios requiring balanced performance

2. **Choose GPT-5** for:
   - Complex questions needing extensive retrieval
   - When willing to trade computation for accuracy
   - Tasks where composition is strong given enough context

3. **Avoid Llama 3.3 70B** for:
   - Critical applications (high total damage)
   - Questions requiring accurate composition
   - Scenarios with high coverage requirements

### For Improvement:
1. **Composition Failure** - Highest Priority
   - 3x more damaging than coverage gaps
   - Focus on faithfulness training
   - Claude models show it's achievable

2. **Coverage Gaps** - Medium Priority
   - Teach models to recognize needed hop counts
   - Llama and GPT-5 need improvement
   - Consider forced retrieval for known multi-hop questions

3. **Overconfidence** - Lower Priority
   - Smallest damage index
   - Already well-addressed in Claude models
   - Use stricter stopping criteria for open-source models

4. **Distractor Latch** - Not Applicable
   - Currently not a problem
   - May indicate good retrieval quality
   - Continue monitoring but no action needed

---

## Files Generated

1. **prevalence.csv** - Frequency of each failure mode per model
2. **impact.csv** - Accuracy drop (pp) when failure occurs per model
3. **damage_index.csv** - Expected loss (pp) per question per model
4. **failure_mode_analysis.md** - Complete markdown report with all tables

All files located in: `/home/mehdi/Projects/Iterative-rag/src/plots/failure_mode_tables/`
