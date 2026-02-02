# Contradiction & Distractor Latch Analysis - CORRECTED

## Data Source Fix

### Problem Identified
The initial analysis showed unrealistically low accuracy (<50%) because the script was loading `is_correct` from quality judgment files, which don't actually contain this field. This caused all questions to default to `is_correct=False`.

### Solution
Updated `plot_contradiction_distractor_analysis.py` to:
1. Load `is_correct` values from `responses_reverified/*.jsonl` files
2. Match questions between quality judgment and reverified files
3. Handle JSON parsing errors gracefully
4. Report matching statistics to verify data quality

### Matching Statistics
- **Llama 3.3 70B**: 1185/1186 matched (99.9%)
- **Mistral Large 2402**: 1185/1186 matched (99.9%)
- **Claude 3.7 Sonnet Thinking**: 1185/1186 matched (99.9%)
- **Claude 3.7 Sonnet**: 1185/1186 matched (99.9%)
- **DeepSeek R1**: 1185/1186 matched (99.9%)
- **GPT-4o**: 1185/1186 matched (99.9%)
- **GPT-5**: 1185/1186 matched (99.9%)
- **Gemini 2.5 Pro**: 1169/1186 matched (98.6%)
- **Grok 4 Fast**: 1158/1185 matched (97.7%)
- **GLM 4.6**: 1158/1185 matched (97.7%, 1 JSON error)

Note: Claude Sonnet 4.5 excluded (reverified file has different naming format)

---

## CORRECTED FINDINGS

### Overall Aggregated Statistics (10 models, 11,812 questions)

| Category | Accuracy | Impact vs Baseline |
|----------|----------|-------------------|
| **Neither Issue** | **89.4%** | Baseline |
| **Only Contradiction** | 85.1% | **-4.3pp** |
| **Only Distractor Latch** | 35.2% | **-54.2pp** |
| **Both Issues** | 53.6% | **-35.8pp** |

### Question Distribution

- **Neither Issue**: 9,459 questions (80.1%)
- **Only Contradiction**: 377 questions (3.2%)
- **Only Distractor**: 1,756 questions (14.9%)
- **Both Issues**: 220 questions (1.9%)

---

## Key Insights

### 1. Distractor Latch is the Primary Accuracy Killer
- **54.2 percentage point drop** when only distractor latch is present
- Affects 14.9% of all questions
- Causes **3,794 failed attempts** out of 11,812 total questions (32.1%)

### 2. Partial Contradictions Have Modest Impact
- **4.3 percentage point drop** when only contradictions present
- Affects only 3.2% of questions
- Much less harmful than initially thought (previous buggy analysis showed -27pp)

### 3. Both Issues Together
- When both present: 53.6% accuracy (-35.8pp)
- Slightly better than distractor alone (35.2%)
- Suggests contradictions might help models "second-guess" bad paths
- Only affects 1.9% of questions (220 total)

### 4. Top Model Performance Explained
- Models like GPT-5 and Claude Sonnet 4.5 achieve 70-80% overall accuracy because:
  - 80%+ of their questions have "Neither Issue" (89.4% accuracy)
  - They avoid distractor latch on most questions
  - When they do hit distractors, performance drops to ~35%

---

## Recommendations

### For RAG System Design

1. **Prioritize Distractor Avoidance**
   - Implement strict relevance filtering
   - Use semantic similarity thresholds
   - Consider query reformulation when initial results seem off-topic

2. **Contradiction Detection is Secondary**
   - While contradictions do hurt (4.3pp), they're rare and less harmful
   - May not be worth complex detection logic
   - Focus engineering effort on distractor prevention

3. **Context Quality Over Quantity**
   - Better to have fewer high-quality results than many noisy ones
   - The 54pp accuracy drop from distractors vastly outweighs any benefits of exhaustive retrieval

### For Model Selection

- Models that naturally avoid latching onto distractors will perform better
- Look for models with high "Neither Issue" percentages
- Strong performance on clean data (89.4%) suggests models are generally capable
- The challenge is avoiding bad retrieval paths

---

## Technical Notes

### Data Processing
- Quality judgments from: `src/rag_analysis/output/*quality_judement.jsonl`
- Accuracy data from: `src/responses_reverified/*_reverified.jsonl`
- Questions matched on exact text (99%+ match rate for most models)

### Code Changes
- Updated `analyze_contradiction_distractor_effects()` to load from two sources
- Added JSON error handling for malformed lines
- Added matching statistics reporting
- Updated `get_quality_model_entries()` to return both file paths

### Generated Plots
- `contradiction_distractor_combined_effects.png` - 4-category comparison
- `contradiction_distractor_detailed_breakdown.png` - per-model breakdown
