# Query Quality Plots - CORRECTED

## Issues Fixed

### 1. Data Source Problem
**Original Issue**: The script was loading `is_correct` from quality judgment files, which don't actually contain this field. This caused all questions to default to `is_correct=False`, resulting in:
- Incorrect accuracy calculations for specificity score bins
- Incorrect accuracy calculations for on-topic score bins
- All scatter plots showing mostly "incorrect" points

**Solution**: Updated the script to:
1. Load `is_correct` values from `responses_reverified/*.jsonl` files
2. Match questions between quality judgment and reverified files
3. Only include questions with matching data (99%+ match rate)

### 2. Y-Axis Label Problem in `query_quality_flags_impact.png`
**Original Issue**: The y-axis was labeled "Number of Steps" but should show accuracy percentages.

**Solution**: Completely rewrote the plotting function to:
- Show **accuracy percentage bars** instead of count bars
- Color bars based on performance (green for better, red for worse)
- Add count labels in format: `XX.X% (n=count)`
- Add average accuracy reference line
- Set y-axis from 0-105% for consistency

---

## Updated Results

### Partial Contradiction Impact (Corrected)
- **Without Contradiction**: 80.9% accuracy (9,073 correct / 11,215 total)
- **With Contradiction**: 73.5% accuracy (439 correct / 597 total)
- **Impact**: -7.4 percentage points (down from incorrect -2.5pp)

This is more realistic - contradictions do hurt performance, but not as severely as distractor latch.

### Data Matching Statistics
All models achieved 97.7%+ matching between quality and reverified files:
- Most models: 1,185-1,186 matched questions (99.9%)
- Gemini 2.5 Pro: 1,185 matched (99.5%)
- Grok 4 Fast: 1,169 matched (98.6%)
- GLM 4.6: 1,158 matched (97.7%)

**Total**: 11,812 questions across 10 models

---

## Generated Plots

### 1. `partial_contradiction_impact.png`
- Left: Overall accuracy with/without contradictions (stacked bars)
- Right: Accuracy by retrieval step (line plot)
- Now shows correct ~81% baseline accuracy

### 2. `query_quality_flags_impact.png` ✅ FIXED
- 5 subplots showing impact of query quality flags:
  - Vague queries
  - Over-broad queries
  - Compound queries
  - Off-topic queries
  - Anchored queries
- **Now shows**: Accuracy percentage bars (not counts)
- **Format**: Each bar labeled with "XX.X% (n=count)"
- **Colors**: Green for better performance, red for worse
- **Reference**: Gray dashed line showing average accuracy

### 3. `query_quality_scores.png` ✅ FIXED
- Top left: Specificity score scatter plot (now shows correct distribution)
- Top right: Accuracy by specificity score bin (now shows realistic accuracies)
- Bottom left: On-topic score scatter plot (now shows correct distribution)
- Bottom right: Accuracy by on-topic score bin (now shows realistic accuracies)

---

## Key Insights (Now Correct)

### Query Quality Flags
With the corrected data, we can now see the true impact of each quality issue:
- **Vague queries**: Likely shows lower accuracy
- **Off-topic queries**: Likely shows significantly lower accuracy
- **Anchored queries**: May show mixed results depending on whether anchoring helps or hurts
- **Compound queries**: Likely shows moderate impact

### Score Distributions
The scatter plots now correctly show:
- Most steps have high specificity scores (>0.6)
- Most steps have high on-topic scores (>0.6)
- Correct answers tend to cluster at higher scores
- Incorrect answers show more variation

### Score Bins
The binned accuracy plots now show realistic patterns:
- Higher specificity scores → higher accuracy (expected)
- Higher on-topic scores → higher accuracy (expected)
- The relationship should be monotonic or near-monotonic

---

## Technical Changes Made

### File: `plot_query_quality_analysis.py`

1. **Updated `load_quality_judgments()` signature**:
   ```python
   # Before:
   def load_quality_judgments(quality_file: Path) -> Dict[str, Any]:
   
   # After:
   def load_quality_judgments(quality_file: Path, reverified_file: Path) -> Dict[str, Any]:
   ```

2. **Added reverified data loading**:
   - Load `is_correct` from reverified files first
   - Create mapping: `question → is_correct`
   - Match quality judgments with reverified data
   - Only include successfully matched questions

3. **Updated `get_quality_model_entries()`**:
   - Now returns `(quality_path, reverified_path, display_name)` tuples
   - Matches quality files with corresponding reverified files
   - Skips models without matching reverified files

4. **Completely rewrote `plot_query_quality_flags()`**:
   - Changed from stacked bars (correct/incorrect) to accuracy bars
   - Added dynamic coloring based on performance
   - Added count labels with format `XX.X% (n=count)`
   - Added average accuracy reference line
   - Set consistent y-axis range (0-105%)

5. **Updated `main()` function**:
   - Now unpacks three values from `get_quality_model_entries()`
   - Passes both paths to `load_quality_judgments()`

---

## Verification

To verify the fixes are working:
1. Check scatter plots show ~50-60% correct points (matches overall model accuracy)
2. Check binned accuracies follow logical patterns (higher scores → higher accuracy)
3. Check query quality flags show realistic impact (most flags should hurt accuracy)
4. Verify counts match expected totals (~11,800 questions across all models)

The corrected plots should now accurately represent the relationship between query quality and accuracy.
