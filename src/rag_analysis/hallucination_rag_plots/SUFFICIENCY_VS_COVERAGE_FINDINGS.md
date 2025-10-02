# Sufficiency vs Coverage Analysis - Key Findings

## Plot: 2_sufficiency_vs_coverage.png

This plot shows the relationship between sufficiency scores and hop coverage estimates across different models, colored by miscalibration direction (OK, Underconfident, Overconfident). Point size indicates the number of unsupported claims.

---

## Key Findings by Model

### GPT-5: Best Overall Performance
- **Highest sufficiency scores**: 0.975 for OK cases
- **Lowest unsupported claims**: 0.11 average
- Shows strong correlation between high sufficiency and proper calibration
- Most consistent performer across all metrics

### Claude Models: High Underconfidence but Good Quality
- Show **high underconfidence rates** but maintain **good sufficiency** when calibrated correctly
- Claude 3.7 Sonnet: 730 underconfident cases with 0.877 avg sufficiency
- Claude 3.7 Sonnet + Reasoning: 694 underconfident cases with 0.870 avg sufficiency
- Both maintain high coverage estimates (>0.95) even when underconfident
- The reasoning variant shows slightly better calibration (358 OK vs 338 OK)

### DeepSeek R1: Balanced but Overconfident
- Has the **most balanced calibration** (545 OK cases - highest among all models)
- **Struggles with overconfidence**: 467 overconfident cases
- Overconfident cases show low sufficiency (0.500) and high unsupported claims (1.60)
- When properly calibrated, maintains excellent sufficiency (0.899)

### GPT-4o: Extreme Underconfidence
- Has the **highest underconfidence rate** (757 cases - 63.8% of all responses)
- Despite underconfidence, **maintains good sufficiency** (0.875 avg)
- Only 214 OK cases (18.0%) - most conservative model
- Overconfident cases have lowest sufficiency (0.482) and highest unsupported claims (2.33)

### Mistral Large: Quality Issues with Overconfidence
- Shows **lower sufficiency scores for overconfident cases** (0.449)
- 410 overconfident cases (34.6%) with 1.95 avg unsupported claims
- Second-highest overconfidence rate after DeepSeek R1
- When calibrated correctly (252 OK cases), shows good sufficiency (0.922)

---

## Overall Patterns

### Universal Trend
All models show a clear pattern: **overconfident cases have much lower sufficiency scores and higher unsupported claims**, confirming the strong relationship between miscalibration and response quality.

### Coverage vs Sufficiency
- **OK cases**: High coverage (0.95-0.99) + High sufficiency (0.90-0.98)
- **Underconfident cases**: Very high coverage (0.97-0.99) + Good sufficiency (0.78-0.88)
- **Overconfident cases**: Lower coverage (0.76-0.92) + Poor sufficiency (0.45-0.67)

### Unsupported Claims Pattern
- OK cases: 0.11-0.57 unsupported claims
- Underconfident cases: 0.70-1.21 unsupported claims  
- Overconfident cases: 1.09-2.33 unsupported claims

The data clearly shows that **overconfidence is a strong predictor of poor response quality**, while underconfidence often occurs even when the model has sufficient information.

---

## Actionable Insights

1. **For GPT-5**: Leverage its high sufficiency scores; can be used as a baseline for quality
2. **For Claude models**: Work on reducing underconfidence without sacrificing quality
3. **For DeepSeek R1**: Focus on reducing overconfidence in complex multi-hop scenarios
4. **For GPT-4o**: Can afford to be more confident given its high sufficiency scores
5. **For Mistral Large**: Critical need to improve sufficiency in overconfident cases

---

*Generated: October 2, 2025*
*Data: 7,114 responses across 6 models, 1,186 unique questions*
