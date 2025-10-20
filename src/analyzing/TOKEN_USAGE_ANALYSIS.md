# Token Usage Analysis - Four Visualization Approaches

## Overview

Created 4 complementary visualizations to analyze how different models use tokens across difficulty levels (Easy, Medium, Hard). Each plot reveals different insights about model behavior.

---

## Generated Plots

### 1. **Box Plot - Token Distribution** 
**File:** `token_distribution_boxplot.png` (207 KB)

**Layout:** 11 subplots (one per model), 3×4 grid

**What it shows:**
- Box plots showing token distribution for each model across Easy/Medium/Hard
- Each box plot shows: median, quartiles, outliers
- Reveals variance in token usage (some questions trigger longer responses)

**Key Insights:**
- **Distribution shape:** Are token counts consistent or highly variable?
- **Outliers:** Which questions cause unusually long/short responses?
- **Scaling:** How does median token usage change Easy → Medium → Hard?
- **Model comparison:** Which models have tight vs wide distributions?

**Best for:**
- Understanding token usage variance per model
- Identifying outlier questions
- Seeing full data distribution, not just averages

---

### 2. **Bar Chart - Average Tokens per Category**
**File:** `token_usage_barchart.png` (165 KB)

**Layout:** Single plot with all 11 models on x-axis

**What it shows:**
- 3 bars per model (Easy, Medium, Hard) side-by-side
- Direct comparison of average token usage
- Color-coded: Green (Easy), Orange (Medium), Red (Hard)
- Separator line between non-reasoning and reasoning models

**Key Insights:**
- **Direct comparison:** Which models use most/least tokens?
- **Scaling behavior:** Do all models increase tokens with difficulty?
- **Model groups:** Clear visual separation between reasoning vs non-reasoning
- **Consistency:** Which models maintain similar token usage across difficulties?

**Best for:**
- Quick model-to-model comparison
- Identifying most efficient models
- Seeing how difficulty affects token usage per model

---

### 3. **Heatmap - Models × Categories**
**File:** `token_usage_heatmap.png` (129 KB)

**Layout:** 11 rows (models) × 3 columns (difficulties)

**What it shows:**
- Color-coded cells showing average token usage (log scale)
- Yellow (low) → Red (high) color gradient
- Numerical values in each cell
- Separator line between non-reasoning and reasoning models

**Key Insights:**
- **Pattern recognition:** Visual patterns across all models/categories at once
- **Compact view:** All data in one compact visualization
- **Hot spots:** Which model-difficulty combinations use most tokens?
- **Consistency:** Which models maintain similar colors (consistent token usage)?

**Best for:**
- Quick overview of entire dataset
- Identifying patterns across all models
- Compact visualization for presentations
- Comparing multiple dimensions simultaneously

---

### 4. **Correct vs Incorrect Side-by-Side**
**File:** `token_usage_correct_vs_incorrect.png` (202 KB)

**Layout:** 2 subplots side-by-side (Correct | Incorrect)

**What it shows:**
- Left subplot: Average tokens when answer is CORRECT (green title)
- Right subplot: Average tokens when answer is INCORRECT (red title)
- 3 bars per model (Easy, Medium, Hard) in each subplot
- Same scale for direct comparison

**Key Insights:**
- **Efficiency:** Do models use more tokens when correct or incorrect?
- **Waste detection:** Models using many tokens but still wrong
- **Confidence:** Models using fewer tokens when correct = efficient
- **Error patterns:** Do incorrect answers have different token patterns?

**Best for:**
- Understanding relationship between tokens and correctness
- Identifying inefficient reasoning patterns
- Comparing success vs failure token usage
- Detecting models that "overthink" incorrect answers

---

## Comparison Matrix

| Plot Type | Best For | Strengths | Limitations |
|-----------|----------|-----------|-------------|
| **Box Plot** | Distribution analysis | Shows variance, outliers | Takes more space |
| **Bar Chart** | Quick comparison | Easy to read, clear | No variance info |
| **Heatmap** | Pattern recognition | Compact, overview | Less precise values |
| **Correct vs Incorrect** | Efficiency analysis | Reveals waste | Doesn't show variance |

---

## Key Findings Across All Plots

### 1. Non-Reasoning vs Reasoning Models

**Non-reasoning models (Rows 1-9):**
- Token range: ~400-1,200 tokens
- Consistent across difficulties
- Tight distributions (less variance)
- Similar token usage correct vs incorrect

**Reasoning models (Rows 10-11):**
- Token range: ~7,000-20,000 tokens
- Increases with difficulty
- Wide distributions (high variance)
- Often use more tokens when incorrect

### 2. Difficulty Impact

**Easy questions:**
- All models: lowest token usage
- High success rate → efficient responses
- Tight distributions

**Medium questions:**
- Moderate increase in tokens
- More variance in responses
- Clear separation between model tiers

**Hard questions:**
- Highest token usage (especially reasoning models)
- Widest distributions
- More tokens when incorrect (overthinking failures)

### 3. Efficiency Patterns

**Most efficient:**
- GPT-4o, Gemini 2.5 Pro: Low tokens, high accuracy
- Consistent token usage across difficulties

**Most thorough:**
- Claude 3.7 Sonnet Thinking: Very high tokens, good accuracy
- DeepSeek R1: High tokens, scales with difficulty

**Struggling:**
- Llama, Mistral: Moderate tokens, lower accuracy
- More tokens on incorrect answers (inefficient failures)

---

## How to Use Each Plot

### For Paper/Presentation:

**Use Box Plot when:**
- Discussing variance in model behavior
- Showing that some questions trigger longer responses
- Demonstrating distribution shape (skewed, normal, etc.)

**Use Bar Chart when:**
- Comparing models directly
- Showing clear hierarchy of token usage
- Presenting to non-technical audience (easiest to understand)

**Use Heatmap when:**
- Need compact visualization
- Showing patterns across all models at once
- Space is limited (posters, slides)

**Use Correct vs Incorrect when:**
- Discussing efficiency
- Showing relationship between computation and accuracy
- Identifying wasteful reasoning patterns

### For Analysis:

**Research questions answered by each plot:**

1. **"Which model is most efficient?"**
   → Bar Chart (compare average tokens)

2. **"Are some questions unpredictable?"**
   → Box Plot (look for outliers and wide distributions)

3. **"Do models scale well with difficulty?"**
   → Heatmap (see color progression Easy → Hard)

4. **"Do models use more tokens when wrong?"**
   → Correct vs Incorrect (compare left vs right)

5. **"What's the full token usage pattern?"**
   → Use all 4 plots together!

---

## Technical Details

### Data Source
- **Questions:** 1,019 questions from `hard_question_categories.json`
  - Easy: 832 questions (categories 0, 1, 2)
  - Medium: 120 questions (categories 5, 6, 7)
  - Hard: 67 questions (categories 9, 10, 11)
- **Models:** 11 models from `ITERATIVE_MODEL_ENTRIES`
- **Responses:** `src/responses_reverified/*.jsonl`

### Metrics Used
- **output_tokens:** Full computational cost (includes reasoning for reasoning models)
- **is_correct:** Answer correctness from reverified responses
- **category:** Question difficulty (0-11 models wrong)

### Visualization Parameters
- **Y-axis scale:** Logarithmic (100-25,000 tokens)
- **Color scheme:** 
  - Easy: Green (#2ecc71)
  - Medium: Orange (#f39c12)
  - Hard: Red (#e74c3c)
- **Model order:** Non-reasoning first, then reasoning

---

## Scripts Created

1. **`plot_token_boxplot.py`** - Generates box plot visualization
2. **`plot_token_barchart.py`** - Generates bar chart visualization
3. **`plot_token_heatmap.py`** - Generates heatmap visualization
4. **`plot_token_correct_vs_incorrect.py`** - Generates correct vs incorrect comparison

---

## Commands to Regenerate

```bash
cd /media/torontoai/Iterative-rag/src/analyzing

# Generate all 4 plots
python3 plot_token_boxplot.py
python3 plot_token_barchart.py
python3 plot_token_heatmap.py
python3 plot_token_correct_vs_incorrect.py
```

---

## Summary

✅ **4 complementary visualizations** created  
✅ **Different insights** from each plot type  
✅ **Consistent styling** across all plots  
✅ **Easy to regenerate** with provided scripts  
✅ **Publication-ready** high-resolution outputs (150 DPI)  

These 4 plots provide a comprehensive analysis of token usage patterns across models and difficulty levels, each revealing different aspects of model behavior. Use them together for complete insights or individually based on your specific analysis needs.
