# Model Performance Tables: No-Context Wrong Questions

Data from: `all_models_correctness_by_steps_no_context_wrong_no_coverage.png`

These tables show:
- **Accuracy Slope**: Change in accuracy from step 1 to step 5 (or max step)
- **Oracle Hop Distribution**: Number of questions at each step, broken down by the number of hops in the gold reasoning path

Note: Questions are filtered to only include those that were answered **incorrectly** in the no-context baseline.

# Summary: Accuracy Slopes Across All Models

Sorted by slope (steepest improvement first):

| Rank | Model | Slope (pp/step) | Step 1 Acc | Step 5 Acc | Notes |
|------|-------|-----------------|------------|------------|-------|
| 1 | Llama 3.3 70B | +0.47 | 63.3% | 65.2% | Step 1→5 |
| 2 | GLM 4.6 | -3.18 | 82.2% | 69.5% | Step 1→5 |
| 3 | GPT-4o | -4.23 | 95.7% | 78.8% | Step 1→5 |
| 4 | Claude 3.7 Sonnet | -4.53 | 95.9% | 77.8% | Step 1→5 |
| 5 | Claude Sonnet 4.5 | -4.62 | 90.8% | 72.3% | Step 1→5 |
| 6 | Mistral Large | -4.76 | 87.6% | 68.5% | Step 1→5 |
| 7 | Claude 3.7 + Reasoning | -4.97 | 96.2% | 76.3% | Step 1→5 |
| 8 | Gemini 2.5 Pro | -5.71 | 91.1% | 68.2% | Step 1→5 |
| 9 | DeepSeek R1 | -6.10 | 88.3% | 63.9% | Step 1→5 |
| 10 | Grok 4 Fast | -6.14 | 86.7% | 62.1% | Step 1→5 |
| 11 | GPT-5 | -6.95 | 84.3% | 56.5% | Step 1→5 |

---

# Detailed Tables by Model

## Claude 3.7 + Reasoning
**Accuracy Slope (Step 1 to 5)**: -4.97 pp/step (from 96.2% to 76.3%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 96.2% | 130 | 93 | 16 | 10 | 11 |
| 2 | 92.1% | 229 | 72 | 85 | 38 | 34 |
| 3 | 89.7% | 145 | 26 | 41 | 44 | 34 |
| 4 | 85.8% | 134 | 13 | 36 | 46 | 39 |
| 5 | 76.3% | 447 | 55 | 93 | 134 | 165 |

## Claude 3.7 Sonnet
**Accuracy Slope (Step 1 to 5)**: -4.53 pp/step (from 95.9% to 77.8%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 95.9% | 98 | 74 | 16 | 4 | 4 |
| 2 | 88.5% | 234 | 105 | 75 | 32 | 22 |
| 3 | 85.4% | 213 | 32 | 74 | 70 | 37 |
| 4 | 77.0% | 113 | 9 | 27 | 40 | 37 |
| 5 | 77.8% | 428 | 39 | 80 | 126 | 183 |

## Claude Sonnet 4.5
**Accuracy Slope (Step 1 to 5)**: -4.62 pp/step (from 90.8% to 72.3%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 90.8% | 390 | 178 | 93 | 62 | 57 |
| 2 | 91.4% | 278 | 41 | 89 | 77 | 71 |
| 3 | 88.4% | 138 | 7 | 30 | 49 | 52 |
| 4 | 83.6% | 67 | 7 | 18 | 20 | 22 |
| 5 | 72.3% | 213 | 26 | 42 | 64 | 81 |

## DeepSeek R1
**Accuracy Slope (Step 1 to 5)**: -6.10 pp/step (from 88.3% to 63.9%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 88.3% | 377 | 170 | 88 | 55 | 64 |
| 2 | 81.5% | 356 | 57 | 105 | 97 | 97 |
| 3 | 80.3% | 137 | 10 | 34 | 50 | 43 |
| 4 | 72.3% | 83 | 6 | 19 | 31 | 27 |
| 5 | 63.9% | 133 | 16 | 26 | 39 | 52 |

## GLM 4.6
**Accuracy Slope (Step 1 to 5)**: -3.18 pp/step (from 82.2% to 69.5%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 82.2% | 292 | 140 | 60 | 47 | 45 |
| 2 | 86.4% | 184 | 44 | 68 | 40 | 32 |
| 3 | 83.6% | 140 | 22 | 41 | 40 | 37 |
| 4 | 80.5% | 113 | 11 | 24 | 44 | 34 |
| 5 | 69.5% | 331 | 35 | 72 | 98 | 126 |

## GPT-4o
**Accuracy Slope (Step 1 to 5)**: -4.23 pp/step (from 95.7% to 78.8%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 95.7% | 117 | 71 | 16 | 11 | 19 |
| 2 | 83.3% | 66 | 21 | 20 | 17 | 8 |
| 3 | 81.5% | 108 | 34 | 28 | 30 | 16 |
| 4 | 75.5% | 229 | 43 | 54 | 75 | 57 |
| 5 | 78.8% | 566 | 90 | 154 | 139 | 183 |

## GPT-5
**Accuracy Slope (Step 1 to 5)**: -6.95 pp/step (from 84.3% to 56.5%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 84.3% | 688 | 192 | 163 | 168 | 165 |
| 2 | 79.6% | 181 | 30 | 64 | 44 | 43 |
| 3 | 80.4% | 56 | 7 | 12 | 18 | 19 |
| 4 | 56.5% | 46 | 7 | 11 | 14 | 14 |
| 5 | 56.5% | 115 | 23 | 22 | 28 | 42 |

## Gemini 2.5 Pro
**Accuracy Slope (Step 1 to 5)**: -5.71 pp/step (from 91.1% to 68.2%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 91.1% | 326 | 167 | 74 | 44 | 41 |
| 2 | 87.9% | 290 | 43 | 96 | 88 | 63 |
| 3 | 81.0% | 153 | 15 | 36 | 47 | 55 |
| 4 | 78.1% | 105 | 8 | 23 | 27 | 47 |
| 5 | 68.2% | 211 | 25 | 43 | 66 | 77 |

## Grok 4 Fast
**Accuracy Slope (Step 1 to 5)**: -6.14 pp/step (from 86.7% to 62.1%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 86.7% | 412 | 175 | 100 | 65 | 72 |
| 2 | 78.0% | 223 | 32 | 76 | 68 | 47 |
| 3 | 79.1% | 91 | 9 | 23 | 28 | 31 |
| 4 | 70.3% | 74 | 13 | 16 | 26 | 19 |
| 5 | 62.1% | 269 | 28 | 56 | 81 | 104 |

## Llama 3.3 70B
**Accuracy Slope (Step 1 to 5)**: +0.47 pp/step (from 63.3% to 65.2%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 63.3% | 49 | 22 | 8 | 11 | 8 |
| 2 | 73.1% | 175 | 71 | 35 | 35 | 34 |
| 3 | 69.3% | 205 | 58 | 48 | 43 | 56 |
| 4 | 68.2% | 261 | 48 | 74 | 69 | 70 |
| 5 | 65.2% | 396 | 60 | 107 | 114 | 115 |

## Mistral Large
**Accuracy Slope (Step 1 to 5)**: -4.76 pp/step (from 87.6% to 68.5%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 87.6% | 129 | 82 | 13 | 16 | 18 |
| 2 | 78.0% | 241 | 85 | 74 | 43 | 39 |
| 3 | 71.9% | 146 | 22 | 46 | 36 | 42 |
| 4 | 69.3% | 179 | 22 | 48 | 54 | 55 |
| 5 | 68.5% | 391 | 48 | 91 | 123 | 129 |
