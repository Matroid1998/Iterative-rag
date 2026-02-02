# Model Performance Tables: All Questions

Data from: `all_models_correctness_by_steps_all_questions_no_coverage.png`

These tables show:
- **Accuracy Slope**: Change in accuracy from step 1 to step 5 (or max step)
- **Oracle Hop Distribution**: Number of questions at each step, broken down by the number of hops in the gold reasoning path

Note: This includes **ALL questions** (no filtering).

# Summary: Accuracy Slopes Across All Models

Sorted by slope (steepest improvement first):

| Rank | Model | Slope (pp/step) | Step 1 Acc | Step 5 Acc | Notes |
|------|-------|-----------------|------------|------------|-------|
| 1 | Llama 3.3 70B | +0.87 | 64.7% | 68.2% | Step 1→5 |
| 2 | GLM 4.6 | -3.28 | 83.6% | 70.5% | Step 1→5 |
| 3 | GPT-4o | -3.88 | 95.5% | 80.0% | Step 1→5 |
| 4 | Claude 3.7 Sonnet | -4.38 | 96.3% | 78.8% | Step 1→5 |
| 5 | Claude Sonnet 4.5 | -4.40 | 91.4% | 73.8% | Step 1→5 |
| 6 | Mistral Large | -4.42 | 87.8% | 70.1% | Step 1→5 |
| 7 | Claude 3.7 + Reasoning | -4.79 | 96.4% | 77.3% | Step 1→5 |
| 8 | Gemini 2.5 Pro | -5.59 | 91.8% | 69.4% | Step 1→5 |
| 9 | DeepSeek R1 | -6.19 | 89.3% | 64.5% | Step 1→5 |
| 10 | Grok 4 Fast | -6.20 | 87.8% | 62.9% | Step 1→5 |
| 11 | GPT-5 | -6.89 | 85.8% | 58.2% | Step 1→5 |

---

# Detailed Tables by Model

## Claude 3.7 + Reasoning
**Accuracy Slope (Step 1 to 5)**: -4.79 pp/step (from 96.4% to 77.3%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 96.4% | 140 | 101 | 16 | 12 | 11 |
| 2 | 93.2% | 263 | 91 | 94 | 44 | 34 |
| 3 | 90.9% | 165 | 31 | 46 | 50 | 38 |
| 4 | 86.8% | 151 | 15 | 43 | 51 | 42 |
| 5 | 77.3% | 466 | 60 | 98 | 137 | 171 |

## Claude 3.7 Sonnet
**Accuracy Slope (Step 1 to 5)**: -4.38 pp/step (from 96.3% to 78.8%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 96.3% | 109 | 82 | 17 | 6 | 4 |
| 2 | 89.9% | 268 | 124 | 85 | 37 | 22 |
| 3 | 86.6% | 232 | 41 | 78 | 72 | 41 |
| 4 | 79.0% | 124 | 10 | 32 | 43 | 39 |
| 5 | 78.8% | 453 | 41 | 86 | 136 | 190 |

## Claude Sonnet 4.5
**Accuracy Slope (Step 1 to 5)**: -4.40 pp/step (from 91.4% to 73.8%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 91.4% | 417 | 192 | 98 | 68 | 59 |
| 2 | 92.3% | 310 | 57 | 99 | 80 | 74 |
| 3 | 89.7% | 156 | 11 | 36 | 53 | 56 |
| 4 | 85.9% | 78 | 9 | 22 | 25 | 22 |
| 5 | 73.8% | 225 | 29 | 43 | 68 | 85 |

## DeepSeek R1
**Accuracy Slope (Step 1 to 5)**: -6.19 pp/step (from 89.3% to 64.5%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 89.3% | 410 | 185 | 96 | 64 | 65 |
| 2 | 83.7% | 406 | 79 | 119 | 105 | 103 |
| 3 | 81.2% | 144 | 11 | 36 | 53 | 44 |
| 4 | 72.7% | 88 | 7 | 20 | 32 | 29 |
| 5 | 64.5% | 138 | 16 | 27 | 40 | 55 |

## GLM 4.6
**Accuracy Slope (Step 1 to 5)**: -3.28 pp/step (from 83.6% to 70.5%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 83.6% | 318 | 154 | 65 | 52 | 47 |
| 2 | 88.4% | 216 | 60 | 75 | 46 | 35 |
| 3 | 84.5% | 155 | 26 | 47 | 41 | 41 |
| 4 | 82.1% | 123 | 12 | 28 | 48 | 35 |
| 5 | 70.5% | 346 | 38 | 76 | 103 | 129 |

## GPT-4o
**Accuracy Slope (Step 1 to 5)**: -3.88 pp/step (from 95.5% to 80.0%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 95.5% | 134 | 82 | 19 | 14 | 19 |
| 2 | 86.6% | 82 | 33 | 20 | 19 | 10 |
| 3 | 83.6% | 122 | 40 | 31 | 32 | 19 |
| 4 | 77.0% | 243 | 45 | 60 | 78 | 60 |
| 5 | 80.0% | 605 | 98 | 168 | 151 | 188 |

## GPT-5
**Accuracy Slope (Step 1 to 5)**: -6.89 pp/step (from 85.8% to 58.2%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 85.8% | 759 | 223 | 179 | 183 | 174 |
| 2 | 81.0% | 195 | 34 | 69 | 49 | 43 |
| 3 | 81.7% | 60 | 8 | 14 | 18 | 20 |
| 4 | 60.0% | 50 | 8 | 13 | 14 | 15 |
| 5 | 58.2% | 122 | 25 | 23 | 30 | 44 |

## Gemini 2.5 Pro
**Accuracy Slope (Step 1 to 5)**: -5.59 pp/step (from 91.8% to 69.4%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 91.8% | 353 | 184 | 77 | 50 | 42 |
| 2 | 89.3% | 328 | 59 | 109 | 94 | 66 |
| 3 | 82.8% | 169 | 17 | 43 | 52 | 57 |
| 4 | 79.3% | 116 | 11 | 25 | 31 | 49 |
| 5 | 69.4% | 219 | 26 | 44 | 67 | 82 |

## Grok 4 Fast
**Accuracy Slope (Step 1 to 5)**: -6.20 pp/step (from 87.8% to 62.9%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 87.8% | 449 | 197 | 109 | 70 | 73 |
| 2 | 81.2% | 260 | 43 | 86 | 79 | 52 |
| 3 | 81.0% | 100 | 12 | 26 | 30 | 32 |
| 4 | 73.2% | 82 | 15 | 18 | 28 | 21 |
| 5 | 62.9% | 278 | 29 | 58 | 83 | 108 |

## Llama 3.3 70B
**Accuracy Slope (Step 1 to 5)**: +0.87 pp/step (from 64.7% to 68.2%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 64.7% | 51 | 22 | 8 | 12 | 9 |
| 2 | 75.4% | 195 | 80 | 38 | 41 | 36 |
| 3 | 71.6% | 225 | 66 | 52 | 46 | 61 |
| 4 | 70.5% | 281 | 55 | 81 | 73 | 72 |
| 5 | 68.2% | 434 | 75 | 119 | 122 | 118 |

## Mistral Large
**Accuracy Slope (Step 1 to 5)**: -4.42 pp/step (from 87.8% to 70.1%)

| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |
|------|----------|----------|-------|-------|-------|-------|
| 1 | 87.8% | 139 | 88 | 13 | 19 | 19 |
| 2 | 80.3% | 274 | 104 | 82 | 48 | 40 |
| 3 | 75.2% | 165 | 28 | 54 | 38 | 45 |
| 4 | 70.5% | 190 | 27 | 53 | 55 | 55 |
| 5 | 70.1% | 418 | 51 | 96 | 134 | 137 |
