# Failure Mode Analysis Tables

Generated: November 20, 2025

## Definitions

For each model *m* and failure *f* ∈ {Coverage-Gap, Composition-Failure, Overconfident, Distractor-Latch}:

### 1. Prevalence
```
p_{m,f} = (#runs with f) / (#all runs for m)
```
(reported as %)

### 2. Impact (pp drop)
```
Δ_{m,f} = Acc_m(¬f) - Acc_m(f)
```
(average accuracy difference in percentage points)

### 3. Damage Index (expected loss)
```
d_{m,f} = p_{m,f} × Δ_{m,f}
```
Interpretable as: *expected pp of accuracy lost per question due to f for model m*

## Prevalence (%)

| Model | Coverage Gap | Composition Failure | Overconfident | Distractor Latch |
| --- | ---: | ---: | ---: | ---: |
| Claude 3.7 + Reasoning | 10.1 | 24.3 | 4.8 | 14.7 |
| Claude 3.7 Sonnet | 8.3 | 27.1 | 3.3 | 16.2 |
| Claude Sonnet 4.5 | 11.6 | 21.2 | 10.4 | 11.0 |
| DeepSeek R1 | 16.9 | 26.5 | 26.5 | 18.6 |
| GLM 4.6 | 15.8 | 23.5 | 15.5 | 13.5 |
| GPT-4o | 13.2 | 29.3 | 5.8 | 19.1 |
| GPT-5 | 28.9 | 16.5 | 24.6 | 11.1 |
| Gemini 2.5 Pro | 13.0 | 23.6 | 15.6 | 14.8 |
| Grok 4 Fast | 11.9 | 23.9 | 17.5 | 12.8 |
| Llama 3.3 70B | 26.7 | 37.2 | 15.3 | 21.7 |
| Mistral Large | 15.4 | 36.8 | 12.5 | 24.2 |

## Impact (pp drop)

| Model | Coverage Gap | Composition Failure | Overconfident | Distractor Latch |
| --- | ---: | ---: | ---: | ---: |
| Claude 3.7 + Reasoning | 24.5 | 45.5 | 1.3 | 44.2 |
| Claude 3.7 Sonnet | 28.5 | 47.5 | 13.5 | 49.1 |
| Claude Sonnet 4.5 | 12.5 | 47.0 | 8.9 | 52.2 |
| DeepSeek R1 | 25.6 | 56.5 | 6.7 | 51.1 |
| GLM 4.6 | 28.2 | 68.2 | 18.7 | 49.2 |
| GPT-4o | 19.6 | 52.7 | 14.7 | 49.3 |
| GPT-5 | 8.3 | 80.8 | 7.9 | 61.1 |
| Gemini 2.5 Pro | 26.8 | 60.0 | 7.8 | 56.1 |
| Grok 4 Fast | 5.3 | 55.1 | 15.9 | 47.6 |
| Llama 3.3 70B | 19.9 | 65.5 | 22.4 | 53.1 |
| Mistral Large | 26.4 | 62.5 | 22.0 | 48.3 |

## Damage Index (expected loss)

| Model | Coverage Gap | Composition Failure | Overconfident | Distractor Latch |
| --- | ---: | ---: | ---: | ---: |
| Claude 3.7 + Reasoning | 2.48 | 11.04 | 0.06 | 6.49 |
| Claude 3.7 Sonnet | 2.36 | 12.85 | 0.44 | 7.95 |
| Claude Sonnet 4.5 | 1.44 | 9.96 | 0.93 | 5.77 |
| DeepSeek R1 | 4.32 | 14.95 | 1.77 | 9.52 |
| GLM 4.6 | 4.45 | 16.01 | 2.90 | 6.65 |
| GPT-4o | 2.59 | 15.42 | 0.85 | 9.40 |
| GPT-5 | 2.41 | 13.30 | 1.93 | 6.81 |
| Gemini 2.5 Pro | 3.49 | 14.16 | 1.21 | 8.28 |
| Grok 4 Fast | 0.62 | 13.14 | 2.77 | 6.10 |
| Llama 3.3 70B | 5.31 | 24.36 | 3.43 | 11.51 |
| Mistral Large | 4.07 | 22.97 | 2.74 | 11.69 |

## Summary Statistics

### Average Across All Models

| Failure Mode | Avg Prevalence (%) | Avg Impact (pp) | Avg Damage Index |
| --- | ---: | ---: | ---: |
| Coverage Gap | 15.6 | 20.5 | 3.05 |
| Composition Failure | 26.3 | 58.3 | 15.29 |
| Overconfident | 13.8 | 12.7 | 1.73 |
| Distractor Latch | 16.2 | 51.0 | 8.20 |