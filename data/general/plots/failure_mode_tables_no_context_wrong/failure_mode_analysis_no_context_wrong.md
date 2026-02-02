# Failure Mode Analysis - No-Context Wrong Questions Only

**Analysis Date**: 2025-11-21 00:06

**Scope**: This analysis includes ONLY questions that were answered incorrectly in no-context mode.
This shows how failure modes specifically affect questions that required context to answer correctly.

## Formulas

- **Prevalence**: p_{m,f} = (# runs with failure f) / (# all no-context-wrong runs for model m) × 100%
- **Impact**: Δ_{m,f} = Accuracy_m(¬f) - Accuracy_m(f) [in percentage points]
- **Damage Index**: d_{m,f} = p_{m,f} × Δ_{m,f}

## Tables

### Prevalence (%)

| Model | Coverage Gap | Composition Failure | Overconfident | Distractor Latch |
| ---: | ---: | ---: | ---: | ---: |
| Claude 3.7 Sonnet | 9.2 | 35.7 | 3.2 | 20.4 |
| Grok 4 Fast | 12.7 | 34.5 | 17.9 | 16.7 |
| Gemini 2.5 Pro | 13.9 | 32.3 | 15.5 | 19.1 |
| Mistral Large | 15.6 | 42.1 | 12.6 | 25.8 |
| GPT-5 | 29.2 | 27.4 | 22.9 | 17.2 |
| Llama 3.3 70B | 27.4 | 44.2 | 15.8 | 24.2 |
| Claude 3.7 + Reasoning | 10.7 | 33.4 | 4.5 | 19.1 |
| GLM 4.6 | 16.4 | 29.6 | 14.7 | 16.4 |
| DeepSeek R1 | 18.4 | 33.9 | 27.3 | 21.6 |
| Claude Sonnet 4.5 | 13.6 | 29.6 | 11.5 | 15.0 |
| GPT-4o | 14.0 | 35.0 | 6.5 | 22.1 |

### Impact (pp drop)

| Model | Coverage Gap | Composition Failure | Overconfident | Distractor Latch |
| ---: | ---: | ---: | ---: | ---: |
| Claude 3.7 Sonnet | 38.9 | 51.2 | 21.5 | 53.5 |
| Grok 4 Fast | 12.2 | 53.6 | 21.3 | 43.6 |
| Gemini 2.5 Pro | 30.8 | 62.6 | 9.0 | 60.8 |
| Mistral Large | 31.2 | 64.8 | 23.8 | 47.9 |
| GPT-5 | 15.8 | 78.9 | 15.6 | 58.6 |
| Llama 3.3 70B | 21.9 | 65.3 | 24.0 | 53.8 |
| Claude 3.7 + Reasoning | 28.7 | 46.2 | -1.2 | 46.1 |
| GLM 4.6 | 31.0 | 69.2 | 23.1 | 51.8 |
| DeepSeek R1 | 27.7 | 61.5 | 8.2 | 55.0 |
| Claude Sonnet 4.5 | 14.7 | 48.8 | 11.1 | 54.4 |
| GPT-4o | 24.4 | 57.8 | 13.7 | 52.2 |

### Damage Index (expected loss)

| Model | Coverage Gap | Composition Failure | Overconfident | Distractor Latch |
| ---: | ---: | ---: | ---: | ---: |
| Claude 3.7 Sonnet | 3.57 | 18.28 | 0.70 | 10.91 |
| Grok 4 Fast | 1.55 | 18.48 | 3.82 | 7.26 |
| Gemini 2.5 Pro | 4.29 | 20.20 | 1.39 | 11.61 |
| Mistral Large | 4.86 | 27.31 | 3.00 | 12.36 |
| GPT-5 | 4.62 | 21.60 | 3.58 | 10.11 |
| Llama 3.3 70B | 6.01 | 28.85 | 3.79 | 13.04 |
| Claude 3.7 + Reasoning | 3.06 | 15.42 | -0.05 | 8.79 |
| GLM 4.6 | 5.07 | 20.51 | 3.39 | 8.49 |
| DeepSeek R1 | 5.11 | 20.87 | 2.25 | 11.88 |
| Claude Sonnet 4.5 | 2.00 | 14.47 | 1.27 | 8.18 |
| GPT-4o | 3.40 | 20.24 | 0.89 | 11.53 |

## Summary Statistics

### Average Prevalence
- **Coverage Gap**: 16.5%
- **Composition Failure**: 34.3%
- **Overconfident**: 13.9%
- **Distractor Latch**: 19.8%

### Average Impact
- **Coverage Gap**: 25.2 pp
- **Composition Failure**: 60.0 pp
- **Overconfident**: 15.5 pp
- **Distractor Latch**: 52.5 pp

### Average Damage Index
- **Coverage Gap**: 3.96 pp
- **Composition Failure**: 20.57 pp
- **Overconfident**: 2.18 pp
- **Distractor Latch**: 10.38 pp
