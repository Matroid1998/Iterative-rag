#!/bin/bash
# Run quality judgement for all 11 models in responses_reverified
# Output will be saved to src/rag_analysis/quality_output/

cd /media/torontoai/Iterative-rag

# Model 1: Mistral Large
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_bedrock_mistral.mistral-large-2402-v1:0_quality_judgement.jsonl

# Model 2: Claude 3.7 Sonnet with Reasoning
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_quality_judgement.jsonl

# Model 3: Claude 3.7 Sonnet
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_quality_judgement.jsonl

# Model 4: DeepSeek R1 with Reasoning
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_bedrock_us.deepseek.r1-v1:0-reasoning_quality_judgement.jsonl

# Model 5: Llama 3.3 70B
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_quality_judgement.jsonl

# Model 6: GPT-4o
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_openai_gpt-4o_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_openai_gpt-4o_quality_judgement.jsonl

# Model 7: GPT-5
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_openai_gpt-5_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_openai_gpt-5_quality_judgement.jsonl

# Model 8: Claude Sonnet 4.5 Reasoning (OpenRouter)
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_openrouter_anthropic_claude_sonnet_4_5_reasoning.jsonl \
  --output src/rag_analysis/quality_output/responses_openrouter_anthropic_claude_sonnet_4_5_reasoning_quality_judgement.jsonl

# Model 9: Gemini 2.5 Pro
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_openrouter_google__gemini-2.5-pro_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_openrouter_google__gemini-2.5-pro_quality_judgement.jsonl

# Model 10: Grok 4 Fast
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_openrouter_x-ai__grok-4-fast_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_openrouter_x-ai__grok-4-fast_quality_judgement.jsonl

# Model 11: GLM 4.6
python src/rag_analysis/quality_judgement.py \
  --jsonl src/responses_reverified/responses_openrouter_z-ai__glm-4.6_reverified.jsonl \
  --output src/rag_analysis/quality_output/responses_openrouter_z-ai__glm-4.6_quality_judgement.jsonl

echo "All quality judgements completed!"
