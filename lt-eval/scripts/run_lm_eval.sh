#!/usr/bin/env bash
unset http_proxy
unset https_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
set -euo pipefail

# ================= 参数解析 =================
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <GPU_ID> <MODEL_PATH> <TASK_NAME>"
  echo ""
  echo "Example:"
  echo "  $0 4 /data/PLMs/Qwen2.5-7B-Instruct gsm8k_cot_zeroshot"
  exit 1
fi

GPU_ID="$1"
MODEL_PATH="$2"
TASK_NAME="$3"

# ================= 固定参数 =================

# MODEL_PATH="/disk0/wanzhenjie/LLMs/DeepSeek-R1-Distill-Qwen-7B"
RESULT_DIR="/disk0/wanzhenjie/lt-eval/results_prompt_token_cnt"

# ================= 运行 =================
CUDA_VISIBLE_DEVICES="${GPU_ID}" lm_eval \
  --model vllm \
  --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,dtype=auto \
  --tasks "${TASK_NAME}" \
  --gen_kwargs max_tokens=16384,temperature=0.7,top_p=0.95,repetition_penalty=1.05 \
  --batch_size auto \
  --apply_chat_template \
  -o "${RESULT_DIR}" \
  --log_samples \
  --num_fewshot 3

# CUDA_VISIBLE_DEVICES="${GPU_ID}" lm_eval \
#   --model vllm \
#   --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,dtype=auto \
#   --tasks "${TASK_NAME}" \
#   --gen_kwargs max_tokens=16384 \
#   --batch_size auto \
#   --apply_chat_template \
#   -o "${RESULT_DIR}" \
#   --log_samples \
#   --num_fewshot 3

# mmlu_ltthinker, bbh_ltthinker_fewshot, gpqa_diamond_generative_n_shot,gsm8k
# gsm8k_ltthinker_vanilla, gpqa_diamond_ltthinker_vanilla, mmlu_ltthinker_vanilla, bbh_ltthinker_vanilla