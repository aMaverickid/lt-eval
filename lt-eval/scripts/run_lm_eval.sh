#!/usr/bin/env bash
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
RESULT_DIR="/data2/wanzhenjie/CODE/lm-evaluation-harness/results"

# ================= 运行 =================
CUDA_VISIBLE_DEVICES="${GPU_ID}" lm_eval \
  --model vllm \
  --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,dtype=auto \
  --tasks "${TASK_NAME}" \
  --gen_kwargs temperature=0.7,max_tokens=16384,repetition_penalty=1.05,top_p=0.95,do_samples=True \
  --batch_size auto \
  --apply_chat_template \
  -o "${RESULT_DIR}" \
  --log_samples \
  --num_fewshot 5 


# mmlu_ltthinker, bbh_ltthinker, gpqa_diamond_generative_n_shot,gsm8k