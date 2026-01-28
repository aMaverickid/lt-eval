#!/usr/bin/env bash
set -euo pipefail

############################################
# =============== 基本配置 =================
############################################

MODEL_PATH="/data/PLMs/Qwen2.5-7B-Instruct"
RESULT_DIR="/data2/wanzhenjie/CODE/lm-evaluation-harness/results"

# 要跑的任务
TASKS=(
  mmlu_ltthinker
  bbh_ltthinker
  gpqa_diamond_generative_n_shot
  gsm8k
)

# 允许使用的 GPU（自行改）
GPU_LIST=(0 1 2 3 4 5 6 7)

# 判定 GPU 空闲的显存阈值（MiB）
# vLLM 启动后一般 > 2000MiB
FREE_MEM_THRESHOLD=1000

# 轮询间隔（秒）
SLEEP_INTERVAL=20

############################################
# ============ 工具函数 =====================
############################################

# 判断某张 GPU 是否空闲
is_gpu_free() {
  local gpu_id="$1"
  local used_mem

  used_mem=$(nvidia-smi \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits \
    -i "$gpu_id")

  if [ "$used_mem" -lt "$FREE_MEM_THRESHOLD" ]; then
    return 0
  else
    return 1
  fi
}

# 找一张空闲 GPU，找到就 echo GPU_ID
# 找不到返回 1
find_free_gpu() {
  for gpu in "${GPU_LIST[@]}"; do
    if is_gpu_free "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

############################################
# =============== 主逻辑 ====================
############################################

echo "=========================================="
echo " Model: $MODEL_PATH"
echo " Tasks: ${TASKS[*]}"
echo " GPUs : ${GPU_LIST[*]}"
echo "=========================================="

for task in "${TASKS[@]}"; do
  echo ""
  echo ">>> Waiting for free GPU for task: $task"

  while true; do
    if GPU_ID=$(find_free_gpu); then
      echo ">>> Launching task [$task] on GPU $GPU_ID"

      CUDA_VISIBLE_DEVICES="$GPU_ID" lm_eval \
        --model vllm \
        --model_args pretrained="$MODEL_PATH",tensor_parallel_size=1,dtype=auto \
        --tasks "$task" \
        --gen_kwargs temperature=0.7,max_tokens=16384,repetition_penalty=1.05,top_p=0.95,do_sample=True \
        --batch_size auto \
        --apply_chat_template \
        -o "$RESULT_DIR" \
        --log_samples \
        --num_fewshot 5 \
        &

      # 给 vLLM 一点时间占显存，避免误判
      sleep 10
      break
    else
      echo ">>> No free GPU, sleeping ${SLEEP_INTERVAL}s..."
      sleep "$SLEEP_INTERVAL"
    fi
  done
done

echo ""
echo ">>> All tasks have been dispatched."
echo ">>> Waiting for remaining jobs to finish..."
wait
echo ">>> Done."
