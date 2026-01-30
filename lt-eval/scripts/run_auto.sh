export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
#!/usr/bin/env bash
set -euo pipefail

#################################################
# ================= 基本配置 ===================
#################################################

MODEL_PATH="/disk0/wanzhenjie/LLMs/DeepSeek-R1-Distill-Qwen-7B"
RESULT_DIR="/disk0/wanzhenjie/lt-eval/results"

# 需要评测的任务队列（顺序执行，多卡并行）
TASKS=(
  mmlu_ltthinker
  bbh_ltthinker
  gsm8k
  gpqa_diamond_generative_n_shot,gsm8k
)

# 允许调度的 GPU
GPU_LIST=(0 1 2 3 4 5 6 7)

# vLLM 运行时最多使用的显存比例
VLLM_GPU_MEMORY_UTILIZATION=0.9

# 调度阈值：至少有 90% 显存是空闲的，才允许启动新任务
GPU_FREE_RATIO=0.9

# 轮询间隔（秒）
SLEEP_INTERVAL=20

#################################################
# ================ 工具函数 ====================
#################################################

# 判断某张 GPU 是否“足够空闲”
# 条件：free_mem / total_mem >= GPU_FREE_RATIO
is_gpu_available() {
  local gpu_id="$1"
  local used total

  IFS=',' read -r used total <<< "$(
    nvidia-smi \
      --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits \
      -i "$gpu_id"
  )"

  # 去掉可能的空格
  used="${used// /}"
  total="${total// /}"

  local free=$((total - used))

  local free_ratio
  free_ratio=$(awk -v f="$free" -v t="$total" 'BEGIN { printf "%.4f", f/t }')

  awk -v r="$free_ratio" -v thresh="$GPU_FREE_RATIO" \
    'BEGIN { exit !(r >= thresh) }'
}


# 从 GPU_LIST 中找到一张满足条件的 GPU
find_available_gpu() {
  for gpu in "${GPU_LIST[@]}"; do
    if is_gpu_available "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

#################################################
# ================= 主逻辑 =====================
#################################################

echo "================================================"
echo " Model path      : $MODEL_PATH"
echo " Result dir      : $RESULT_DIR"
echo " Tasks           : ${TASKS[*]}"
echo " GPUs            : ${GPU_LIST[*]}"
echo " Free ratio >=   : $GPU_FREE_RATIO"
echo " vLLM max usage  : $VLLM_GPU_MEMORY_UTILIZATION"
echo "================================================"

for task in "${TASKS[@]}"; do
  echo ""
  echo ">>> Waiting for available GPU for task: $task"

  while true; do
    if GPU_ID=$(find_available_gpu); then
      echo ">>> Launching [$task] on GPU $GPU_ID"

      CUDA_VISIBLE_DEVICES="$GPU_ID" lm_eval \
        --model vllm \
        --model_args \
          pretrained="$MODEL_PATH",\
tensor_parallel_size=1,\
dtype=auto,\
gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
        --tasks "$task" \
        --gen_kwargs \
          temperature=0.7,\
max_tokens=16384,\
repetition_penalty=1.05,\
top_p=0.95,\
do_sample=True \
        --batch_size auto \
        --apply_chat_template \
        -o "$RESULT_DIR/$task" \
        --log_samples \
        --num_fewshot 3 \
        &

      # 给 vLLM allocator 一点时间占显存，防止调度误判
      sleep 15
      break
    else
      echo ">>> No sufficiently free GPU, sleeping ${SLEEP_INTERVAL}s..."
      sleep "$SLEEP_INTERVAL"
    fi
  done
done

echo ""
echo ">>> All tasks have been dispatched."
echo ">>> Waiting for all jobs to finish..."
wait
echo ">>> All jobs finished successfully."
