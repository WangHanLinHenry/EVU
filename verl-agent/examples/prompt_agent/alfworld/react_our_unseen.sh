#!/usr/bin/env bash

# 不再需要 Ray 相关的环境变量

# ========== 可配置参数 ==========
# API Key - 必填
: "${API_KEY:?Set API_KEY before running}"

# 结果保存文件路径 - 可选，直接指定完整文件路径（如：/path/to/result.txt）
# 如果不设置，将使用默认目录 + 时间戳文件名
export RESULT_FILE="/code/EUV/outputs/results/alfworld/traj_stat_result_summary_react_our_method_out_of_distribution.txt"

# Prompting 方法 - 可选，默认为 prompting_baseline
# 可选值: prompting_baseline 或 prompting_our_method
export PROMPTING_METHOD="prompting_our_method"

# 评估数据集类型 - 可选，默认为 eval_in_distribution
# 可选值: eval_in_distribution 或 eval_out_of_distribution
export EVAL_DATASET="eval_out_of_distribution"

# 是否保存每个 task 的每轮生成内容到 JSON（1=保存，0=不保存）
export SAVE_TRAJECTORIES="1"
# 轨迹 JSON 保存目录
export TRAJECTORY_DIR="/code/EUV/outputs/results/alfworld/our_method_traj_unseen"

# =================================

ENV_NAME="alfoworld"

if [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld agent (串行模式 - 不使用 Ray)..."
  echo "配置信息:"
  echo "  - API Key: ${API_KEY:0:20}..."
  echo "  - 结果保存文件: $RESULT_FILE"
  echo "  - Prompting 方法: $PROMPTING_METHOD"
  echo "  - 评估数据集: $EVAL_DATASET"
  echo "  - 保存轨迹: $SAVE_TRAJECTORIES, 目录: $TRAJECTORY_DIR"
  python3 -m examples.prompt_agent.alfworld.gpt4o_alfworld_sequential_original
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  exit 1
fi