#!/usr/bin/env bash

# 不再需要 Ray 相关的环境变量

# ========== 可配置参数 ==========
# API Key - 必填
: "${API_KEY:?Set API_KEY before running}"

# 结果保存文件路径 - 可选，直接指定完整文件路径（如：/path/to/result.txt）
# 如果不设置，将使用默认目录 + 时间戳文件名
export RESULT_FILE="/code/EUV/outputs/results/sciworld/vagen/our_method_test2.txt"

# Prompting 方法 - 可选，默认为 baseline
# 可选值: prompting_baseline 或 prompting_our_method
export PROMPTING_METHOD="prompting_our_method"

# 任务数据路径 - 可选，用于指定测试任务列表（JSON 或 JSONL 格式）
# 如果不设置，将使用环境默认的任务列表
export TASK_DATA_PATH="/code/EUV/data/scienceworld/test_indices_filtered_shuffled_half2.json"

# seen有151(77,74)，而unseen有161（91，70）
export TOTAL_TEST_CASES="70"

# ScienceWorld 服务器路径 - 可选
export SERVER_PATH="/code/EUV/other_env/env/scienceworld/scienceworld.jar"

# 最大步数字典路径 - 可选
export MAX_STEPS_PATH="/code/EUV/other_env/eval_agent/data/sciworld/max_steps.json"

# Task name 到 ID 的映射路径 - 可选，用于 max_steps 查找时的转换
export TASKNAME2ID_PATH="/code/EUV/other_env/eval_agent/data/sciworld/taskname2id.json"

# =================================

ENV_NAME="scienceworld"

if [[ "$ENV_NAME" == "scienceworld" ]]; then
  echo "Launching ScienceWorld agent (串行模式 - 不使用 Ray)..."
  echo "配置信息:"
  echo "  - API Key: ${API_KEY:0:20}..."
  echo "  - 结果保存文件: $RESULT_FILE"
  echo "  - Prompting 方法: $PROMPTING_METHOD"
  echo "  - 任务数据路径: ${TASK_DATA_PATH:-未设置}"
  echo "  - 总测试案例数: ${TOTAL_TEST_CASES:-100}"
  python3 -m examples.prompt_agent.sciworld.gpt4o_scienceworld_sequential
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'scienceworld'." >&2
  exit 1
fi

