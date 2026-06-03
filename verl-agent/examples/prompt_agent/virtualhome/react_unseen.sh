#!/usr/bin/env bash

# 不再需要 Ray 相关的环境变量

# ========== 可配置参数 ==========
# API Key - 必填
: "${API_KEY:?Set API_KEY before running}"

# 结果保存文件路径 - 可选，直接指定完整文件路径（如：/path/to/result.txt）
# 如果不设置，将使用默认目录 + 时间戳文件名
export RESULT_FILE="/code/EUV/outputs/results/virtualhome/result_summary_no_thinking_unseen.txt"

# Prompting 方法 - 可选，默认为 baseline
# 可选值: baseline 或 our_method
export PROMPTING_METHOD="prompting_baseline"

# 任务数据路径 - 可选，用于指定测试任务列表（JSON 或 JSONL 格式）
# 如果不设置，将使用环境默认的任务列表
export TASK_DATA_PATH="/code/EUV/data/virtualhome/new_unseen_test_shuffled.jsonl"

# 总测试案例数 - 可选，默认为 134
export TOTAL_TEST_CASES="125"

# 最大步数 - 可选，默认为 40
export MAX_STEPS="40"

# =================================

ENV_NAME="virtualhome"

if [[ "$ENV_NAME" == "virtualhome" ]]; then
  echo "Launching VirtualHome agent (串行模式 - 不使用 Ray)..."
  echo "配置信息:"
  echo "  - API Key: ${API_KEY:0:20}..."
  echo "  - 结果保存文件: $RESULT_FILE"
  echo "  - Prompting 方法: $PROMPTING_METHOD"
  echo "  - 任务数据路径: ${TASK_DATA_PATH:-未设置}"
  echo "  - 总测试案例数: ${TOTAL_TEST_CASES:-134}"
  echo "  - 最大步数: ${MAX_STEPS:-40}"
  python3 -m examples.prompt_agent.virtualhome.gpt4o_virtualhome_sequential
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'virtualhome'." >&2
  exit 1
fi

