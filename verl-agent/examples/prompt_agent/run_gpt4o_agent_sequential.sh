#!/usr/bin/env bash

# 不再需要 Ray 相关的环境变量

ENV_NAME="alfoworld"

if [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld agent (串行模式 - 不使用 Ray)..."
  python3 -m examples.prompt_agent.gpt4o_alfworld_sequential
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  exit 1
fi

