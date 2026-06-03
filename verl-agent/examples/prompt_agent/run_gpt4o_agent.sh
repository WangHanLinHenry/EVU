#!/usr/bin/env bash

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

ENV_NAME="alfoworld"

if [[ "$ENV_NAME" == "alfoworld" ]]; then
  echo "Launching AlfWorld agent..."
  python3 -m examples.prompt_agent.gpt4o_alfworld
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfoworld'." >&2
  exit 1
fi
