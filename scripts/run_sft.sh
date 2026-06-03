#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${EUV_ROOT:-/code/EUV}"

ENV_NAME="alfworld"
MODEL="qwen1.5b"
CONDA_ENV="auto"
SCRIPT=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_sft.sh [options] [-- extra args]

Options:
  --env alfworld|scienceworld|virtualhome
  --model qwen1.5b|llama3b
  --script /code/EUV/scripts/sft/file.sh
  --conda-env auto|none|ENV_NAME
  --dry-run
  -h, --help

Examples:
  scripts/run_sft.sh --env alfworld --model qwen1.5b
  scripts/run_sft.sh --env alfworld --model llama3b --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$ENV_NAME" in
  alfworld) AUTO_CONDA_ENV="verl-agent-alfworld" ;;
  scienceworld|sciworld) ENV_NAME="scienceworld"; AUTO_CONDA_ENV="verl-agent-sciworld" ;;
  virtualhome|vh) ENV_NAME="virtualhome"; AUTO_CONDA_ENV="verl-agent-vh" ;;
  *) echo "Unsupported env: $ENV_NAME" >&2; exit 2 ;;
esac

if [[ -z "$SCRIPT" ]]; then
  case "$ENV_NAME:$MODEL" in
    alfworld:qwen1.5b|alfworld:qwen1_5b|alfworld:qwen) SCRIPT="$ROOT_DIR/scripts/sft/alfworld_qwen1.5b.sh" ;;
    alfworld:llama3b|alfworld:llama) SCRIPT="$ROOT_DIR/scripts/sft/alfworld_llama3b.sh" ;;
    *) echo "No mapped SFT script for env=$ENV_NAME model=$MODEL. Use --script." >&2; exit 2 ;;
  esac
fi

if [[ "$CONDA_ENV" == "auto" ]]; then
  CONDA_ENV="$AUTO_CONDA_ENV"
fi

CMD=(bash "$SCRIPT" "${EXTRA_ARGS[@]}")
echo "[EUV] env=$ENV_NAME model=$MODEL"
echo "[EUV] script=$SCRIPT"
echo "[EUV] conda_env=$CONDA_ENV"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$CONDA_ENV" == "none" ]]; then
  exec "${CMD[@]}"
fi
exec conda run -n "$CONDA_ENV" "${CMD[@]}"
