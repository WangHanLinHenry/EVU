#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${EUV_ROOT:-/code/EUV}"
VERL_DIR="$ROOT_DIR/verl-agent"

ENV_NAME="alfworld"
METHOD="baseline"
SPLIT="seen"
CONDA_ENV="auto"
SCRIPT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_inference.sh [options]

Options:
  --env alfworld|scienceworld|virtualhome
  --method baseline|ours|our_method|reflact|vagen
  --split seen|unseen|half1|half2
  --script examples/prompt_agent/.../file.sh
  --conda-env auto|none|ENV_NAME
  --dry-run
  -h, --help

Examples:
  scripts/run_inference.sh --env alfworld --method ours --split unseen
  scripts/run_inference.sh --env scienceworld --method baseline --split seen
  scripts/run_inference.sh --env virtualhome --method ours --split seen
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$ENV_NAME" in
  alfworld) AUTO_CONDA_ENV="verl-agent-alfworld" ;;
  scienceworld|sciworld) ENV_NAME="scienceworld"; AUTO_CONDA_ENV="verl-agent-sciworld" ;;
  virtualhome|vh) ENV_NAME="virtualhome"; AUTO_CONDA_ENV="verl-agent-vh" ;;
  *) echo "Unsupported env: $ENV_NAME" >&2; exit 2 ;;
esac

if [[ -z "$SCRIPT" ]]; then
  case "$ENV_NAME:$METHOD:$SPLIT" in
    alfworld:baseline:seen) SCRIPT="examples/prompt_agent/alfworld/react_seen.sh" ;;
    alfworld:baseline:unseen) SCRIPT="examples/prompt_agent/alfworld/react_unseen.sh" ;;
    alfworld:ours:seen|alfworld:our_method:seen) SCRIPT="examples/prompt_agent/alfworld/react_our_seen.sh" ;;
    alfworld:ours:unseen|alfworld:our_method:unseen) SCRIPT="examples/prompt_agent/alfworld/react_our_unseen.sh" ;;
    scienceworld:baseline:seen) SCRIPT="examples/prompt_agent/sciworld/react_seen.sh" ;;
    scienceworld:baseline:unseen) SCRIPT="examples/prompt_agent/sciworld/react_unseen.sh" ;;
    scienceworld:ours:seen|scienceworld:our_method:seen) SCRIPT="examples/prompt_agent/sciworld/react_our_seen.sh" ;;
    scienceworld:ours:unseen|scienceworld:our_method:unseen) SCRIPT="examples/prompt_agent/sciworld/react_our_unseen.sh" ;;
    scienceworld:reflact:half1) SCRIPT="examples/prompt_agent/sciworld/reflact.sh" ;;
    scienceworld:reflact:half2) SCRIPT="examples/prompt_agent/sciworld/reflact_dev2.sh" ;;
    scienceworld:vagen:half1) SCRIPT="examples/prompt_agent/sciworld/vagen.sh" ;;
    scienceworld:vagen:half2) SCRIPT="examples/prompt_agent/sciworld/vagen_dev2.sh" ;;
    virtualhome:baseline:seen) SCRIPT="examples/prompt_agent/virtualhome/react_seen.sh" ;;
    virtualhome:baseline:unseen) SCRIPT="examples/prompt_agent/virtualhome/react_unseen.sh" ;;
    virtualhome:ours:seen|virtualhome:our_method:seen) SCRIPT="examples/prompt_agent/virtualhome/react_our_seen.sh" ;;
    virtualhome:ours:unseen|virtualhome:our_method:unseen) SCRIPT="examples/prompt_agent/virtualhome/react_our_unseen.sh" ;;
    *) echo "No mapped inference script for env=$ENV_NAME method=$METHOD split=$SPLIT. Use --script." >&2; exit 2 ;;
  esac
fi

if [[ "$CONDA_ENV" == "auto" ]]; then
  CONDA_ENV="$AUTO_CONDA_ENV"
fi

CMD=(bash "$SCRIPT")
echo "[EUV] env=$ENV_NAME method=$METHOD split=$SPLIT"
echo "[EUV] workdir=$VERL_DIR"
echo "[EUV] script=$SCRIPT"
echo "[EUV] conda_env=$CONDA_ENV"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

cd "$VERL_DIR"
if [[ "$CONDA_ENV" == "none" ]]; then
  exec "${CMD[@]}"
fi
exec conda run -n "$CONDA_ENV" "${CMD[@]}"
