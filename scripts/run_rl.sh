#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${EUV_ROOT:-/code/EUV}"
VERL_DIR="$ROOT_DIR/verl-agent"

ENV_NAME="alfworld"
ALGO="gigpo"
CONDA_ENV="auto"
ENGINE="vllm"
SCRIPT=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_rl.sh [options] [-- extra hydra overrides]

Options:
  --env alfworld|scienceworld|virtualhome
  --algo gigpo|gigpo_lora|grpo|ppo|dapo
  --engine vllm|sglang
  --script examples/.../file.sh
  --conda-env auto|none|ENV_NAME
  --dry-run
  -h, --help

Examples:
  scripts/run_rl.sh --env alfworld --algo gigpo
  scripts/run_rl.sh --env scienceworld --algo grpo -- trainer.total_epochs=10
  scripts/run_rl.sh --env virtualhome --algo ppo --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --algo) ALGO="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
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
  case "$ALGO:$ENV_NAME" in
    gigpo:alfworld) SCRIPT="examples/gigpo_trainer/run_alfworld.sh" ;;
    gigpo_lora:alfworld) SCRIPT="examples/gigpo_trainer/run_alfworld_lora.sh" ;;
    grpo:alfworld) SCRIPT="examples/grpo_trainer/run_alfworld.sh" ;;
    grpo:scienceworld) SCRIPT="examples/grpo_trainer/run_scienceworld.sh" ;;
    grpo:virtualhome) SCRIPT="examples/grpo_trainer/run_virtualhome.sh" ;;
    ppo:alfworld) SCRIPT="examples/ppo_trainer/run_alfworld.sh" ;;
    ppo:scienceworld) SCRIPT="examples/ppo_trainer/run_scienceworld_ppo.sh" ;;
    ppo:virtualhome) SCRIPT="examples/ppo_trainer/run_virtualhome_ppo.sh" ;;
    dapo:alfworld) SCRIPT="examples/dapo_trainer/run_alfworld.sh" ;;
    dapo:scienceworld) SCRIPT="examples/dapo_trainer/run_scienceworld.sh" ;;
    *) echo "No mapped RL script for algo=$ALGO env=$ENV_NAME. Use --script." >&2; exit 2 ;;
  esac
fi

if [[ "$CONDA_ENV" == "auto" ]]; then
  CONDA_ENV="$AUTO_CONDA_ENV"
fi

CMD=(bash "$SCRIPT" "$ENGINE" "${EXTRA_ARGS[@]}")
echo "[EUV] env=$ENV_NAME algo=$ALGO engine=$ENGINE"
echo "[EUV] workdir=$VERL_DIR"
echo "[EUV] script=$SCRIPT"
echo "[EUV] conda_env=$CONDA_ENV"
printf '[EUV] extra_args='
printf '%q ' "${EXTRA_ARGS[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

cd "$VERL_DIR"
if [[ "$CONDA_ENV" == "none" ]]; then
  exec "${CMD[@]}"
fi
exec conda run -n "$CONDA_ENV" "${CMD[@]}"
