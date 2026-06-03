# Experiment Entry Points

This project keeps the original low-level experiment scripts, but exposes three wrapper scripts for easier reproduction.

The intended method pipeline is:

```text
SFT checkpoint -> RL training -> prompt-agent inference/evaluation
```

Use separate conda environments for different embodied environments:

```bash
conda env create -f envs/verl-agent-alfworld.yml
conda env create -f envs/verl-agent-sciworld.yml
conda env create -f envs/verl-agent-vh.yml
```

The wrapper scripts use `--conda-env auto` by default, mapping ALFWorld to `verl-agent-alfworld`, ScienceWorld to `verl-agent-sciworld`, and VirtualHome to `verl-agent-vh`.

## SFT

SFT scripts live under:

```text
/code/EUV/scripts/sft/
```

Use:

```bash
cd /code/EUV
bash scripts/run_sft.sh --env alfworld --model qwen1.5b
bash scripts/run_sft.sh --env alfworld --model llama3b
```

Place the resulting SFT checkpoint under `/code/EUV/checkpoints/`, then use that checkpoint as the starting model for RL.

Only the original ALFWorld SFT launch scripts are currently included. For ScienceWorld or VirtualHome SFT, add the corresponding shell script under `scripts/sft/` and pass it with `--script`.

## RL

RL scripts live under:

```text
/code/EUV/verl-agent/examples/
```

The codebase was developed from the GIGPO-style verl-agent implementation. The ALFWorld GIGPO entrypoint is:

```text
/code/EUV/verl-agent/examples/gigpo_trainer/run_alfworld.sh
```

Use:

```bash
cd /code/EUV
bash scripts/run_rl.sh --env alfworld --algo gigpo -- \
  actor_rollout_ref.model.path=/code/EUV/checkpoints/alfworld/qwen_sft \
  trainer.default_local_dir=/code/EUV/checkpoints/alfworld/gigpo

bash scripts/run_rl.sh --env scienceworld --algo grpo -- \
  actor_rollout_ref.model.path=/code/EUV/checkpoints/scienceworld/sft \
  trainer.default_local_dir=/code/EUV/checkpoints/scienceworld/grpo

bash scripts/run_rl.sh --env virtualhome --algo ppo -- \
  actor_rollout_ref.model.path=/code/EUV/checkpoints/virtualhome/sft \
  trainer.default_local_dir=/code/EUV/checkpoints/virtualhome/ppo
```

Extra Hydra overrides can be passed after `--`:

```bash
bash scripts/run_rl.sh --env alfworld --algo gigpo -- trainer.total_epochs=10 trainer.logger=['console']
```

### Changing RL Context

RL context templates live in:

```text
/code/EUV/verl-agent/agent_system/environments/prompts/alfworld.py
/code/EUV/verl-agent/agent_system/environments/prompts/scienceworld.py
/code/EUV/verl-agent/agent_system/environments/prompts/virtualhome.py
```

Runtime context assembly lives in:

```text
/code/EUV/verl-agent/agent_system/environments/env_manager.py
```

Edit `env_manager.py` if you want to change how action history, previous observations, current observation, and last-turn information are inserted into the prompt. The relevant methods are the `build_text_obs(...)` methods in `AlfWorldEnvironmentManager`, `ScienceWorldEnvironmentManager`, and `VirtualHomeEnvironmentManager`.

Switch prompt variants at launch time with Hydra overrides:

```bash
bash scripts/run_rl.sh --env alfworld --algo gigpo -- \
  env.alfworld.baseline=our_method \
  env.history_length=10

bash scripts/run_rl.sh --env scienceworld --algo grpo -- \
  env.scienceworld.baseline=prompting_variant_only_belief \
  env.history_length=10

bash scripts/run_rl.sh --env virtualhome --algo ppo -- \
  env.virtualhome.baseline=prompting_our_method \
  env.history_length=10
```

## Inference

The main paper inference scripts live under:

```text
/code/EUV/verl-agent/examples/prompt_agent/
```

Use:

```bash
cd /code/EUV
export API_KEY=your_api_key_here
bash scripts/run_inference.sh --env alfworld --method baseline --split seen
bash scripts/run_inference.sh --env alfworld --method ours --split unseen
bash scripts/run_inference.sh --env scienceworld --method baseline --split seen
bash scripts/run_inference.sh --env virtualhome --method ours --split seen
```

For full control, call the low-level scripts directly, for example:

```bash
cd /code/EUV/verl-agent
bash examples/prompt_agent/alfworld/react_seen.sh
bash examples/prompt_agent/sciworld/react_our_unseen.sh
bash examples/prompt_agent/virtualhome/react_seen.sh
```
