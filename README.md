# 👀 Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents

**🏆 ACL 2026** · **🤖 Embodied Agents** · **🔁 SFT → RL → Inference** · **🌍 ALFWorld / ScienceWorld / VirtualHome**

This repository is the official implementation of **Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents**, accepted to **ACL 2026**. The code supports the paper's SFT -> RL -> inference pipeline across ALFWorld, ScienceWorld, and VirtualHome.

This release packages the useful implementation from the original internal workspace into a reproducible layout for training, evaluation, and data preparation. The original experimental workspace contained checkpoints, logs, W&B runs, core dumps, and temporary outputs. Those files are intentionally excluded here. Put downloaded or newly trained checkpoints under `checkpoints/` when reproducing experiments.

## 📁 Repository Layout

```text
/code/EUV
|-- data/                    # Curated SFT data and task split/index files
|   |-- alfworld/
|   |-- scienceworld/
|   `-- virtualhome/
|-- docs/                    # Reproduction and experiment notes
|-- envs/                    # Conda environment exports
|-- eval/                    # Standalone evaluation and inference helpers
|-- other_env/               # Lightweight external environment adapters/assets
|-- scripts/                 # SFT scripts and unified experiment wrappers
`-- verl-agent/              # Modified verl-agent training framework and env code
```

Key implementation areas:

- `verl-agent/agent_system/environments/`: embodied-environment wrappers, prompts, projections, and multi-turn interaction glue.
- `verl-agent/examples/grpo_trainer/`: GRPO launch scripts for ALFWorld, ScienceWorld, and VirtualHome.
- `verl-agent/examples/ppo_trainer/`: PPO launch scripts.
- `verl-agent/examples/sft/`: supervised fine-tuning examples.
- `verl-agent/examples/prompt_agent/`: closed-source/API model evaluation scripts.
- `eval/`: local/vLLM/closed-source evaluation helpers migrated from the original workspace.

## 🛠️ Setup

We recommend using separate conda environments for the three embodied environments:

```bash
conda env create -f envs/verl-agent-alfworld.yml
conda env create -f envs/verl-agent-sciworld.yml
conda env create -f envs/verl-agent-vh.yml
```

Environment snapshots and key package versions are documented in `envs/README.md`. Use the environment that matches the benchmark you are running:

```bash
conda activate verl-agent-alfworld   # ALFWorld
conda activate verl-agent-sciworld   # ScienceWorld
conda activate verl-agent-vh         # VirtualHome
```

After creating an environment, install the modified verl-agent package in editable mode:

```bash
conda activate verl-agent-alfworld
cd /code/EUV/verl-agent
pip install -e .
```

If you are using a freshly created environment and editable install misses optional runtime packages, install the top-level requirements as well:

```bash
cd /code/EUV
pip install -r requirements.txt
```

For closed-source/API model inference, set credentials at runtime instead of editing scripts:

```bash
export API_KEY=your_api_key_here
export BASE_URL=https://api.openai.com/v1
```

## 📦 Data

Curated data copied into this release:

- `data/alfworld/new_final_alfworld_sft_data.json`
- `data/scienceworld/new_final_scienceworld_sft_data.json`
- `data/scienceworld/*indices*.json`
- `data/virtualhome/*.jsonl`

Large checkpoints and generated trajectories are not included. Recommended local paths:

```text
/code/EUV/checkpoints/        # model checkpoints
/code/EUV/outputs/            # generated logs, trajectories, summaries
```

## 🚀 Reproduction Pipeline

The intended training pipeline is:

```text
SFT checkpoint -> RL training -> prompt-agent inference/evaluation
```

The release keeps the original low-level scripts and adds flexible wrappers under `scripts/`. See `docs/EXPERIMENTS.md` for more examples.

### 1. 🎯 SFT

SFT scripts live under `/code/EUV/scripts/sft`. The wrapper selects the recommended conda environment automatically:

```bash
cd /code/EUV
bash scripts/run_sft.sh --env alfworld --model qwen1.5b
bash scripts/run_sft.sh --env alfworld --model llama3b
```

Place the produced SFT checkpoint under `/code/EUV/checkpoints/`, for example:

```text
/code/EUV/checkpoints/alfworld/qwen_sft/
```

### 2. 🧠 RL

RL scripts live under `/code/EUV/verl-agent/examples`. This codebase is developed from the GIGPO-style verl-agent implementation; the main GIGPO ALFWorld entrypoint is `verl-agent/examples/gigpo_trainer/run_alfworld.sh`.

Use the SFT checkpoint as the initial policy by overriding `actor_rollout_ref.model.path`:

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

#### 🧩 Changing the RL Context

The RL prompt/context is controlled in two places:

```text
/code/EUV/verl-agent/agent_system/environments/prompts/
/code/EUV/verl-agent/agent_system/environments/env_manager.py
```

Use the prompt files when you want to edit the text templates shown to the model:

```text
prompts/alfworld.py       # ALFWorld context templates
prompts/scienceworld.py   # ScienceWorld context templates and variants
prompts/virtualhome.py    # VirtualHome context templates
```

Use `env_manager.py` when you want to change how the runtime context is assembled, for example how `action_history`, previous observations, the current observation, and `last_turn_information` are inserted into the template.

The main methods are:

```text
AlfWorldEnvironmentManager.build_text_obs(...)
VirtualHomeEnvironmentManager.build_text_obs(...)
ScienceWorldEnvironmentManager.build_text_obs(...)
```

At launch time, select a context/prompt variant with Hydra overrides. Common values are:

```text
env.alfworld.baseline=baseline|our_method|prompting_baseline|prompting_our_method
env.scienceworld.baseline=baseline|our_method|variant_only_belief|variant_repeat_o|prompting_baseline|prompting_our_method|prompting_variant_only_belief|prompting_variant_repeat_o|prompting_vagen|prompting_reflact
env.virtualhome.baseline=baseline|our_method|prompting_baseline|prompting_our_method
```

Examples:

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

### 3. 🔍 Inference

Paper inference scripts live under `/code/EUV/verl-agent/examples/prompt_agent`. Use the trained checkpoint or closed-source API model setting expected by the selected low-level script:

```bash
cd /code/EUV
export API_KEY=your_api_key_here
bash scripts/run_inference.sh --env alfworld --method baseline --split seen
bash scripts/run_inference.sh --env scienceworld --method ours --split unseen
bash scripts/run_inference.sh --env virtualhome --method ours --split seen
```

All wrappers accept `--conda-env auto|none|ENV_NAME`; `auto` maps ALFWorld, ScienceWorld, and VirtualHome to `verl-agent-alfworld`, `verl-agent-sciworld`, and `verl-agent-vh` respectively.

## 🧹 Release Notes

This cleaned release intentionally excludes:

- model checkpoints and Hugging Face weight folders
- `wandb/`, `outputs/`, `logs/`, `results/`, and temporary run artifacts
- `core.*` crash dumps
- wheel files and Python caches
- real API keys or private credentials

## 🙏 Acknowledgements

We thank the GIGPO / verl-agent project for their excellent work on embodied-agent RL training infrastructure, and we also thank the verl library for providing the underlying RLHF training framework that this implementation builds on.

## 📚 Citation

```bibtex
@article{wang2026seeing,
  title={Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents},
  author={Wang, Hanlin and Leong, Chak Tou and Wang, Jian and Li, Wenjie},
  journal={arXiv preprint arXiv:2604.17252},
  year={2026}
}
```
