# Environment Exports

The original experiments used separate conda environments for the three embodied environments. This is recommended because ALFWorld, ScienceWorld, and VirtualHome have slightly different dependency constraints.

## Recommended Environments

| Embodied environment | Conda environment | Export file |
| --- | --- | --- |
| ALFWorld | `verl-agent-alfworld` | `envs/verl-agent-alfworld.yml` |
| ScienceWorld | `verl-agent-sciworld` | `envs/verl-agent-sciworld.yml` |
| VirtualHome | `verl-agent-vh` | `envs/verl-agent-vh.yml` |

The exact package snapshots exported from the server are included as:

- `envs/verl-agent-alfworld.yml`
- `envs/verl-agent-alfworld.pip-freeze.txt`
- `envs/verl-agent-sciworld.yml`
- `envs/verl-agent-sciworld.pip-freeze.txt`
- `envs/verl-agent-vh.yml`
- `envs/verl-agent-vh.pip-freeze.txt`

## Key Versions

| Environment | Python | Key packages |
| --- | --- | --- |
| `verl-agent-alfworld` | 3.12.0 | `alfworld==0.4.2`, `torch==2.6.0+cu124`, `transformers==4.51.1`, `vllm==0.8.5`, `ray==2.49.2`, `openai==2.8.1` |
| `verl-agent-sciworld` | 3.12.0 | `scienceworld==1.1.3`, `torch==2.6.0+cu124`, `transformers==4.51.1`, `vllm==0.8.5`, `ray==2.49.2`, `openai==2.8.1` |
| `verl-agent-vh` | 3.12.0 | `torch==2.6.0+cu124`, `transformers==4.51.1`, `vllm==0.8.5`, `ray==2.49.2`, `openai==2.8.1`, `numpy==1.26.3` |

## Recreating Environments

```bash
conda env create -f envs/verl-agent-alfworld.yml
conda env create -f envs/verl-agent-sciworld.yml
conda env create -f envs/verl-agent-vh.yml
```

After creating an environment, install this repository in editable mode:

```bash
conda activate verl-agent-alfworld
cd /code/EUV/verl-agent
pip install -e .
```

The wrapper scripts under `scripts/` select the corresponding environment automatically unless `--conda-env` is provided.
