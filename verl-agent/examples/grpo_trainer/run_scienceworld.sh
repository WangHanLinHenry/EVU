#!/bin/bash
set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS

export RAY_memory_usage_threshold=0.98  # 提高内存阈值（默认0.9）
export RAY_memory_monitor_refresh_ms=1000  # 调整监控频率

num_cpus_per_env_worker=0.01 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=2
val_data_size=5
group_size=6

# seen有151(77,74)，而unseen有161（91，70）

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/code/EUV/verl-agent/text/train.parquet \
    data.val_files=/code/EUV/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5120 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.checkpoint.contents="['model', 'optimizer', 'hf_model', 'extra']" \
    actor_rollout_ref.model.path=/code/EUV/other_env/baselines/reflact/global_step_12\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=scienceworld \
    env.seed=0 \
    env.max_steps=10 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.scienceworld.task_indices_path=/code/EUV/other_env/train_indices_filtered.json \
    env.scienceworld.val_indices_path=/code/EUV/other_env/test_indices_filtered_shuffled_half2.json \
    env.scienceworld.taskname2id_path=/code/EUV/other_env/eval_agent/data/sciworld/taskname2id.json \
    env.scienceworld.max_steps_path=/code/EUV/other_env/eval_agent/data/sciworld/max_steps.json \
    env.scienceworld.server_path=/code/EUV/other_env/env/scienceworld/scienceworld.jar \
    env.scienceworld.split=train \
    env.scienceworld.simplification_str=easy \
    env.scienceworld.generate_gold_path=False \
    env.scienceworld.baseline=baseline \
    env.scienceworld.instruction_path=/code/EUV/other_env/eval_agent/prompt/instructions/sciworld_inst.txt \
    env.history_length=10 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_scienceworld_reflact' \
    trainer.experiment_name='qwen3_grpo_reflact_train' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=150 \
    trainer.total_epochs=70 \
    trainer.val_before_train=True $@ \
    trainer.val_only=False \
    trainer.default_local_dir=/code/EUV/other_env/baselines/reflact2 \
    env.model_name=qwen

