set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS

export RAY_memory_usage_threshold=0.98  # 提高内存阈值（默认0.9）
export RAY_memory_monitor_refresh_ms=1000  # 调整监控频率

num_cpus_per_env_worker=0.05 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=12 # match GRPO and GiGPO configuration (16 × 8)
val_data_size=125

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=/code/EUV/verl-agent/text/train.parquet \
    data.val_files=/code/EUV/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5120 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.checkpoint.contents="['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.model.path=/code/EUV/checkpoints/virtualhome/qwen3/ppo_baseline/global_step_100/actor/huggingface \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
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
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/code/EUV/checkpoints/virtualhome/qwen3/ppo_baseline/best_checkpoint/actor/huggingface \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=virtualhome \
    env.seed=0 \
    env.max_steps=40 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.virtualhome.task_data_path=/code/STeCa/IPR/data/vh_data/final_data/new_train.jsonl \
    env.virtualhome.val_data_path=/code/EUV/other_env/vh_test_data/new_unseen_test_shuffled.jsonl \
    env.virtualhome.max_steps=40 \
    env.history_length=10 \
    env.model_name=qwen \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_virtualhome' \
    trainer.experiment_name='result_baseline_ppo_qwen3_1.7b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.default_local_dir=/code/EUV/checkpoints/virtualhome/qwen3/ppo_baseline2 \
    trainer.val_before_train=True $@ \
    trainer.val_only=True \
    env.virtualhome.baseline=baseline \
    
