#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
#      -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=/code/EUV/verl-agent/trial_data/alfworld/train.parquet \
#     data.val_files=/code/EUV/verl-agent/trial_data/alfworld/test.parquet \
#     data.multiturn.enable=true \
#     data.multiturn.messages_key=messages \
#     data.micro_batch_size=4 \
#     data.max_length=4096 \
#     data.truncation=left \
#     model.partial_pretrain=/code/models/Llama-3.2-3B-Instruct \
#     trainer.default_local_dir=$save_path \
#     trainer.project_name=multiturn-sft \
#     trainer.experiment_name=multiturn-sft-qwen-2.5-0.5b-instruct-sp2 \
#     trainer.logger=['console','wandb'] \
#     trainer.total_epochs=10 \
#     trainer.default_hdfs_dir=null $@ \
#     ulysses_sequence_parallel_size=2 \
#     use_remove_padding=true



torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/code/EUV/other_env/baselines/reflact/train.parquet \
    data.val_files=/code/EUV/other_env/baselines/reflact/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=4 \
    data.max_length=4096 \
    data.truncation=left \
    model.partial_pretrain=/code/models/Qwen2.5-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=our-method-sft \
    trainer.experiment_name=reflact-sft \
    trainer.logger=['console'] \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    model.enable_gradient_checkpointing=true