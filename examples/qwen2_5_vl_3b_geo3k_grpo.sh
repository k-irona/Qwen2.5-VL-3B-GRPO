#!/bin/bash

set -euo pipefail
set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
N_GPUS=${N_GPUS:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_UPDATE=${MICRO_BATCH_SIZE_UPDATE:-1}
MICRO_BATCH_SIZE_EXPERIENCE=${MICRO_BATCH_SIZE_EXPERIENCE:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_PIXELS=${MAX_PIXELS:-524288}
LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
KL_COEF=${KL_COEF:-1.0e-2}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-64}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
VAL_FREQ=${VAL_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
LOGGER=${LOGGER:-"['file','wandb']"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_geo_grpo_3090_bf16}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.max_pixels=${MAX_PIXELS} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=${LORA_RANK} \
    worker.actor.model.lora.alpha=${LORA_ALPHA} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_EXPERIENCE} \
    worker.actor.optim.lr=${LEARNING_RATE} \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    worker.rollout.tensor_parallel_size=${N_GPUS} \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    algorithm.kl_coef=${KL_COEF} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.val_before_train=true \
    trainer.val_generations_to_log=8 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.logger=${LOGGER} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    "$@"
