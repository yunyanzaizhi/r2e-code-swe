#!/bin/bash
# =============================================================================
# ALFWorld Full History Strategy — lab-server (2x V100S)
# Strategy: Keep full interaction history (history_length=100)
# Model: Qwen2.5-1.5B-Instruct (text mode, AlfredTWEnv)
# Params: V100-optimized (same core params as Sokoban experiments)
# =============================================================================
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate
source "$PROJECT_ROOT/experiments/logging_utils.sh"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$PROJECT_ROOT/agent_system/environments/env_package/sokoban:$PROJECT_ROOT/agent_system/environments/env_package/alfworld:$PROJECT_ROOT/agent_system/environments/env_package/gym_cards:${PYTHONPATH}"
export PATH=/home/caiting/.local/bin:$PATH
# ALFWORLD_DATA defaults to ~/.cache/alfworld (auto-detected by alfworld/info.py)

ENGINE=${1:-vllm}
GPU_COUNT=${GPU_COUNT:-2}
train_data_size=16
val_data_size=32
group_size=2
history_length=100  # Full history: keep all steps
LOG_FILE="$(make_log_path alfworld "$GPU_COUNT" full_history)"
announce_log_path "$LOG_FILE"

# Data preparation (creates placeholder parquet files)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Training — 2 GPU config, V100-optimized
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length=$history_length \
    env.rollout.n=$group_size \
    env.alfworld.eval_dataset='eval_in_distribution' \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name='alfworld_history_exp' \
    trainer.experiment_name='full_history' \
    trainer.n_gpus_per_node=$GPU_COUNT \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=50 \
    trainer.val_before_train=True 2>&1 | tee "$LOG_FILE"
