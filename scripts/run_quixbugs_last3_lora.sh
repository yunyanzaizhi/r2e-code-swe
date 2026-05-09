#!/bin/bash
# =============================================================================
# QuixBugs Recent Window k=3 LoRA
# Strategy: Keep only the most recent 3 interaction steps.
# Base: scripts/run_quixbugs_fullhistory_lora.sh
# Model/LoRA: Qwen2.5-1.5B-Instruct, LoRA rank=8 alpha=16 (same as full history)
# V100-32GB note: ProgrammingEnv is currently singleton, so train_batch_size=1
# is required. We raise vLLM GPU memory utilization conservatively for V100-32GB.
# =============================================================================
set -x
cd /home/caiting/verl-agent-exp
source .venv/bin/activate

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="/home/caiting/verl-agent-exp/agent_system/environments/env_package/programming:${PYTHONPATH}"
export PATH=/home/caiting/.local/bin:$PATH

mkdir -p experiments/logs experiments/results checkpoints/programming_history_exp /tmp/verl-ray
export RAY_TMPDIR="/tmp/verl-ray"

ENGINE=${1:-vllm}
train_data_size=1
val_data_size=1
group_size=1
history_length=3
MODEL_PATH=/home/caiting/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.50}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1536 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    +actor_rollout_ref.model.tokenizer_path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Programming \
    env.seed=0 \
    env.max_steps=10 \
    env.history_length=$history_length \
    env.rollout.n=$group_size \
    +env.programming.data_root=/home/caiting/verl-agent-exp/third_party/QuixBugs \
    +env.programming.memory_type=simple \
    +env.programming.reward_mode=rltf_scalar \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name='programming_history_exp' \
    trainer.experiment_name='quixbugs_recent_window_k3_lora' \
    trainer.default_local_dir=$PWD/checkpoints/programming_history_exp/recent_window_k3 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=999999 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True 2>&1 | tee experiments/logs/quixbugs_recent_window_k3_lora.log
