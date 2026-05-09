set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/logging_utils.sh"

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Please set MODEL_PATH to an existing local instruct model path or an explicitly approved HF model id."
  exit 1
fi

train_data_size=${TRAIN_DATA_SIZE:-2}
val_data_size=${VAL_DATA_SIZE:-2}
group_size=${GROUP_SIZE:-2}
gpu_count=${GPU_COUNT:-2}
max_steps=${CODE_SWE_MAX_STEPS:-6}
repo_cache_dir=${CODE_SWE_REPO_CACHE_DIR:-}
workspace_root=${CODE_SWE_WORKSPACE_ROOT:-/tmp/verl_agent_code_swe_train}
train_dataset=${CODE_SWE_TRAIN_DATASET:-R2E-Gym/R2E-Gym-Subset}
val_dataset=${CODE_SWE_VAL_DATASET:-R2E-Gym/R2E-Gym-Lite}
train_split=${CODE_SWE_TRAIN_SPLIT:-train}
val_split=${CODE_SWE_VAL_SPLIT:-dev_10pr_v1}
LOG_FILE="$(make_log_path code_swe "${gpu_count}" gigpo_lora_v100_smoke)"
announce_log_path "$LOG_FILE"

python3 -m examples.data_preprocess.prepare_code_swe \
  --train_data_size "${train_data_size}" \
  --val_data_size "${val_data_size}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  data.train_files="$HOME/data/verl-agent/code_swe/train.parquet" \
  data.val_files="$HOME/data/verl-agent/code_swe/test.parquet" \
  data.train_batch_size="${train_data_size}" \
  data.val_batch_size="${val_data_size}" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.05 \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=1.0 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_norm \
  env.env_name=code_swe \
  env.seed=0 \
  env.max_steps="${max_steps}" \
  env.history_length=2 \
  env.rollout.n="${group_size}" \
  env.resources_per_worker.num_cpus=0.25 \
  env.code_swe.train_dataset_name="${train_dataset}" \
  env.code_swe.val_dataset_name="${val_dataset}" \
  env.code_swe.train_split="${train_split}" \
  env.code_swe.val_split="${val_split}" \
  env.code_swe.max_train_samples="${train_data_size}" \
  env.code_swe.max_val_samples="${val_data_size}" \
  env.code_swe.runtime.workspace_root="${workspace_root}" \
  env.code_swe.runtime.repo_cache_dir="${repo_cache_dir}" \
  env.code_swe.runtime.allow_network_clone=false \
  env.code_swe.runtime.allow_install=false \
  env.code_swe.runtime.command_timeout=30 \
  env.code_swe.runtime.reward_timeout=180 \
  env.code_swe.runtime.max_output_chars=10000 \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name='verl_agent_code_swe' \
  trainer.experiment_name='gigpo_lora_v100_smoke' \
  trainer.n_gpus_per_node="${gpu_count}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False "$@" 2>&1 | tee "$LOG_FILE"
