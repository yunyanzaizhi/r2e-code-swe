set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/logging_utils.sh"

ENGINE=${ENGINE:-vllm}
if [[ $# -gt 0 && "$1" != *=* ]]; then
  ENGINE="$1"
  shift
fi
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

clear_loopback_hf_proxy_env() {
  if [[ "${R2E_CLEAR_LOCAL_PROXY:-true}" != "true" ]]; then
    return 0
  fi

  local cleared=()
  local name value
  for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
    value="${!name-}"
    case "$value" in
      *://127.0.0.1:*|*://localhost:*|127.0.0.1:*|localhost:*)
        unset "$name"
        cleared+=("$name")
        ;;
    esac
  done

  if [[ ${#cleared[@]} -gt 0 ]]; then
    echo "[r2e_code_swe] cleared loopback proxy env for training: ${cleared[*]}"
    echo "[r2e_code_swe] set R2E_CLEAR_LOCAL_PROXY=false only if a working proxy is running on this server."
  fi
}

configure_hf_model_offline_mode() {
  if [[ "${R2E_HF_MODEL_OFFLINE:-true}" == "true" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[r2e_code_swe] HF model loading is offline: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
    echo "[r2e_code_swe] set R2E_HF_MODEL_OFFLINE=false only when the model is not cached locally."
  fi
}

clear_loopback_hf_proxy_env

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
train_data_size=${TRAIN_DATA_SIZE:-20}
val_data_size=${VAL_DATA_SIZE:-3}
group_size=${GROUP_SIZE:-2}
gpu_count=${GPU_COUNT:-2}
max_steps=${R2E_MAX_STEPS:-8}
max_response_length=${MAX_RESPONSE_LENGTH:-384}
rollout_dtype=${ROLLOUT_DTYPE:-float16}
actor_model_dtype=${ACTOR_MODEL_DTYPE:-float16}
actor_mp_param_dtype=${ACTOR_MP_PARAM_DTYPE:-fp16}
actor_mp_reduce_dtype=${ACTOR_MP_REDUCE_DTYPE:-fp32}
actor_mp_buffer_dtype=${ACTOR_MP_BUFFER_DTYPE:-fp32}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-True}
actor_use_torch_compile=${ACTOR_USE_TORCH_COMPILE:-False}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-True}
train_temperature=${TRAIN_TEMPERATURE:-0.3}
train_top_p=${TRAIN_TOP_P:-0.9}
val_temperature=${VAL_TEMPERATURE:-0.2}
val_top_p=${VAL_TOP_P:-0.9}
tensor_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-${gpu_count}}
train_dataset=${R2E_TRAIN_DATASET:-R2E-Gym/R2E-Gym-Lite}
train_split=${R2E_TRAIN_SPLIT:-train}
val_dataset=${R2E_VAL_DATASET:-R2E-Gym/R2E-Gym-Lite}
val_split=${R2E_VAL_SPLIT:-dev_10pr_v1}
trajectory_dir=${R2E_TRAJECTORY_DIR:-experiments/logs/r2e_code_swe/train_trajectories}
patches_dir=${R2E_PATCHES_DIR:-experiments/logs/r2e_code_swe/train_patches}
max_repeated_no_progress_actions=${R2E_MAX_REPEATED_NO_PROGRESS_ACTIONS:-6}
max_repeated_failed_action_blocks=${R2E_MAX_REPEATED_FAILED_ACTION_BLOCKS:-5}
end_on_repeated_no_progress_limit=${R2E_END_ON_REPEATED_NO_PROGRESS_LIMIT:-false}
end_on_repeated_failed_action_limit=${R2E_END_ON_REPEATED_FAILED_ACTION_LIMIT:-false}

if [[ "${train_split}" == "dev_10pr_v1" && "${ALLOW_TRAIN_ON_DEV:-false}" != "true" ]]; then
  echo "Refusing to train on dev_10pr_v1. Use R2E_TRAIN_SPLIT=train or set ALLOW_TRAIN_ON_DEV=true for an intentional debug run."
  exit 1
fi

LOG_FILE="$(make_log_path r2e_code_swe "${gpu_count}" gigpo_lora_v100_small)"
announce_log_path "$LOG_FILE"

python3 -m examples.data_preprocess.prepare_r2e_code_swe \
  --train_data_size "${train_data_size}" \
  --val_data_size "${val_data_size}"

configure_hf_model_offline_mode

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  data.train_files="$HOME/data/verl-agent/r2e_code_swe/train.parquet" \
  data.val_files="$HOME/data/verl-agent/r2e_code_swe/test.parquet" \
  data.train_batch_size="${train_data_size}" \
  data.val_batch_size="${val_data_size}" \
  data.max_prompt_length=8192 \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=${LORA_RANK:-8} \
  actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-16} \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload="${enable_activation_offload}" \
  actor_rollout_ref.actor.use_torch_compile="${actor_use_torch_compile}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype="${actor_model_dtype}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype="${actor_mp_param_dtype}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype="${actor_mp_reduce_dtype}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype="${actor_mp_buffer_dtype}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_parallel_size}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.dtype="${rollout_dtype}" \
  actor_rollout_ref.rollout.temperature="${train_temperature}" \
  actor_rollout_ref.rollout.top_p="${train_top_p}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine="${rollout_free_cache_engine}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.05 \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=1.0 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_norm \
  env.env_name=r2e_code_swe \
  env.seed=0 \
  env.max_steps="${max_steps}" \
  env.history_length=2 \
  env.rollout.n="${group_size}" \
  env.resources_per_worker.num_cpus=0.25 \
  env.r2e_code_swe.train_dataset_name="${train_dataset}" \
  env.r2e_code_swe.val_dataset_name="${val_dataset}" \
  env.r2e_code_swe.train_split="${train_split}" \
  env.r2e_code_swe.val_split="${val_split}" \
  env.r2e_code_swe.max_train_samples="${train_data_size}" \
  env.r2e_code_swe.max_val_samples="${val_data_size}" \
  env.r2e_code_swe.allow_train_on_dev="${ALLOW_TRAIN_ON_DEV:-false}" \
  env.r2e_code_swe.max_repeated_no_progress_actions="${max_repeated_no_progress_actions}" \
  env.r2e_code_swe.max_repeated_failed_action_blocks="${max_repeated_failed_action_blocks}" \
  env.r2e_code_swe.end_on_repeated_no_progress_limit="${end_on_repeated_no_progress_limit}" \
  env.r2e_code_swe.end_on_repeated_failed_action_limit="${end_on_repeated_failed_action_limit}" \
  env.r2e_code_swe.runtime.command_timeout=60 \
  env.r2e_code_swe.runtime.reward_timeout=300 \
  env.r2e_code_swe.runtime.max_output_chars=12000 \
  env.r2e_code_swe.runtime.trajectory_dir="${trajectory_dir}" \
  env.r2e_code_swe.runtime.patches_dir="${patches_dir}" \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name='verl_agent_r2e_code_swe' \
  trainer.experiment_name='gigpo_lora_v100_small' \
  trainer.n_gpus_per_node="${gpu_count}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs=${TOTAL_EPOCHS:-1} \
  trainer.val_before_train=False "$@" 2>&1 | tee "$LOG_FILE"
