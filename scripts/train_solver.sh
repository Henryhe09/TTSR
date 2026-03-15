#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${ROOT_DIR}/verl"

MODEL_PATH="${MODEL_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-}"
VAL_FILE="${VAL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ -z "${MODEL_PATH}" ] || [ -z "${TRAIN_FILE}" ] || [ -z "${VAL_FILE}" ] || [ -z "${OUTPUT_DIR}" ]; then
  echo "Usage:"
  echo "  MODEL_PATH=... TRAIN_FILE=... VAL_FILE=... OUTPUT_DIR=... bash scripts/train_solver.sh"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${VERL_DIR}:${PYTHONPATH:-}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-30}"
N_GPUS="${N_GPUS:-1}"
PROJECT_NAME="${PROJECT_NAME:-ttsr_verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-solver_grpo}"

cd "${VERL_DIR}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.n="${ROLLOUT_N}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  algorithm.use_kl_in_reward=False \
  reward.num_workers=1 \
  reward.reward_manager.source=importlib \
  reward.reward_manager.name=TTSRMajorityRewardManager \
  reward.reward_manager.module.path=ttsr.reward_managers \
  +reward.reward_kwargs.group_wait_ms=80 \
  +reward.reward_kwargs.result_timeout_s=5.0 \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.save_freq=10 \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data" \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation_data" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpts" \
  "$@"

