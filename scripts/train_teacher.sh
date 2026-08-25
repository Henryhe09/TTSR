#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${ROOT_DIR}/verl"

MODEL_PATH="${MODEL_PATH:-}"
TEACHER_TRAIN_FILE="${TEACHER_TRAIN_FILE:-}"
TEACHER_VAL_FILE="${TEACHER_VAL_FILE:-${TEACHER_TRAIN_FILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ -z "${MODEL_PATH}" ] || [ -z "${TEACHER_TRAIN_FILE}" ] || [ -z "${OUTPUT_DIR}" ] || [ -z "${STUDENT_EVAL_URL:-}" ]; then
  echo "Usage:"
  echo "  MODEL_PATH=... TEACHER_TRAIN_FILE=... OUTPUT_DIR=... STUDENT_EVAL_URL=... bash scripts/train_teacher.sh"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${VERL_DIR}:${PYTHONPATH:-}"

TRAIN_BATCH_SIZE="${TEACHER_TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N="${TEACHER_ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TEACHER_TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TEACHER_TOTAL_STEPS:-10}"
N_GPUS="${N_GPUS:-1}"
PROJECT_NAME="${PROJECT_NAME:-ttsr_verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-teacher_grpo}"
TTSR_TAU="${TTSR_TAU:-0.75}"
TTSR_LAMBDA="${TTSR_LAMBDA:-1.0}"
STUDENT_EVAL_URL="${STUDENT_EVAL_URL:-}"
STUDENT_EVAL_TIMEOUT_S="${STUDENT_EVAL_TIMEOUT_S:-240}"
TEACHER_REWARD_RESULT_TIMEOUT_S="${TEACHER_REWARD_RESULT_TIMEOUT_S:-300}"

cd "${VERL_DIR}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TEACHER_TRAIN_FILE}" \
  data.val_files="${TEACHER_VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=3072 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.n="${ROLLOUT_N}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  algorithm.use_kl_in_reward=False \
  reward.num_workers=1 \
  reward.reward_manager.source=importlib \
  reward.reward_manager.name=TTSRStudentFrontierTeacherRewardManager \
  reward.reward_manager.module.path=ttsr.teacher_reward \
  +reward.reward_kwargs.group_wait_ms=80 \
  +reward.reward_kwargs.result_timeout_s="${TEACHER_REWARD_RESULT_TIMEOUT_S}" \
  +reward.reward_kwargs.ttsr_tau="${TTSR_TAU}" \
  +reward.reward_kwargs.ttsr_lambda="${TTSR_LAMBDA}" \
  +reward.reward_kwargs.student_eval_url="${STUDENT_EVAL_URL}" \
  +reward.reward_kwargs.student_eval_timeout_s="${STUDENT_EVAL_TIMEOUT_S}" \
  +reward.reward_kwargs.expected_group_size="${ROLLOUT_N}" \
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
  trainer.save_freq=5 \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data" \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation_data" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpts" \
  "$@"
