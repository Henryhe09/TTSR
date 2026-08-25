#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-}"
TEST_FILE="${TEST_FILE:-${BASE_TRAIN:-}}"
VAL_FILE="${VAL_FILE:-${BASE_VAL:-${TEST_FILE}}}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/ttsr_loop}"
ROUNDS="${ROUNDS:-3}"
MAX_VARIANTS="${MAX_VARIANTS:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
STUDENT_EVAL_URL="${STUDENT_EVAL_URL:-}"
STUDENT_EVAL_CUDA_VISIBLE_DEVICES="${STUDENT_EVAL_CUDA_VISIBLE_DEVICES:-}"
STUDENT_EVAL_PORT="${STUDENT_EVAL_PORT:-8877}"
STUDENT_EVAL_TIMEOUT_S="${STUDENT_EVAL_TIMEOUT_S:-240}"
STUDENT_EVAL_GPU_MEMORY_UTILIZATION="${STUDENT_EVAL_GPU_MEMORY_UTILIZATION:-0.8}"
STUDENT_EVAL_TENSOR_PARALLEL_SIZE="${STUDENT_EVAL_TENSOR_PARALLEL_SIZE:-1}"

if [ -n "${TRAIN_CUDA_VISIBLE_DEVICES}" ]; then export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"; fi

if [ -z "${MODEL_PATH}" ] || [ -z "${TEST_FILE}" ]; then
  echo "Usage: MODEL_PATH=... TEST_FILE=... [VAL_FILE=...] bash scripts/run_ttsr_loop.sh"
  exit 1
fi

pick_latest_actor_model() {
  local ckpt_root="$1"
  local latest_step_dir
  latest_step_dir="$(ls -d "${ckpt_root}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "${latest_step_dir}" ] && [ -d "${latest_step_dir}/actor/huggingface" ]; then
    echo "${latest_step_dir}/actor/huggingface"
  elif [ -n "${latest_step_dir}" ] && [ -d "${latest_step_dir}/actor" ]; then
    echo "${latest_step_dir}/actor"
  fi
}

EVALUATOR_PID=""
EVALUATOR_URL="${STUDENT_EVAL_URL}"
stop_student_evaluator() {
  if [ -n "${EVALUATOR_PID}" ]; then
    kill "${EVALUATOR_PID}" 2>/dev/null || true
    wait "${EVALUATOR_PID}" 2>/dev/null || true
    EVALUATOR_PID=""
  fi
}
trap stop_student_evaluator EXIT

start_student_evaluator() {
  local model="$1"
  local strategy_note="$2"
  local log_file="$3"
  if [ -n "${EVALUATOR_URL}" ]; then
    return
  fi
  if [ -z "${STUDENT_EVAL_CUDA_VISIBLE_DEVICES}" ]; then
    echo "Set STUDENT_EVAL_URL for a managed evaluator, or reserve an evaluator GPU with STUDENT_EVAL_CUDA_VISIBLE_DEVICES."
    exit 1
  fi
  EVALUATOR_URL="http://127.0.0.1:${STUDENT_EVAL_PORT}"
  local strategy_args=()
  if [ -n "${strategy_note}" ] && [ -f "${strategy_note}" ]; then
    strategy_args=(--strategy_note "${strategy_note}")
  fi
  CUDA_VISIBLE_DEVICES="${STUDENT_EVAL_CUDA_VISIBLE_DEVICES}" \
    python -m ttsr.student_evaluator \
      --model "${model}" \
      --host 127.0.0.1 \
      --port "${STUDENT_EVAL_PORT}" \
      --rollout_n "${ROLLOUT_N}" \
      --gpu_memory_utilization "${STUDENT_EVAL_GPU_MEMORY_UTILIZATION}" \
      --tensor_parallel_size "${STUDENT_EVAL_TENSOR_PARALLEL_SIZE}" \
      "${strategy_args[@]}" > "${log_file}" 2>&1 &
  EVALUATOR_PID=$!
  for _ in $(seq 1 90); do
    if python -c "from urllib.request import urlopen; urlopen('${EVALUATOR_URL}/health', timeout=2).read()" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  echo "Student evaluator did not become ready; see ${log_file}"
  exit 1
}

mkdir -p "${WORK_DIR}"
CURRENT_MODEL="${MODEL_PATH}"
ROUND_TRAIN="${TEST_FILE}"
PREV_MEMORY=""
PREV_STRATEGY=""

for ROUND in $(seq 1 "${ROUNDS}"); do
  ROUND_DIR="${WORK_DIR}/round_${ROUND}"
  mkdir -p "${ROUND_DIR}"
  echo "========== ROUND ${ROUND} =========="

  echo "[1/7] Train Student on D_t = X_test union X_var..."
  MODEL_PATH="${CURRENT_MODEL}" TRAIN_FILE="${ROUND_TRAIN}" VAL_FILE="${VAL_FILE}" \
    OUTPUT_DIR="${ROUND_DIR}/solver" EXPERIMENT_NAME="ttsr_solver_round_${ROUND}" \
    bash "${ROOT_DIR}/scripts/train_solver.sh"
  NEW_SOLVER_MODEL="$(pick_latest_actor_model "${ROUND_DIR}/solver/ckpts")"
  CURRENT_MODEL="${NEW_SOLVER_MODEL:-${CURRENT_MODEL}}"

  echo "[2/7] Extract failed Student trajectories..."
  python -m ttsr.extract_failed --rollout_dir "${ROUND_DIR}/solver/rollout_data" \
    --out "${ROUND_DIR}/failed_instances.jsonl" --max_samples "$((MAX_VARIANTS * 8))"
  FAILED_COUNT="$(python -c "from pathlib import Path; p=Path(r'''${ROUND_DIR}/failed_instances.jsonl'''); print(sum(1 for line in p.open(encoding='utf-8') if line.strip()))")"

  echo "[3/7] Reflect and update weakness memory..."
  REFLECT_ARGS=(--model "${CURRENT_MODEL}" --failed_instances "${ROUND_DIR}/failed_instances.jsonl" \
    --memory_out "${ROUND_DIR}/memory.json" --strategy_out "${ROUND_DIR}/strategy_note.txt" \
    --reflections_out "${ROUND_DIR}/reflections.jsonl" --iteration "${ROUND}")
  if [ -n "${PREV_MEMORY}" ] && [ -f "${PREV_MEMORY}" ]; then REFLECT_ARGS+=(--memory_in "${PREV_MEMORY}"); fi
  python -m ttsr.reflect "${REFLECT_ARGS[@]}"

  if [ "${FAILED_COUNT}" -gt 0 ]; then
    # This frozen Student is the online assessor for both Teacher GRPO and final variants.
    start_student_evaluator "${CURRENT_MODEL}" "${PREV_STRATEGY}" "${ROUND_DIR}/student_evaluator.log"

    echo "[4/7] Build Teacher GRPO prompts..."
    python -m ttsr.build_teacher_dataset --failed_instances "${ROUND_DIR}/failed_instances.jsonl" \
      --reflections "${ROUND_DIR}/reflections.jsonl" --memory "${ROUND_DIR}/memory.json" \
      --out "${ROUND_DIR}/teacher_train.parquet"

    echo "[5/7] Train Teacher from the current Student checkpoint..."
    MODEL_PATH="${CURRENT_MODEL}" TEACHER_TRAIN_FILE="${ROUND_DIR}/teacher_train.parquet" \
      TEACHER_VAL_FILE="${ROUND_DIR}/teacher_train.parquet" OUTPUT_DIR="${ROUND_DIR}/teacher" \
      STUDENT_EVAL_URL="${EVALUATOR_URL}" STUDENT_EVAL_TIMEOUT_S="${STUDENT_EVAL_TIMEOUT_S}" \
      EXPERIMENT_NAME="ttsr_teacher_round_${ROUND}" bash "${ROOT_DIR}/scripts/train_teacher.sh"
    CURRENT_TEACHER_MODEL="$(pick_latest_actor_model "${ROUND_DIR}/teacher/ckpts")"
    CURRENT_TEACHER_MODEL="${CURRENT_TEACHER_MODEL:-${CURRENT_MODEL}}"

    echo "[6/7] Synthesize and pre-evaluate variants with the frozen Student..."
    python -m ttsr.synthesize --model "${CURRENT_TEACHER_MODEL}" \
      --failed_instances "${ROUND_DIR}/failed_instances.jsonl" --memory "${ROUND_DIR}/memory.json" \
      --reflections "${ROUND_DIR}/reflections.jsonl" --out "${ROUND_DIR}/variants.jsonl" \
      --rollout_cache "${ROUND_DIR}/variant_rollout_cache.jsonl" --max_variants "${MAX_VARIANTS}" \
      --student_eval_url "${EVALUATOR_URL}" --student_eval_timeout_s "${STUDENT_EVAL_TIMEOUT_S}"
    stop_student_evaluator
  else
    : > "${ROUND_DIR}/variants.jsonl"
    : > "${ROUND_DIR}/variant_rollout_cache.jsonl"
  fi

  echo "[7/7] Construct next D_t from all original test questions plus variants..."
  python -m ttsr.build_round_dataset --base_train "${TEST_FILE}" --variants "${ROUND_DIR}/variants.jsonl" \
    --strategy_note "${ROUND_DIR}/strategy_note.txt" --out "${ROUND_DIR}/round_train.parquet"
  ROUND_TRAIN="${ROUND_DIR}/round_train.parquet"
  PREV_MEMORY="${ROUND_DIR}/memory.json"
  PREV_STRATEGY="${ROUND_DIR}/strategy_note.txt"
done

echo "TTSR loop finished. Outputs: ${WORK_DIR}"
