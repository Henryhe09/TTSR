#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-}"
BASE_TRAIN="${BASE_TRAIN:-}"
BASE_VAL="${BASE_VAL:-}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/ttsr_loop}"
ROUNDS="${ROUNDS:-3}"
REAL_RATIO="${REAL_RATIO:-5.0}"
MAX_VARIANTS="${MAX_VARIANTS:-8}"
TEACHER_MAX_TRAIN_SAMPLES="${TEACHER_MAX_TRAIN_SAMPLES:-0}"

if [ -z "${MODEL_PATH}" ] || [ -z "${BASE_TRAIN}" ] || [ -z "${BASE_VAL}" ]; then
  echo "Usage:"
  echo "  MODEL_PATH=... BASE_TRAIN=... BASE_VAL=... bash scripts/run_ttsr_loop.sh"
  exit 1
fi

pick_latest_actor_model() {
  local ckpt_root="$1"
  local latest_step_dir
  latest_step_dir="$(ls -d "${ckpt_root}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "${latest_step_dir}" ] && [ -d "${latest_step_dir}/actor/huggingface" ]; then
    echo "${latest_step_dir}/actor/huggingface"
    return
  fi
  if [ -n "${latest_step_dir}" ] && [ -d "${latest_step_dir}/actor" ]; then
    echo "${latest_step_dir}/actor"
    return
  fi
  echo ""
}

mkdir -p "${WORK_DIR}"
CURRENT_MODEL="${MODEL_PATH}"
CURRENT_TEACHER_MODEL="${MODEL_PATH}"
PREV_MEMORY=""
ROUND_TRAIN="${BASE_TRAIN}"

for ROUND in $(seq 1 "${ROUNDS}"); do
  ROUND_DIR="${WORK_DIR}/round_${ROUND}"
  mkdir -p "${ROUND_DIR}"

  echo "========== ROUND ${ROUND} =========="
  echo "[1/7] Train solver with verl..."
  MODEL_PATH="${CURRENT_MODEL}" \
  TRAIN_FILE="${ROUND_TRAIN}" \
  VAL_FILE="${BASE_VAL}" \
  OUTPUT_DIR="${ROUND_DIR}/solver" \
  EXPERIMENT_NAME="ttsr_solver_round_${ROUND}" \
  bash "${ROOT_DIR}/scripts/train_solver.sh"

  NEW_SOLVER_MODEL="$(pick_latest_actor_model "${ROUND_DIR}/solver/ckpts")"
  if [ -n "${NEW_SOLVER_MODEL}" ]; then
    CURRENT_MODEL="${NEW_SOLVER_MODEL}"
  fi
  echo "Current solver model: ${CURRENT_MODEL}"

  echo "[2/7] Extract failed trajectories..."
  python -m ttsr.extract_failed \
    --rollout_dir "${ROUND_DIR}/solver/rollout_data" \
    --out "${ROUND_DIR}/failed_instances.jsonl" \
    --max_samples "$((MAX_VARIANTS * 8))"

  FAILED_COUNT="$(python -c "from pathlib import Path; p=Path(r'''${ROUND_DIR}/failed_instances.jsonl'''); print(0 if not p.exists() else sum(1 for x in p.open('r', encoding='utf-8') if x.strip()))")"
  echo "Failed instances: ${FAILED_COUNT}"

  echo "[3/7] Reflect and update weakness memory..."
  REFLECT_ARGS=(
    --model "${CURRENT_MODEL}"
    --failed_instances "${ROUND_DIR}/failed_instances.jsonl"
    --memory_out "${ROUND_DIR}/memory.json"
    --strategy_out "${ROUND_DIR}/strategy_note.txt"
    --reflections_out "${ROUND_DIR}/reflections.jsonl"
    --iteration "${ROUND}"
  )
  if [ -n "${PREV_MEMORY}" ] && [ -f "${PREV_MEMORY}" ]; then
    REFLECT_ARGS+=(--memory_in "${PREV_MEMORY}")
  fi
  python -m ttsr.reflect "${REFLECT_ARGS[@]}"

  if [ "${FAILED_COUNT}" -gt 0 ]; then
    echo "[4/7] Build Teacher train dataset..."
    python -m ttsr.build_teacher_dataset \
      --failed_instances "${ROUND_DIR}/failed_instances.jsonl" \
      --reflections "${ROUND_DIR}/reflections.jsonl" \
      --memory "${ROUND_DIR}/memory.json" \
      --out "${ROUND_DIR}/teacher_train.parquet" \
      --max_samples "${TEACHER_MAX_TRAIN_SAMPLES}"

    echo "[5/7] Train Teacher with GRPO..."
    MODEL_PATH="${CURRENT_TEACHER_MODEL}" \
    TEACHER_TRAIN_FILE="${ROUND_DIR}/teacher_train.parquet" \
    TEACHER_VAL_FILE="${ROUND_DIR}/teacher_train.parquet" \
    OUTPUT_DIR="${ROUND_DIR}/teacher" \
    EXPERIMENT_NAME="ttsr_teacher_round_${ROUND}" \
    bash "${ROOT_DIR}/scripts/train_teacher.sh"

    NEW_TEACHER_MODEL="$(pick_latest_actor_model "${ROUND_DIR}/teacher/ckpts")"
    if [ -n "${NEW_TEACHER_MODEL}" ]; then
      CURRENT_TEACHER_MODEL="${NEW_TEACHER_MODEL}"
    fi
    echo "Current teacher model: ${CURRENT_TEACHER_MODEL}"

    echo "[6/7] Synthesize variants with Teacher model..."
    python -m ttsr.synthesize \
      --model "${CURRENT_TEACHER_MODEL}" \
      --failed_instances "${ROUND_DIR}/failed_instances.jsonl" \
      --memory "${ROUND_DIR}/memory.json" \
      --reflections "${ROUND_DIR}/reflections.jsonl" \
      --out "${ROUND_DIR}/variants.jsonl" \
      --max_variants "${MAX_VARIANTS}"
  else
    echo "[4-6/7] No failed trajectories, skip Teacher GRPO and synthesis."
    : > "${ROUND_DIR}/variants.jsonl"
  fi

  echo "[7/7] Build next round dataset..."
  python -m ttsr.build_round_dataset \
    --base_train "${BASE_TRAIN}" \
    --variants "${ROUND_DIR}/variants.jsonl" \
    --strategy_note "${ROUND_DIR}/strategy_note.txt" \
    --real_ratio "${REAL_RATIO}" \
    --out "${ROUND_DIR}/round_train.parquet"

  PREV_MEMORY="${ROUND_DIR}/memory.json"
  ROUND_TRAIN="${ROUND_DIR}/round_train.parquet"
done

echo "TTSR loop finished. Outputs: ${WORK_DIR}"

