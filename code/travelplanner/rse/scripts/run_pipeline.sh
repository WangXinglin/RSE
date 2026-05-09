#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-<MODEL_PATH>}"
TP="${TP:-4}"
DATA_DIR="${DATA_DIR:-$(dirname "$0")/../../database}"
DATABASE_DIR="${DATABASE_DIR:-${DATA_DIR}}"
BASE_OUTPUT="${BASE_OUTPUT:-$(dirname "$0")/../outputs/rse_$(date +%Y%m%d_%H%M%S)}"

TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
N_COMPLETIONS="${N_COMPLETIONS:-1}"
SEED="${SEED:-42}"

N_ITERS="${N_ITERS:-3}"

DEDUP_THRESHOLD="${DEDUP_THRESHOLD:-0.85}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
MAX_REFLECTION_ITEMS="${MAX_REFLECTION_ITEMS:-50}"

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() { echo -e "\n\033[1;36m>>> $1\033[0m"; }

log "RSE Pipeline — TravelPlanner (Multi-iteration, blind, per-question dedup)"
echo "  MODEL:          ${MODEL}"
echo "  TP:             ${TP}"
echo "  DATA_DIR:       ${DATA_DIR}"
echo "  BASE_OUTPUT:    ${BASE_OUTPUT}"
echo "  TEMPERATURE:    ${TEMPERATURE}"
echo "  N_COMPLETIONS:  ${N_COMPLETIONS}"
echo "  N_ITERS:        ${N_ITERS}  (total inference rounds: $((N_ITERS + 1)))"
echo "  SEED:           ${SEED}"
echo "  RANGE:          [${START_IDX}, ${END_IDX})"

mkdir -p "${BASE_OUTPUT}"

ITER0_DIR="${BASE_OUTPUT}/iteration_0"
ITER0_PLANS="${ITER0_DIR}/plans"
ITER0_EVAL="${ITER0_DIR}/eval.json"
ITER0_REFLECT="${ITER0_DIR}/reflections"
ITER0_DEDUP="${ITER0_DIR}/reflections_dedup"

log "Iteration 0: Generating initial plans (vanilla)..."
python "${RSE_DIR}/step1_plan.py" \
    --model "${MODEL}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ITER0_PLANS}" \
    --n-completions "${N_COMPLETIONS}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --tp "${TP}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}" \
    --seed "${SEED}"
log "Iteration 0 inference done."

if [ "${N_ITERS}" -gt 0 ]; then
    log "Iteration 0: Blind self-reflection on ALL queries..."
    python "${RSE_DIR}/step2_self_reflect.py" \
        --model "${MODEL}" \
        --data-dir "${DATA_DIR}" \
        --plan-dir "${ITER0_PLANS}" \
        --output-dir "${ITER0_REFLECT}" \
        --temperature "${TEMPERATURE}" \
        --max-tokens "${MAX_TOKENS}" \
        --tp "${TP}" \
        --seed "${SEED}" \
        --start-idx "${START_IDX}" \
        --end-idx "${END_IDX}"
    log "Iteration 0 self-reflection done."

    log "Iteration 0: Deduplicating reflections (per-question)..."
    python "${RSE_DIR}/step2dot2_reflect_dedup.py" \
        --reflection-dir "${ITER0_REFLECT}" \
        --output-dir "${ITER0_DEDUP}" \
        --model-path "${EMBEDDING_MODEL}" \
        --threshold "${DEDUP_THRESHOLD}"
    log "Iteration 0 dedup done → ${ITER0_DEDUP}"
fi

if [ ! -f "${ITER0_EVAL}" ]; then
    log "Iteration 0: Evaluating (tracking only)..."
    python "${RSE_DIR}/eval_offline.py" \
        --data-dir "${DATA_DIR}" \
        --database-dir "${DATABASE_DIR}" \
        --input-dir "${ITER0_PLANS}" \
        --output "${ITER0_EVAL}"
    log "Iteration 0 eval done."
else
    log "Iteration 0 eval already done, skipping."
fi

for (( iter=1; iter<=N_ITERS; iter++ )); do
    PREV_ITER=$((iter - 1))

    PREV_DEDUP="${BASE_OUTPUT}/iteration_${PREV_ITER}/reflections_dedup"
    ITER_DIR="${BASE_OUTPUT}/iteration_${iter}"
    ITER_PLANS="${ITER_DIR}/plans"
    ITER_EVAL="${ITER_DIR}/eval.json"
    ITER_REFLECT="${ITER_DIR}/reflections"
    ITER_DEDUP="${ITER_DIR}/reflections_dedup"

    log "Iteration ${iter}: Generating plans with per-question reflections (from iter ${PREV_ITER})..."
    python "${RSE_DIR}/step3_plan_with_reflection.py" \
        --model "${MODEL}" \
        --data-dir "${DATA_DIR}" \
        --reflection-dir "${PREV_DEDUP}" \
        --output-dir "${ITER_PLANS}" \
        --n-completions "${N_COMPLETIONS}" \
        --temperature "${TEMPERATURE}" \
        --max-tokens "${MAX_TOKENS}" \
        --tp "${TP}" \
        --start-idx "${START_IDX}" \
        --end-idx "${END_IDX}" \
        --seed "${SEED}" \
        --max-reflection-items "${MAX_REFLECTION_ITEMS}"
    log "Iteration ${iter} inference done."

    if [ "${iter}" -lt "${N_ITERS}" ]; then
        log "Iteration ${iter}: Blind self-reflection on ALL queries..."
        python "${RSE_DIR}/step2_self_reflect.py" \
            --model "${MODEL}" \
            --data-dir "${DATA_DIR}" \
            --plan-dir "${ITER_PLANS}" \
            --output-dir "${ITER_REFLECT}" \
            --temperature "${TEMPERATURE}" \
            --max-tokens "${MAX_TOKENS}" \
            --tp "${TP}" \
            --seed "${SEED}" \
            --start-idx "${START_IDX}" \
            --end-idx "${END_IDX}"
        log "Iteration ${iter} self-reflection done."

        log "Iteration ${iter}: Deduplicating reflections (per-question, merging with iter ${PREV_ITER})..."
        python "${RSE_DIR}/step2dot2_reflect_dedup.py" \
            --reflection-dir "${ITER_REFLECT}" \
            --previous-reflection-dir "${PREV_DEDUP}" \
            --output-dir "${ITER_DEDUP}" \
            --model-path "${EMBEDDING_MODEL}" \
            --threshold "${DEDUP_THRESHOLD}"
        log "Iteration ${iter} dedup done → ${ITER_DEDUP}"
    fi

    if [ ! -f "${ITER_EVAL}" ]; then
        log "Iteration ${iter}: Evaluating (tracking only)..."
        python "${RSE_DIR}/eval_offline.py" \
            --data-dir "${DATA_DIR}" \
            --database-dir "${DATABASE_DIR}" \
            --input-dir "${ITER_PLANS}" \
            --output "${ITER_EVAL}"
        log "Iteration ${iter} eval done."
    else
        log "Iteration ${iter} eval already done, skipping."
    fi
done

log "RSE Pipeline complete! (${N_ITERS} reflect-dedup-infer cycles)"
echo ""
echo "Results by iteration:"
for (( iter=0; iter<=N_ITERS; iter++ )); do
    ITER_EVAL="${BASE_OUTPUT}/iteration_${iter}/eval.json"
    if [ -f "${ITER_EVAL}" ]; then
        echo "  Iteration ${iter}: ${ITER_EVAL}"
    fi
done
echo ""
echo "Compare iteration 0 (vanilla) vs iteration ${N_ITERS} (final) to see the RSE effect."
