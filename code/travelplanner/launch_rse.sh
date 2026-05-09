#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash $0 <start_idx> <end_idx> <group>"
    echo "Example: bash $0 0 60 0"
    exit 1
fi

START_IDX=$1
END_IDX=$2
GROUP=$3

MODEL="${MODEL:-<MODEL_PATH>}"
TP=4
TEMPERATURE=0.7
MAX_TOKENS=4096
N_COMPLETIONS=1
SEED=$GROUP

N_ITERS=3

DEDUP_THRESHOLD=0.85
EMBEDDING_MODEL="all-MiniLM-L6-v2"
MAX_REFLECTION_ITEMS=50

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RSE_DIR="$(cd "${SCRIPT_DIR}/rse" && pwd)"
DATA_DIR="$(cd "${SCRIPT_DIR}/database" && pwd)"
DATABASE_DIR="${DATA_DIR}"

BASE_OUTPUT="${RSE_DIR}/outputs/rse/group_${GROUP}"

echo "============================================"
echo " TravelPlanner RSE Pipeline"
echo "  (blind — no eval signal in reflection)"
echo "============================================"
echo "  MODEL:          ${MODEL}"
echo "  TP:             ${TP}"
echo "  DATA_DIR:       ${DATA_DIR}"
echo "  BASE_OUTPUT:    ${BASE_OUTPUT}"
echo "  TEMPERATURE:    ${TEMPERATURE}"
echo "  N_COMPLETIONS:  ${N_COMPLETIONS}"
echo "  N_ITERS:        ${N_ITERS}  (total inference rounds: $((N_ITERS + 1)))"
echo "  SEED:           ${SEED}"
echo "  GROUP:          ${GROUP}"
echo "  RANGE:          [${START_IDX}, ${END_IDX})"
echo "============================================"
echo ""

export MODEL TP DATA_DIR DATABASE_DIR TEMPERATURE MAX_TOKENS N_COMPLETIONS SEED N_ITERS
export DEDUP_THRESHOLD EMBEDDING_MODEL MAX_REFLECTION_ITEMS
export START_IDX END_IDX
export BASE_OUTPUT

bash "${RSE_DIR}/scripts/run_pipeline.sh"
