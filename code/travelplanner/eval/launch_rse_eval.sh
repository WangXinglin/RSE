#!/usr/bin/env bash
# ==============================================================================
# TravelPlanner RSE Evaluation Launch Script (BLIND — no eval signal)
#
# All hyperparameters are specified here. The command line accepts three
# positional arguments:
#   start_idx  — sample start index (inclusive)
#   end_idx    — sample end index (exclusive)
#   group      — group identifier, only affects output path suffix
#
# Usage:
#   bash launch_rse_eval.sh 0 60 0
#   bash launch_rse_eval.sh 60 120 1
#   bash launch_rse_eval.sh 120 180 2
# ==============================================================================
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash $0 <start_idx> <end_idx> <group>"
    echo "Example: bash $0 0 60 0"
    exit 1
fi

START_IDX=$1
END_IDX=$2
GROUP=$3

pip install sentence-transformers

# ======================== Hyperparameters ========================

MODEL="<MODEL_PATH>"
TP=8
TEMPERATURE=0.6
MAX_TOKENS=38912
N_COMPLETIONS=32
SEED=$GROUP

# Iteration count: reflect -> dedup -> re-infer cycles (total inference rounds = N_ITERS + 1)
N_ITERS=3

# Reflection params (blind self-reflection, ALL queries)
N_REFLECTIONS=1
DEDUP_THRESHOLD=0.85
EMBEDDING_MODEL="<EMBEDDING_MODEL_PATH>"
MAX_REFLECTION_ITEMS=1500

# ======================== Path Config ========================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RSE_DIR="$(cd "${SCRIPT_DIR}/../rse" && pwd)"
DATA_DIR="$(cd "${SCRIPT_DIR}/../database" && pwd)"

# Full database path (containing flights/ accommodations/ etc.)
DATABASE_DIR="${DATA_DIR}"

# Group only affects path suffix
BASE_OUTPUT="${RSE_DIR}/outputs/rse/group_${GROUP}"

# ======================== Print Config ========================

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
echo "  N_REFLECTIONS:  ${N_REFLECTIONS}"
echo "  SEED:           ${SEED}"
echo "  GROUP:          ${GROUP}"
echo "  RANGE:          [${START_IDX}, ${END_IDX})"
echo "============================================"
echo ""

# ======================== Run Inner Script ========================

export MODEL TP DATA_DIR DATABASE_DIR TEMPERATURE MAX_TOKENS N_COMPLETIONS SEED N_ITERS
export N_REFLECTIONS DEDUP_THRESHOLD EMBEDDING_MODEL MAX_REFLECTION_ITEMS
export START_IDX END_IDX
export BASE_OUTPUT

bash "${SCRIPT_DIR}/run_pipeline_eval.sh"
