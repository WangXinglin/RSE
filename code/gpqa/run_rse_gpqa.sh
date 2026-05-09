#!/bin/bash

# =======================================================
# 1. CLI argument parsing
# =======================================================
if [ $# -lt 3 ]; then
    echo "Error: Insufficient arguments"
    echo "Usage: bash run.sh <start_index> <end_index> <group>"
    echo "Example: bash run.sh 0 5 0"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2
GROUP=$3

echo "======================================================="
echo "Starting inference task"
echo "Data range: $START_INDEX to $END_INDEX"
echo "Group: $GROUP"
echo "======================================================="

pip install sentence-transformers

# =======================================================
# 2. Inference parameter config
# =======================================================
BATCH_SIZE=2000
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
N_COMPLETIONS=128
MAX_TOKENS=32768
N_REFLECTION_COMPLETIONS=32

# =======================================================
# 3. Path & environment config
# =======================================================
MODEL_NAME="<MODEL_PATH>"
QUESTION_FILE="<DATA_DIR>/gpqa.jsonl"

# Output directories
STEP_1_ANSWER_DIR="<OUTPUT_DIR>/step1_answer"
STEP_2_REFLECTION_DIR="<OUTPUT_DIR>/step2-reflection"
STEP_3_ANSWER_DIR="<OUTPUT_DIR>/step3-answer-reference"
STEP_4_REFLECTION_DIR="<OUTPUT_DIR>/step4-reflection"
STEP_5_ANSWER_DIR="<OUTPUT_DIR>/step5-answer-reference"
STEP_6_REFLECTION_DIR="<OUTPUT_DIR>/step6-reflection"
STEP_7_ANSWER_DIR="<OUTPUT_DIR>/step7-answer-reference"

mkdir -p $STEP_1_ANSWER_DIR
mkdir -p $STEP_2_REFLECTION_DIR
mkdir -p $STEP_3_ANSWER_DIR
mkdir -p $STEP_4_REFLECTION_DIR
mkdir -p $STEP_5_ANSWER_DIR
mkdir -p $STEP_6_REFLECTION_DIR
mkdir -p $STEP_7_ANSWER_DIR


# =======================================================
# 4. Execute Python scripts
# =======================================================

# mkdir -p "${STEP_1_ANSWER_DIR}/random_seed_42_group_${GROUP}"
# python <SCRIPT_DIR>/step1_answer.py \
#     --model $MODEL_NAME \
#     --input $QUESTION_FILE \
#     --output "${STEP_1_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
#     --n-completions $N_REFLECTION_COMPLETIONS \
#     --tensor-parallel-size 8 \
#     --batch-size $BATCH_SIZE \
#     --temperature $TEMPERATURE \
#     --top-p $TOP_P \
#     --top-k $TOP_K \
#     --max-tokens $MAX_TOKENS \
#     --start-idx $START_INDEX \
#     --end-idx $END_INDEX

mkdir -p "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}"
python <SCRIPT_DIR>/step2_reflect_gpqa.py \
    --model $MODEL_NAME \
    --question-file $QUESTION_FILE \
    --answer-dir "${STEP_1_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
    --output-dir "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}" \
    --tensor-parallel-size 8 \
    --batch-size 2000 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --n-samples 1 \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

mkdir -p "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup"
python <SCRIPT_DIR>/step2dot2_reflection_dedup_by_emb_gpqa.py \
    --reflection-dir "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}" \
    --output-dir "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup" \
    --debug-dir  "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup_debug" \
    --model-path <EMBEDDING_MODEL_PATH>

mkdir -p "${STEP_3_ANSWER_DIR}/random_seed_42_group_${GROUP}"
python <SCRIPT_DIR>/step3_answer_with_reflection_v4_gpqa.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --reflection-dir "${STEP_2_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup" \
    --output "${STEP_3_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
    --n-reflection-completions $N_REFLECTION_COMPLETIONS \
    --n-completions $N_REFLECTION_COMPLETIONS \
    --tensor-parallel-size 8 \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

# ---------- Iteration 2 ----------

mkdir -p "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}"
python <SCRIPT_DIR>/step2_reflect_gpqa.py \
    --model $MODEL_NAME \
    --question-file $QUESTION_FILE \
    --answer-dir "${STEP_3_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
    --output-dir "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}" \
    --tensor-parallel-size 8 \
    --batch-size 2000 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --n-samples 1 \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

mkdir -p "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup"
python <SCRIPT_DIR>/step2dot2_reflection_dedup_by_emb_gpqa.py \
    --reflection-dir "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}" \
    --output-dir "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup" \
    --debug-dir  "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup_debug" \
    --model-path <EMBEDDING_MODEL_PATH>

mkdir -p "${STEP_5_ANSWER_DIR}/random_seed_42_group_${GROUP}"
python <SCRIPT_DIR>/step3_answer_with_reflection_v4_gpqa.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --reflection-dir "${STEP_4_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup" \
    --output "${STEP_5_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
    --n-reflection-completions $N_REFLECTION_COMPLETIONS \
    --n-completions $N_REFLECTION_COMPLETIONS \
    --tensor-parallel-size 8 \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

# ---------- Iteration 3 ----------

mkdir -p "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}"
python <SCRIPT_DIR>/step2_reflect_gpqa.py \
    --model $MODEL_NAME \
    --question-file $QUESTION_FILE \
    --answer-dir "${STEP_5_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
    --output-dir "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}" \
    --tensor-parallel-size 8 \
    --batch-size 2000 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --n-samples 1 \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

mkdir -p "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup"
python <SCRIPT_DIR>/step2dot2_reflection_dedup_by_emb_gpqa.py \
    --reflection-dir "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}" \
    --output-dir "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup" \
    --debug-dir  "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup_debug" \
    --model-path <EMBEDDING_MODEL_PATH>

mkdir -p "${STEP_7_ANSWER_DIR}/random_seed_42_group_${GROUP}"
python <SCRIPT_DIR>/step3_answer_with_reflection_v4_gpqa.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --reflection-dir "${STEP_6_REFLECTION_DIR}/random_seed_42_group_${GROUP}_dedup" \
    --output "${STEP_7_ANSWER_DIR}/random_seed_42_group_${GROUP}" \
    --n-reflection-completions $N_REFLECTION_COMPLETIONS \
    --n-completions $N_REFLECTION_COMPLETIONS \
    --tensor-parallel-size 8 \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX
