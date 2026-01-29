#!/bin/bash

# =======================================================
# 1. Command Line Arguments
# =======================================================
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments"
    echo "Usage: bash run.sh <start_index> <end_index>"
    echo "Example: bash run.sh 0 5"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2

echo "======================================================="
echo "Start Inference Task"
echo "Range: $START_INDEX to $END_INDEX"
echo "======================================================="

pip install sentence-transformers

# =======================================================
# 2. Inference Parameters
# =======================================================
BATCH_SIZE=30
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
N_COMPLETIONS=10240
MAX_TOKENS=38912
N_experience_COMPLETIONS=32
THRESHOLD=0.8

# =======================================================
# 3. Paths and Environment
# =======================================================
MODEL_NAME="/path/to/model"
QUESTION_FILE="/path/to/data/questions.jsonl"

# Output Directory Definition
STEP_1_ANSWER_DIR="/path/to/output/step1-answer"
STEP_2_EXPERIENCE_DIR="/path/to/output/step2-experience"
STEP_3_ANSWER_DIR="/path/to/output/step3-answer-reference"
STEP_4_EXPERIENCE_DIR="/path/to/output/step4-experience"
STEP_5_ANSWER_DIR="/path/to/output/step5-answer-reference"
STEP_6_EXPERIENCE_DIR="/path/to/output/step6-experience"
STEP_7_ANSWER_DIR="/path/to/output/step7-answer-reference"

mkdir -p $STEP_1_ANSWER_DIR
mkdir -p $STEP_2_EXPERIENCE_DIR
mkdir -p $STEP_3_ANSWER_DIR
mkdir -p $STEP_4_EXPERIENCE_DIR
mkdir -p $STEP_5_ANSWER_DIR
mkdir -p $STEP_6_EXPERIENCE_DIR
mkdir -p $STEP_7_ANSWER_DIR

# =======================================================
# 4. Execute Python Scripts
# =======================================================

# ---------- Iteration 0 ----------

mkdir -p "${STEP_1_ANSWER_DIR}/results"
python code/standard_sampling.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --output "${STEP_1_ANSWER_DIR}/results" \
    --n-completions "$N_COMPLETIONS" \
    --batch-size 2000 \
    --tensor-parallel-size 8 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

# ---------- Iteration 1 ----------

mkdir -p "${STEP_2_EXPERIENCE_DIR}/results"
python code/experience_distillation.py \
    --model $MODEL_NAME \
    --question-file $QUESTION_FILE \
    --answer-dir "${STEP_1_ANSWER_DIR}/results" \
    --output-dir "${STEP_2_EXPERIENCE_DIR}/results" \
    --tensor-parallel-size 8 \
    --batch-size 2000 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --n-samples 1 \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

mkdir -p "${STEP_2_EXPERIENCE_DIR}/results_dedup"
python code/experience_dedup.py \
    --experience-dir "${STEP_2_EXPERIENCE_DIR}/results" \
    --output-dir "${STEP_2_EXPERIENCE_DIR}/results_dedup" \
    --debug-dir  "${STEP_2_EXPERIENCE_DIR}/results_dedup_debug" \
    --model-path /path/to/embedding/model \
    --threshold $THRESHOLD  \
    --keep-order
    
mkdir -p "${STEP_3_ANSWER_DIR}/results"
python code/experience_guided_search.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --experience-dir "${STEP_2_EXPERIENCE_DIR}/results_dedup" \
    --output "${STEP_3_ANSWER_DIR}/results" \
    --n-experience-completions $N_experience_COMPLETIONS \
    --n-completions 32 \
    --tensor-parallel-size 8 \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

# ---------- Iteration 2 ----------

mkdir -p "${STEP_4_EXPERIENCE_DIR}/results"
python code/experience_distillation.py \
    --model $MODEL_NAME \
    --question-file $QUESTION_FILE \
    --answer-dir "${STEP_3_ANSWER_DIR}/results" \
    --output-dir "${STEP_4_EXPERIENCE_DIR}/results" \
    --tensor-parallel-size 8 \
    --batch-size 2000 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --n-samples 1 \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

mkdir -p "${STEP_4_EXPERIENCE_DIR}/results_dedup"
python code/experience_dedup.py \
    --experience-dir "${STEP_4_EXPERIENCE_DIR}/results" \
    --previous-experience-dir "${STEP_2_EXPERIENCE_DIR}/results_dedup" \
    --output-dir "${STEP_4_EXPERIENCE_DIR}/results_dedup" \
    --debug-dir  "${STEP_4_EXPERIENCE_DIR}/results_dedup_debug" \
    --model-path /path/to/embedding/model \
    --threshold $THRESHOLD  \
    --keep-order

mkdir -p "${STEP_5_ANSWER_DIR}/results"
python code/experience_guided_search.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --experience-dir "${STEP_4_EXPERIENCE_DIR}/results_dedup" \
    --output "${STEP_5_ANSWER_DIR}/results" \
    --n-experience-completions $N_experience_COMPLETIONS \
    --n-completions 32 \
    --tensor-parallel-size 8 \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

# ---------- Iteration 3 ----------

mkdir -p "${STEP_6_EXPERIENCE_DIR}/results"
python code/experience_distillation.py \
    --model $MODEL_NAME \
    --question-file $QUESTION_FILE \
    --answer-dir "${STEP_5_ANSWER_DIR}/results" \
    --output-dir "${STEP_6_EXPERIENCE_DIR}/results" \
    --tensor-parallel-size 8 \
    --batch-size 2000 \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --n-samples 1 \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX

mkdir -p "${STEP_6_EXPERIENCE_DIR}/results_dedup"
python code/experience_dedup.py \
    --experience-dir "${STEP_6_EXPERIENCE_DIR}/results" \
    --previous-experience-dir "${STEP_4_EXPERIENCE_DIR}/results_dedup" \
    --output-dir "${STEP_6_EXPERIENCE_DIR}/results_dedup" \
    --debug-dir  "${STEP_6_EXPERIENCE_DIR}/results_dedup_debug" \
    --model-path /path/to/embedding/model \
    --threshold $THRESHOLD  \
    --keep-order

mkdir -p "${STEP_7_ANSWER_DIR}/results"
python code/experience_guided_search.py \
    --model $MODEL_NAME \
    --input $QUESTION_FILE \
    --experience-dir "${STEP_6_EXPERIENCE_DIR}/results_dedup" \
    --output "${STEP_7_ANSWER_DIR}/results" \
    --n-experience-completions $N_experience_COMPLETIONS \
    --n-completions 32 \
    --tensor-parallel-size 8 \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-tokens $MAX_TOKENS \
    --start-idx $START_INDEX \
    --end-idx $END_INDEX
