#!/usr/bin/env bash
set -euo pipefail

pip install uv

LCB_ROOT="<LCB_ROOT>"
CONVERT_SCRIPT="<SCRIPT_DIR>/convert_livecode_bench.py"
PYTHON_BIN="python"

SKIP_EXISTING_CONVERT=0
SKIP_EXISTING_EVAL=0

export HF_HOME=<CACHE_DIR>/huggingface
export HF_HUB_CACHE=<CACHE_DIR>/huggingface/hub
export HF_DATASETS_CACHE=<CACHE_DIR>/huggingface/datasets

export UV_CACHE_DIR=<CACHE_DIR>/uv-cache
export UV_PROJECT_ENVIRONMENT=<LCB_ROOT>/.venv
export UV_DEFAULT_INDEX=https://pypi.org/simple
export UV_HTTP_TIMEOUT=300

cd "$LCB_ROOT"
source .venv/bin/activate

ROOT_DIR="<OUTPUT_DIR>"

echo "########## Stage 1: Convert all ##########"
for stage_name in step7-answer-reference; do
  for group in {0..7}; do
    WORK_DIR="${ROOT_DIR}/${stage_name}/random_seed_42_group_${group}"
    INPUT_FILE="${WORK_DIR}/lcb_eval_format.json"
    REPORT_FILE="${WORK_DIR}/lcb_eval_format_report.json"
    CONVERT_LOG="${WORK_DIR}/convert.log"

    if [[ ! -d "$WORK_DIR" ]]; then
      echo "[WARN][convert] missing dir: $WORK_DIR"
      continue
    fi

    if [[ "$SKIP_EXISTING_CONVERT" -eq 1 && -f "$INPUT_FILE" ]]; then
      echo "[SKIP][convert] already exists: $INPUT_FILE"
      continue
    fi

    echo "[RUN ][convert] $WORK_DIR"
    "$PYTHON_BIN" "$CONVERT_SCRIPT" \
      --input_dir "$WORK_DIR" \
      --output_file "$INPUT_FILE" \
      --report_file "$REPORT_FILE" \
      >"$CONVERT_LOG" 2>&1

    if [[ -f "$INPUT_FILE" ]]; then
      echo "[DONE][convert] $INPUT_FILE"
    else
      echo "[FAIL][convert] check log: $CONVERT_LOG"
    fi
  done
done

echo "########## Stage 2: Evaluate all ##########"
for stage_name in step7-answer-reference; do
  for group in {0..7}; do
    WORK_DIR="${ROOT_DIR}/${stage_name}/random_seed_42_group_${group}"
    INPUT_FILE="${WORK_DIR}/lcb_eval_format.json"
    EVAL_LOG="${WORK_DIR}/eval.log"

    if [[ ! -d "$WORK_DIR" ]]; then
      echo "[WARN][eval] missing dir: $WORK_DIR"
      continue
    fi

    if [[ ! -f "$INPUT_FILE" ]]; then
      echo "[WARN][eval] missing converted file: $INPUT_FILE"
      continue
    fi

    if [[ "$SKIP_EXISTING_EVAL" -eq 1 && -f "$EVAL_LOG" ]]; then
      echo "[SKIP][eval] already exists: $EVAL_LOG"
      continue
    fi

    echo "[RUN ][eval] $INPUT_FILE"
    "$PYTHON_BIN" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$INPUT_FILE" \
      --timeout 6 \
      --num_process_evaluate 64 \
      --release_version v6 \
      >"$EVAL_LOG" 2>&1

    echo "[DONE][eval] $EVAL_LOG"
  done
done

echo "Finished."
