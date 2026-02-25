pip install openpyxl latex2sympy2 word2number

VERIFICATION_DIR=""
MODEL_PATH=""

python calculate_pass_at_k_from_completions.py \
  --verification_dir "${VERIFICATION_DIR}" \
  --k_values 1 \
  --output_file "${VERIFICATION_DIR}/pass_at_k.json" \
  --max_reference 32 \
  --tokenizer_path "${MODEL_PATH}"
