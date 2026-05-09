#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
EVAL_SCRIPT="<SCRIPT_DIR>/eval_gpqa.py"
QUESTION_FILE="<DATA_DIR>/gpqa.jsonl"

ROOT_DIR="<OUTPUT_DIR>/GPQA-diamond"

STAGES=(
  "step3-answer-reference"
  "step5-answer-reference"
  "step7-answer-reference"
)

GROUP_IDS=(0 1 2 3 4 5 6 7)

KS=(1 4 8 16 32)

NUM_TRIALS=5000
SEED=42
EXPECTED_COUNT=198

SKIP_EXISTING_EVAL=1

echo "########## Stage 1: Evaluate all groups ##########"

for STAGE in "${STAGES[@]}"; do
  for GROUP in "${GROUP_IDS[@]}"; do
    WORK_DIR="${ROOT_DIR}/${STAGE}/random_seed_42_group_${GROUP}"

    if [[ ! -d "$WORK_DIR" ]]; then
      echo "[WARN][eval] missing dir: $WORK_DIR"
      continue
    fi

    OUTPUT_JSON="${WORK_DIR}/eval_summary_mc.json"
    OUTPUT_DETAIL="${WORK_DIR}/eval_detail_mc.jsonl"

    if [[ "$SKIP_EXISTING_EVAL" -eq 1 && -f "$OUTPUT_JSON" ]]; then
      echo "[SKIP][eval] already exists: $OUTPUT_JSON"
      continue
    fi

    echo "[RUN ][eval] ${WORK_DIR}"
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
      --question-file "$QUESTION_FILE" \
      --pred-dir "$WORK_DIR" \
      --output-json "$OUTPUT_JSON" \
      --output-detail-jsonl "$OUTPUT_DETAIL" \
      --ks "${KS[@]}" \
      --num-trials "$NUM_TRIALS" \
      --seed "$SEED" \
      --expected-count "$EXPECTED_COUNT" \
      --worker 32

    echo "[DONE][eval] ${OUTPUT_JSON}"
  done
done

echo "########## Stage 2: Average 8 groups inside each stage ##########"

for STAGE in "${STAGES[@]}"; do
  echo "[RUN ][avg ] ${STAGE}"

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from statistics import mean, pstdev

stage_dir = Path("${ROOT_DIR}/${STAGE}")
group_ids = list(range(8))
file_name = "eval_summary_mc.json"

metrics = [
    "acc@1",
    "valid_prediction_rate",
    "pass@1_mc",
    "pass@4_mc",
    "pass@8_mc",
    "pass@16_mc",
    "pass@32_mc",
    "majority@1_mc",
    "majority@4_mc",
    "majority@8_mc",
    "majority@16_mc",
    "majority@32_mc",
]

def safe_round(x, ndigits=6):
    if x is None:
        return None
    return round(float(x), ndigits)

summary = {
    "stage": "${STAGE}",
    "num_groups_found": 0,
    "groups_found": [],
    "groups_missing": [],
    "metrics": {},
}

metric_to_values = {m: [] for m in metrics}

for group in group_ids:
    file_path = stage_dir / f"random_seed_42_group_{group}" / file_name

    if not file_path.exists():
        summary["groups_missing"].append(group)
        print(f"[WARN] missing: {file_path}")
        continue

    try:
        with file_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception as e:
        summary["groups_missing"].append(group)
        print(f"[WARN] failed to read {file_path}: {e}")
        continue

    summary["groups_found"].append(group)

    for metric in metrics:
        val = result.get(metric, None)
        if val is None:
            print(f"[WARN] metric {metric} missing in {file_path}")
            continue
        metric_to_values[metric].append(float(val))

summary["num_groups_found"] = len(summary["groups_found"])

csv_lines = []
csv_lines.append("stage,metric,n,mean,std,min,max,values")

for metric in metrics:
    values = metric_to_values[metric]

    if len(values) == 0:
        summary["metrics"][metric] = {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "values": [],
        }
        csv_lines.append(f"${STAGE},{metric},0,,,,,")
        continue

    m_mean = mean(values)
    m_std = pstdev(values) if len(values) > 1 else 0.0
    m_min = min(values)
    m_max = max(values)

    summary["metrics"][metric] = {
        "n": len(values),
        "mean": safe_round(m_mean),
        "std": safe_round(m_std),
        "min": safe_round(m_min),
        "max": safe_round(m_max),
        "values": [safe_round(v) for v in values],
    }

    values_str = "|".join(f"{v:.6f}" for v in values)
    csv_lines.append(
        f"${STAGE},{metric},{len(values)},{m_mean:.6f},{m_std:.6f},{m_min:.6f},{m_max:.6f},{values_str}"
    )

out_json = stage_dir / "random_group_mean_summary.json"
out_csv = stage_dir / "random_group_mean_summary.csv"

with out_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

with out_csv.open("w", encoding="utf-8") as f:
    f.write("\\n".join(csv_lines) + "\\n")

print("=" * 80)
print(f"Stage: ${STAGE}")
print(f"  groups_found   = {summary['groups_found']}")
print(f"  groups_missing = {summary['groups_missing']}")
for metric in metrics:
    m = summary["metrics"][metric]
    print(
        f"  {metric:<18} mean={m['mean']} std={m['std']} min={m['min']} max={m['max']} n={m['n']}"
    )

print(f"Saved JSON -> {out_json}")
print(f"Saved CSV  -> {out_csv}")
PY

  echo "[DONE][avg ] ${STAGE}"
done

echo "Finished."