#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

CHOICES = ["A", "B", "C", "D"]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_text(s: str) -> str:
    return (
        s.replace("\u201c", '"')
         .replace("\u201d", '"')
         .replace("\u2018", "'")
         .replace("\u2019", "'")
    )


def extract_answer_letter(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    text = normalize_text(text)

    json_patterns = [
        r'"answer"\s*:\s*"([ABCD])"',
        r"'answer'\s*:\s*'([ABCD])'",
        r'"answer"\s*:\s*([ABCD])',
        r"'answer'\s*:\s*([ABCD])",
    ]
    for pat in json_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).upper()
            if ans in CHOICES:
                return ans

    code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in code_blocks:
        for pat in json_patterns:
            m = re.search(pat, block, flags=re.IGNORECASE)
            if m:
                ans = m.group(1).upper()
                if ans in CHOICES:
                    return ans

    fallback_patterns = [
        r'\banswer\s*[:=]\s*["\']?([ABCD])["\']?\b',
        r'\bfinal answer\s*[:=]\s*["\']?([ABCD])["\']?\b',
        r'\bcorrect answer\s*[:=]\s*["\']?([ABCD])["\']?\b',
        r'\boption\s*[:=]?\s*["\']?([ABCD])["\']?\b',
        r'\bchoice\s*[:=]?\s*["\']?([ABCD])["\']?\b',
    ]
    for pat in fallback_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).upper()
            if ans in CHOICES:
                return ans

    tail = "\n".join(text.strip().splitlines()[-8:])
    for pat in json_patterns + fallback_patterns:
        m = re.search(pat, tail, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).upper()
            if ans in CHOICES:
                return ans

    return None


def build_gold_map(question_file: str) -> Dict[str, Dict[str, Any]]:
    questions = load_jsonl(question_file)
    gold = {}

    for i, item in enumerate(questions):
        qid = (
            item.get("Record ID")
            or item.get("question_id")
            or item.get("id")
            or str(i)
        )
        qid = str(qid)

        gold[qid] = {
            "gold_letter": "A",
            "index": i,
            "question": item.get("Question", ""),
            "record_id": item.get("Record ID", qid),
        }

    return gold


def majority_vote_strict(preds: List[Optional[str]]) -> Optional[str]:
    valid = [p for p in preds if p in CHOICES]
    if not valid:
        return None

    cnt = Counter(valid)
    max_count = max(cnt.values())
    winners = [k for k, v in cnt.items() if v == max_count]

    if len(winners) != 1:
        return None
    return winners[0]


def monte_carlo_pass_at_k(
    preds: List[Optional[str]],
    gold: str,
    k: int,
    num_trials: int,
    rng: random.Random,
) -> float:
    n = len(preds)
    if n == 0:
        return 0.0

    if k > n:
        k = n

    success = 0
    indices = list(range(n))

    for _ in range(num_trials):
        chosen = rng.sample(indices, k)
        subset = [preds[i] for i in chosen]
        if any(p == gold for p in subset):
            success += 1

    return success / num_trials


def monte_carlo_majority_at_k(
    preds: List[Optional[str]],
    gold: str,
    k: int,
    num_trials: int,
    rng: random.Random,
) -> float:
    n = len(preds)
    if n == 0:
        return 0.0

    if k > n:
        k = n

    success = 0
    indices = list(range(n))

    for _ in range(num_trials):
        chosen = rng.sample(indices, k)
        subset = [preds[i] for i in chosen]
        maj = majority_vote_strict(subset)
        if maj == gold:
            success += 1

    return success / num_trials


def evaluate_question(
    pred_file: Path,
    gold_letter: str,
    ks: List[int],
    num_trials: int,
    base_seed: int,
) -> Dict[str, Any]:
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    completions = data.get("completions", [])
    extracted = []

    for comp in completions:
        text = comp.get("text", "")
        pred = extract_answer_letter(text)
        extracted.append(pred)

    qid = str(data.get("question_id", pred_file.stem))
    result = {
        "question_id": qid,
        "num_completions": len(completions),
        "predictions": extracted,
        "gold": gold_letter,
        "acc@1": 0,
        "num_valid_predictions": sum(p in CHOICES for p in extracted),
    }

    if len(extracted) > 0 and extracted[0] in CHOICES:
        result["acc@1"] = int(extracted[0] == gold_letter)

    try:
        qid_offset = int(re.sub(r"\D", "", qid)) if re.search(r"\d", qid) else abs(hash(qid)) % (10**9)
    except Exception:
        qid_offset = abs(hash(qid)) % (10**9)

    for k in ks:
        rng_pass = random.Random(base_seed + 1000003 * (qid_offset + 1) + k * 17 + 1)
        rng_maj = random.Random(base_seed + 1000003 * (qid_offset + 1) + k * 17 + 2)

        result[f"pass@{k}_mc"] = round(
            monte_carlo_pass_at_k(extracted, gold_letter, k, num_trials, rng_pass),
            6,
        )
        result[f"majority@{k}_mc"] = round(
            monte_carlo_majority_at_k(extracted, gold_letter, k, num_trials, rng_maj),
            6,
        )

    return result


def resolve_gold_info(raw: Dict[str, Any], pred_file: Path, gold_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    qid = str(raw.get("question_id", ""))
    if qid in gold_map:
        return gold_map[qid]

    idx_key = pred_file.stem
    if idx_key in gold_map:
        return gold_map[idx_key]

    try:
        idx = int(pred_file.stem)
        return gold_map.get(str(idx))
    except Exception:
        return None


def evaluate_one_file(task: Tuple[str, Dict[str, Any], List[int], int, int]) -> Tuple[bool, Any]:
    pred_file_str, gold_info, ks, num_trials, seed = task
    pred_file = Path(pred_file_str)
    try:
        one = evaluate_question(
            pred_file=pred_file,
            gold_letter=gold_info["gold_letter"],
            ks=ks,
            num_trials=num_trials,
            base_seed=seed,
        )
        one["record_id"] = gold_info["record_id"]
        one["question_index"] = gold_info["index"]
        return True, one
    except Exception as e:
        return False, {"file": pred_file.name, "error": repr(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPQA predictions with Monte Carlo majority@k")
    parser.add_argument("--question-file", type=str, required=True, help="GPQA question jsonl")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing per-question prediction json files")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save summary json")
    parser.add_argument("--output-detail-jsonl", type=str, required=True, help="Path to save per-question detail jsonl")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32], help="k values for Monte Carlo pass@k / majority@k")
    parser.add_argument("--num-trials", type=int, default=5000, help="Monte Carlo trials per question per k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base")
    parser.add_argument("--expected-count", type=int, default=None, help="Optional sanity check, e.g. 198 for GPQA-Diamond")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for parallel evaluation")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    assert pred_dir.exists(), f"pred-dir not found: {pred_dir}"

    gold_map = build_gold_map(args.question_file)

    pred_files = sorted(
        [p for p in pred_dir.glob("*.json") if p.stem.isdigit()],
        key=lambda x: int(x.stem)
    )

    tasks = []
    missing_gold = []
    failed_eval = []
    details = []
    missing_pred = []

    for pred_file in pred_files:
        with open(pred_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        gold_info = resolve_gold_info(raw, pred_file, gold_map)
        if gold_info is None:
            missing_gold.append(pred_file.name)
            continue

        tasks.append((str(pred_file), gold_info, args.ks, args.num_trials, args.seed))

    if args.workers <= 1:
        for task in tqdm(tasks, desc="Evaluating GPQA"):
            ok, payload = evaluate_one_file(task)
            if ok:
                details.append(payload)
            else:
                failed_eval.append(payload)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one_file, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating GPQA"):
                ok, payload = future.result()
                if ok:
                    details.append(payload)
                else:
                    failed_eval.append(payload)

        details.sort(key=lambda x: x["question_index"])

    if args.expected_count is not None:
        found_indices = {d["question_index"] for d in details}
        for i in range(args.expected_count):
            if i not in found_indices:
                missing_pred.append(i)

    summary = {
        "num_questions_evaluated": len(details),
        "num_missing_gold_match": len(missing_gold),
        "missing_gold_match_files": missing_gold,
        "num_failed_eval": len(failed_eval),
        "failed_eval_files": failed_eval,
        "num_missing_predictions": len(missing_pred),
        "missing_prediction_indices": missing_pred,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "workers": args.workers,
    }

    if len(details) > 0:
        summary["acc@1"] = round(sum(d["acc@1"] for d in details) / len(details), 6)
        valid_rate = sum(d["num_valid_predictions"] > 0 for d in details) / len(details)
        summary["valid_prediction_rate"] = round(valid_rate, 6)

        for k in args.ks:
            summary[f"pass@{k}_mc"] = round(
                sum(d[f"pass@{k}_mc"] for d in details) / len(details),
                6,
            )
            summary[f"majority@{k}_mc"] = round(
                sum(d[f"majority@{k}_mc"] for d in details) / len(details),
                6,
            )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.output_detail_jsonl, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()