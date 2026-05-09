"""
RSE Evaluation: Offline evaluation of travel plans using validation data.
No HuggingFace dependency — loads query data from local CSV.

Supports two input modes:
  1. Directory of per-query JSON files (from step1/step3): --input-dir
  2. Single JSONL file (standard TravelPlanner format): --input-file

Outputs a summary JSON with per-query results and aggregated metrics.
"""
import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

# We need to run evaluation from within the TravelPlanner evaluation directory
# so that relative imports in hard_constraint.py / commonsense_constraint.py work.
TRAVELPLANNER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_validation_queries(data_dir: str):
    """Load validation queries from CSV."""
    csv_path = os.path.join(data_dir, "validation.csv")
    queries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['days'] = int(row['days'])
            row['people_number'] = int(row['people_number'])
            row['budget'] = int(row['budget'])
            row['visiting_city_number'] = int(row['visiting_city_number'])
            row['date'] = ast.literal_eval(row['date'])
            row['local_constraint'] = ast.literal_eval(row['local_constraint'])
            row['reference_information'] = ast.literal_eval(row['reference_information'])
            queries.append(row)
    return queries


def load_plans_from_dir(input_dir: str, n_queries: int, completion_idx: int = 0):
    """Load plans from per-query JSON files.
    Returns list of plan dicts (or None for missing/failed).
    """
    plans = []
    for idx in range(n_queries):
        fpath = os.path.join(input_dir, f"{idx}.json")
        if not os.path.exists(fpath):
            plans.append(None)
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get plan from completions
        plan = None
        if 'completions' in data and len(data['completions']) > completion_idx:
            plan = data['completions'][completion_idx].get('plan', None)

        plans.append(plan)
    return plans


def load_plans_from_jsonl(input_file: str):
    """Load plans from a JSONL file (standard TravelPlanner eval format)."""
    plans = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                plans.append(None)
                continue
            data = json.loads(line)
            if isinstance(data, dict) and 'plan' in data:
                plans.append(data['plan'])
            else:
                plans.append(data)
    return plans


def run_evaluation(queries, plans, database_dir=None):
    """Run TravelPlanner evaluation on plans.
    Changes cwd to evaluation/ so that relative imports work.

    The TravelPlanner evaluation modules (commonsense_constraint.py, hard_constraint.py)
    import tool APIs (Flights, Accommodations, etc.) at module level. Those APIs load
    CSVs from ``../database/`` relative to the evaluation/ directory. If the real database
    is somewhere else (e.g. --data-dir points to a different location), we create a
    temporary symlink ``<evaluation>/../database -> <actual database>`` so the hardcoded
    ``../database/`` paths resolve correctly.
    """
    original_cwd = os.getcwd()

    eval_dir = os.path.join(TRAVELPLANNER_ROOT, "evaluation")

    # ── Ensure ../database/ relative to evaluation/ points to the real DB ──
    expected_db_dir = os.path.normpath(os.path.join(eval_dir, "..", "database"))
    created_symlink = False

    if database_dir is not None:
        real_db = os.path.abspath(database_dir)
        # Only create symlink if expected path doesn't already contain the
        # required sub-directories (flights/, accommodations/, etc.)
        needs_link = (
            not os.path.isdir(os.path.join(expected_db_dir, "flights"))
            and os.path.isdir(os.path.join(real_db, "flights"))
            and os.path.normpath(expected_db_dir) != os.path.normpath(real_db)
        )
        if needs_link:
            # If expected_db_dir exists but is incomplete, rename it
            if os.path.exists(expected_db_dir) and not os.path.islink(expected_db_dir):
                backup = expected_db_dir + "_backup"
                if not os.path.exists(backup):
                    os.rename(expected_db_dir, backup)
                    print(f"  Backed up existing database dir → {backup}")
                else:
                    # backup already exists, skip
                    pass

            if not os.path.exists(expected_db_dir):
                os.symlink(real_db, expected_db_dir)
                created_symlink = True
                print(f"  Created symlink: {expected_db_dir} → {real_db}")

    os.chdir(eval_dir)
    sys.path.insert(0, eval_dir)
    sys.path.insert(0, TRAVELPLANNER_ROOT)

    try:
        from commonsense_constraint import evaluation as commonsense_eval
        from hard_constraint import evaluation as hard_eval
    except FileNotFoundError as e:
        os.chdir(original_cwd)
        if created_symlink:
            os.remove(expected_db_dir)
        raise RuntimeError(
            f"Failed to load evaluation modules. The TravelPlanner database "
            f"sub-directories (flights/, accommodations/, restaurants/, attractions/, "
            f"googleDistanceMatrix/, background/) must exist under "
            f"{expected_db_dir}.\n"
            f"You can download them from the TravelPlanner repo or specify "
            f"--database-dir pointing to the directory that contains them.\n"
            f"Original error: {e}"
        ) from e

    per_query_results = []
    delivery_cnt = 0
    final_commonsense_cnt = 0
    final_hard_cnt = 0
    final_all_cnt = 0

    commonsense_micro = {"pass": 0, "total": 0}
    hard_micro = {"pass": 0, "total": 0}

    for idx in range(len(queries)):
        query_data = queries[idx]
        plan = plans[idx] if idx < len(plans) else None

        if isinstance(query_data.get('local_constraint'), str):
            query_data['local_constraint'] = ast.literal_eval(query_data['local_constraint'])

        result = {
            "index": idx,
            "level": query_data['level'],
            "days": query_data['days'],
            "delivered": False,
            "commonsense_constraint": None,
            "hard_constraint": None,
            "commonsense_pass": False,
            "hard_pass": False,
            "final_pass": False,
            "error": None,
        }

        if plan is None or plan == [] or plan == [{}]:
            result["error"] = "No plan generated"
            per_query_results.append(result)
            continue

        # Filter out empty trailing dicts (fine-tuning data pads with {})
        plan = [d for d in plan if d and isinstance(d, dict) and d.get('current_city')]

        if not plan:
            result["error"] = "Plan is empty after filtering"
            per_query_results.append(result)
            continue

        delivery_cnt += 1
        result["delivered"] = True

        try:
            commonsense_info = commonsense_eval(query_data, plan)
        except Exception as e:
            result["error"] = f"Commonsense eval error: {str(e)}"
            per_query_results.append(result)
            continue

        result["commonsense_constraint"] = {
            k: {"pass": v[0], "reason": v[1]} for k, v in commonsense_info.items()
        }

        # Count commonsense micro
        for k, v in commonsense_info.items():
            if v[0] is not None:
                commonsense_micro["total"] += 1
                if v[0]:
                    commonsense_micro["pass"] += 1

        # Check if commonsense passes
        cs_pass = True
        for item in commonsense_info:
            if commonsense_info[item][0] is not None and not commonsense_info[item][0]:
                cs_pass = False
                break
        result["commonsense_pass"] = cs_pass

        if cs_pass:
            final_commonsense_cnt += 1

        # Hard constraint eval (only if commonsense basics pass)
        if commonsense_info.get('is_not_absent', (False,))[0] and \
           commonsense_info.get('is_valid_information_in_sandbox', (False,))[0]:
            try:
                hard_info = hard_eval(query_data, plan)
            except Exception as e:
                result["error"] = f"Hard eval error: {str(e)}"
                per_query_results.append(result)
                continue

            result["hard_constraint"] = {
                k: {"pass": v[0], "reason": v[1]} for k, v in hard_info.items()
            }

            # Count hard micro
            for k, v in hard_info.items():
                if v[0] is not None:
                    hard_micro["total"] += 1
                    if v[0]:
                        hard_micro["pass"] += 1

            # Check if hard passes
            h_pass = True
            for item in hard_info:
                if hard_info[item][0] is not None and hard_info[item][0] == False:
                    h_pass = False
                    break
            result["hard_pass"] = h_pass

            if h_pass:
                final_hard_cnt += 1

            if cs_pass and h_pass:
                final_all_cnt += 1
                result["final_pass"] = True

        per_query_results.append(result)

    total = len(queries)
    summary = {
        "total_queries": total,
        "delivery_rate": delivery_cnt / total if total > 0 else 0,
        "commonsense_micro_pass_rate": commonsense_micro["pass"] / commonsense_micro["total"] if commonsense_micro["total"] > 0 else 0,
        "commonsense_macro_pass_rate": final_commonsense_cnt / total if total > 0 else 0,
        "hard_micro_pass_rate": hard_micro["pass"] / hard_micro["total"] if hard_micro["total"] > 0 else 0,
        "hard_macro_pass_rate": final_hard_cnt / total if total > 0 else 0,
        "final_pass_rate": final_all_cnt / total if total > 0 else 0,
        "counts": {
            "delivered": delivery_cnt,
            "commonsense_pass": final_commonsense_cnt,
            "hard_pass": final_hard_cnt,
            "final_pass": final_all_cnt,
        },
    }

    os.chdir(original_cwd)
    return summary, per_query_results


def main():
    parser = argparse.ArgumentParser(description="RSE Offline Evaluation for TravelPlanner")
    parser.add_argument("--data-dir", type=str, default="../database",
                        help="Path to database dir containing validation.csv")
    parser.add_argument("--database-dir", type=str, default=None,
                        help="Path to the full TravelPlanner database dir "
                        "(containing flights/, accommodations/, restaurants/, etc.). "
                        "If different from --data-dir, a symlink will be created so "
                        "the evaluation code can find the CSVs. "
                        "Defaults to the same as --data-dir.")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with per-query JSON files (from step1/step3)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="JSONL file with plans (standard TravelPlanner format)")
    parser.add_argument("--completion-idx", type=int, default=0,
                        help="Which completion to evaluate (for multi-completion runs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for detailed results (optional)")
    args = parser.parse_args()

    assert args.input_dir or args.input_file, \
        "Must specify either --input-dir or --input-file"

    # Resolve database dir
    database_dir = args.database_dir if args.database_dir else args.data_dir

    # Load queries
    print(f"Loading validation queries from {args.data_dir}...")
    queries = load_validation_queries(args.data_dir)
    print(f"Loaded {len(queries)} queries")

    # Load plans
    if args.input_dir:
        print(f"Loading plans from directory: {args.input_dir}")
        plans = load_plans_from_dir(args.input_dir, len(queries), args.completion_idx)
    else:
        print(f"Loading plans from file: {args.input_file}")
        plans = load_plans_from_jsonl(args.input_file)

    print(f"Loaded {len(plans)} plans")

    # Run evaluation
    print("Running evaluation...")
    summary, per_query_results = run_evaluation(queries, plans, database_dir=database_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, val in summary.items():
        if key == "counts":
            print(f"  Counts:")
            for ck, cv in val.items():
                print(f"    {ck}: {cv}")
        else:
            if isinstance(val, float):
                print(f"  {key}: {val:.4f} ({val*100:.2f}%)")
            else:
                print(f"  {key}: {val}")

    # Count per-level
    level_stats = {}
    for r in per_query_results:
        lv = r['level']
        if lv not in level_stats:
            level_stats[lv] = {"total": 0, "delivered": 0, "cs_pass": 0, "hard_pass": 0, "final_pass": 0}
        level_stats[lv]["total"] += 1
        if r["delivered"]:
            level_stats[lv]["delivered"] += 1
        if r["commonsense_pass"]:
            level_stats[lv]["cs_pass"] += 1
        if r["hard_pass"]:
            level_stats[lv]["hard_pass"] += 1
        if r["final_pass"]:
            level_stats[lv]["final_pass"] += 1

    print("\nPer-Level Results:")
    for lv in ['easy', 'medium', 'hard']:
        if lv in level_stats:
            s = level_stats[lv]
            print(f"  {lv}: final_pass={s['final_pass']}/{s['total']} "
                  f"({s['final_pass']/s['total']*100:.1f}%), "
                  f"cs_pass={s['cs_pass']}/{s['total']}, "
                  f"hard_pass={s['hard_pass']}/{s['total']}")

    # Save detailed results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "per_level": level_stats,
                "per_query": per_query_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
