"""
Self-Reflection Step 2: Blind Self-Assessment (NO evaluation signal)

The model reviews its own plan output and the query constraints to
self-assess potential issues — WITHOUT knowing which checks actually
passed or failed. This extracts BOTH positive and negative experiences
from ALL queries and ALL completions.

For each query, iterates over ALL completions from the plan file,
generates one reflection per completion.  Output reflections[i]
corresponds to completions[i] from the input plan file.

Input:
  - Previous iteration's plan directory (per-query JSON files)
  - Validation data (from database/) — for query details & reference info

Output: one JSON per query under {output_dir}/{idx}.json
Format: {
    "index": int,
    "query": str,
    "query_data": {...},
    "reflections": [
        {
            "completion_idx": 0,
            "plan": [...],
            "text": str, "reasoning_content": str, "reflection_text": str,
            "reflection": {propositions: [...], pitfalls: [...], summary: str},
            "tokens": int, "finish_reason": str
        },
        ...
    ],
    "n_reflections": int
}
"""
import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import OrderedDict

from vllm import LLM, SamplingParams


SELF_REFLECT_SYSTEM_PROMPT = """You are an expert travel planning analyst performing a SELF-ASSESSMENT. You will be given a travel query, reference data, and a generated plan. Your job is to carefully review the plan against the query constraints and the reference data, then extract **generalizable, actionable knowledge** — both things done CORRECTLY (positive experiences) and potential MISTAKES (negative experiences).

**IMPORTANT**: You do NOT have access to any automated evaluation results. You must verify the plan yourself by carefully checking each constraint.

## TravelPlanner Evaluation Criteria

A travel plan is evaluated on **two layers**. You must check BOTH layers yourself.

### Layer 1 — Commonsense Constraints (8 checks)

**1. is_reasonable_visiting_city — Route & City Validation**
Check the `current_city` field across all days:
- Build a `city_list`: if `current_city` contains `"from"`, extract both cities (`from A to B` → append A, B); otherwise append the single city.
- **Closed loop**: `city_list[0]` must equal `city_list[-1]` (both = origin city).
- **Consecutive rule**: Every intermediate city (not first/last) must appear **consecutively at least 2 times**.
- **Day-1 origin**: If Day 1 uses `"from A to B"`, A must be the origin city.

**2. is_valid_restaurants — No Restaurant Repetition**
Collect ALL restaurant names across all meals/days into one list. Any duplicate = FAIL.

**3. is_valid_attractions — No Attraction Repetition**
Split attractions by `;` across all days. Any duplicate = FAIL.

**4. is_valid_accommodation — Minimum Nights & Presence**
- Accommodation must be present for every day EXCEPT the return day.
- Group consecutive identical accommodations; count must be ≥ `minimum_nights` from the database.

**5. is_valid_transportation — Day-1 Required & No Conflicts**
- Day 1 MUST have transportation.
- No mixing: Self-driving + Flight = forbidden; Taxi + Self-driving = forbidden.

**6. is_valid_information_in_current_city — City-Tag Matching**
Each restaurant/attraction/accommodation/transportation must contain the current day's city name as a substring (format: `"ItemName, CityName"`).

**7. is_valid_information_in_sandbox — Database Existence**
Every item must actually exist in the reference data for the specified city.

**8. is_not_absent — Completeness**
ALL 7 fields must be non-empty every day. The ONLY exception: return day's accommodation may be `"-"`. Everything else (including return day meals, attractions, transportation) must be filled.

### Layer 2 — Hard Constraints (5 checks)

**1. valid_cost — Budget Compliance**
```
flights: price × N_people
taxi: cost_per_vehicle × ceil(N_people / 4)    (capacity = 4)
self-driving: cost × ceil(N_people / 5)        (capacity = 5)
meals: average_cost × N_people
accommodation: price × ceil(N_people / max_occupancy)
Total must ≤ budget.
```

**2. valid_room_rule — House Rule Compliance**
If constraint like `"children under 10"` is given, accommodation must NOT have `"No children under 10"` in house_rules.

**3. valid_cuisine — Cuisine Coverage**
If cuisine types are specified, at least one meal in a destination city must serve each required cuisine.

**4. valid_transportation — Mode Restriction**
If `"no flight"` → no flights allowed. If `"no self-driving"` → no self-driving allowed.

**5. valid_room_type — Room Type Matching**
`"entire room"` → must be `"Entire home/apt"` in DB. `"private room"` → `"Private room"`. `"shared room"` → `"Shared room"`. `"not shared room"` → must NOT be `"Shared room"`.

## Your Task

Carefully verify the plan against ALL constraints above. Then produce:

### 1. Propositions — Actionable knowledge (both positive and negative)
- **Positive**: Things the plan got RIGHT that should be reinforced (e.g., "Correctly used 'from A to B' format on travel days, ensuring closed-loop route")
- **Negative**: Things the plan got WRONG with the correct approach (e.g., "Taxi capacity is 4, not 6 — recalculate ground transportation cost as ceil(N/4) × per_vehicle_cost")

### 2. Pitfalls — Specific mistake patterns found in THIS plan
Only include pitfalls for things you actually verified are WRONG. Do not speculate.

### 3. Summary — Brief overall assessment
State which constraints appear satisfied and which appear violated.

## Output Format

```json
{
    "propositions": [
        {
            "statement": "A generalizable, actionable rule (imperative mood)",
            "category": "city_route|restaurant|attraction|accommodation|transportation|budget|completeness|data_validity",
            "severity": "critical|important|minor",
            "eval_check": "the specific evaluation check this addresses",
            "type": "positive|negative"
        }
    ],
    "pitfalls": [
        {
            "statement": "A concrete error pattern found in this plan",
            "category": "city_route|restaurant|attraction|accommodation|transportation|budget|completeness|data_validity",
            "eval_check": "the specific evaluation check this addresses"
        }
    ],
    "summary": "Brief overall assessment: which constraints appear satisfied, which appear violated, and root causes"
}
```

## Guidelines
- Check EVERY constraint yourself. Do not assume anything passes or fails without verification.
- Generate both POSITIVE and NEGATIVE propositions.
- Be concrete: cite actual data from the plan (costs, names, capacity numbers) as examples.
- Pitfalls must describe ACTUAL errors you found, not hypothetical ones.
- If the plan looks completely correct, say so in the summary and generate positive propositions only.

Output ONLY the JSON object, no other text."""


def load_validation_data(data_dir: str):
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


def load_plan_result(plan_dir: str, idx: int):
    """Load a plan result file (contains all completions for one query)."""
    fpath = os.path.join(plan_dir, f"{idx}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_plan_text(plan):
    """Format a plan list into human-readable text."""
    if not plan:
        return "(No plan generated)"

    lines = []
    for day in plan:
        if not day or not isinstance(day, dict):
            continue
        d = day.get('days', '?')
        lines.append(f"Day {d}:")
        lines.append(f"  Current City: {day.get('current_city', '-')}")
        lines.append(f"  Transportation: {day.get('transportation', '-')}")
        lines.append(f"  Breakfast: {day.get('breakfast', '-')}")
        lines.append(f"  Attraction: {day.get('attraction', '-')}")
        lines.append(f"  Lunch: {day.get('lunch', '-')}")
        lines.append(f"  Dinner: {day.get('dinner', '-')}")
        lines.append(f"  Accommodation: {day.get('accommodation', '-')}")
        lines.append("")
    return "\n".join(lines)


def extract_thinking(text: str):
    """Extract <think>...</think> reasoning content."""
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return reasoning, final_text
    return "", text


def extract_json_obj(text: str):
    """Try to extract a JSON object from text."""
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def build_user_message(q, plan_text):
    """Build the user message for self-reflection."""
    lc = q['local_constraint']
    constraint_lines = []
    if lc.get('cuisine'):
        constraint_lines.append(f"  - Cuisine requirement: {lc['cuisine']}")
    if lc.get('room type'):
        constraint_lines.append(f"  - Room type: {lc['room type']}")
    if lc.get('house rule'):
        constraint_lines.append(f"  - House rule: {lc['house rule']}")
    if lc.get('transportation'):
        constraint_lines.append(f"  - Transportation: {lc['transportation']}")
    constraint_text = "\n".join(constraint_lines) if constraint_lines else "  (none)"

    ref_text = json.dumps(q['reference_information'], ensure_ascii=False)
    if len(ref_text) > 8000:
        ref_text = ref_text[:8000] + "\n... (truncated)"

    return f"""## Query
{q['query']}

## Query Details
- Origin city: {q['org']}
- Destination state: {q['dest']}
- Number of days: {q['days']}
- Number of people: {q['people_number']}
- Budget: ${q['budget']}
- Number of cities to visit: {q['visiting_city_number']}
- Date range: {q['date']}
- Local constraints:
{constraint_text}

## Reference Information
{ref_text}

## Generated Plan
{plan_text}

Carefully verify this plan against ALL constraints listed in the system prompt. Check each constraint yourself — you do NOT have access to any automated evaluation results. Identify both what was done correctly (positive experiences) and what was done incorrectly (negative experiences / pitfalls). Output ONLY the JSON object."""


def main():
    parser = argparse.ArgumentParser(
        description="Self-Reflection Step 2: Blind Self-Assessment (no eval signal)")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--data-dir", type=str, default="<DATABASE_PATH>",
                        help="Path to database dir (for query details)")
    parser.add_argument("--plan-dir", type=str, required=True,
                        help="Directory containing plan JSONs (from step1 or step3)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=38912)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading validation data from {args.data_dir}...")
    queries = load_validation_data(args.data_dir)
    total = len(queries)
    print(f"Loaded {total} queries")

    end_idx = total if args.end_idx == -1 else min(args.end_idx, total)
    start_idx = args.start_idx
    print(f"Processing queries [{start_idx}, {end_idx})")

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        seed=args.seed,
        trust_remote_code=True,
        max_model_len=262144,
    )
    tokenizer = llm.get_tokenizer()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = []
    prompt_sampling_params = []
    prompt_keys = []
    plan_cache = {}
    skipped = 0
    no_plan = 0
    total_completions = 0

    for idx in range(start_idx, end_idx):
        plan_result = load_plan_result(args.plan_dir, idx)
        if plan_result is None:
            no_plan += 1
            continue

        completions = plan_result.get('completions', [])
        if not completions:
            no_plan += 1
            continue

        out_path = os.path.join(args.output_dir, f"{idx}.json")
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if existing.get('n_reflections', 0) >= len(completions):
                    skipped += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        q = queries[idx]
        query_has_valid_comp = False

        for comp_idx, comp in enumerate(completions):
            plan = comp.get('plan')
            if plan is None:
                continue

            plan_cache[(idx, comp_idx)] = plan
            plan_text = format_plan_text(plan)
            user_msg = build_user_message(q, plan_text)

            messages = [
                {"role": "system", "content": SELF_REFLECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            prompts.append(prompt)
            prompt_keys.append((idx, comp_idx))
            prompt_sampling_params.append(SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                n=1,
                seed=args.seed + comp_idx,
                top_p=0.95,
                top_k=20,
            ))
            query_has_valid_comp = True

        if query_has_valid_comp:
            total_completions += len([c for c in completions if c.get('plan') is not None])

    n_queries = len(set(k[0] for k in prompt_keys))
    print(f"Range [{start_idx}, {end_idx}): "
          f"{skipped} already done, {no_plan} no plan, "
          f"{n_queries} queries, {len(prompts)} total prompts "
          f"(reflecting on {total_completions} completions)")

    if not prompts:
        print("Nothing to do.")
        return

    print(f"Running self-reflection on {len(prompts)} requests (n=1 each)...")
    outputs = llm.generate(prompts, prompt_sampling_params)

    grouped = OrderedDict()
    for i, output in enumerate(outputs):
        idx, comp_idx = prompt_keys[i]
        if idx not in grouped:
            grouped[idx] = []
        grouped[idx].append((comp_idx, output.outputs[0]))

    for idx, items in grouped.items():
        q = queries[idx]

        items.sort(key=lambda x: x[0])

        reflections = []
        for comp_idx, out in items:
            reasoning_content, answer_text = extract_thinking(out.text)
            reflection_obj = extract_json_obj(answer_text)
            reflections.append({
                "completion_idx": comp_idx,
                "plan": plan_cache[(idx, comp_idx)],
                "text": out.text,
                "reasoning_content": reasoning_content,
                "reflection_text": answer_text,
                "reflection": reflection_obj,
                "tokens": len(out.token_ids),
                "finish_reason": out.finish_reason,
            })

        result = {
            "index": idx,
            "query": q['query'],
            "query_data": {
                "org": q['org'],
                "dest": q['dest'],
                "days": q['days'],
                "visiting_city_number": q['visiting_city_number'],
                "date": q['date'],
                "people_number": q['people_number'],
                "local_constraint": q['local_constraint'],
                "budget": q['budget'],
                "level": q['level'],
            },
            "reflections": reflections,
            "n_reflections": len(reflections),
        }

        out_path = os.path.join(args.output_dir, f"{idx}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Self-reflections saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
