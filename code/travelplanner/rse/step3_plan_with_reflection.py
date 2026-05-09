"""
RSE Step 3: Plan Generation with Reflection for TravelPlanner

Mirrors Math domain's step3_answer_with_reflection_v4.py:
  - Reads per-question deduped reflection files from a directory
  - Each query uses ONLY its own reflections (no cross-query sharing)

Input:
  - Validation data (from database/)
  - Per-question deduped reflection directory ({idx}.jsonl files from step2dot2)

Output: one JSON per query under {output_dir}/{idx}.json
Same format as step1.
"""
import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams


SYSTEM_PROMPT_WITH_REFLECTION = """You are a proficient planner. Based on the provided information, query, and past experiences/lessons learned, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte;
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte;
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte;
Lunch: Taste of Beijing, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

You MUST output the travel plan as a JSON array. Each element is a dict with keys:
"days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation".
Use "-" for fields that are not applicable. Attractions should be separated by semicolons with a trailing semicolon.
Output ONLY the JSON array, no other text.

## CRITICAL RULES (violations cause automatic failure)

1. **current_city format**: On travel days (departure and return), use `"from CityA to CityB"`. On non-travel days, use a plain city name. The trip MUST form a closed loop: the first city in Day 1 must equal the last city on the return day (both = origin city). Every intermediate city must appear on at least 2 consecutive days.
2. **No restaurant repetition**: Every restaurant name must appear EXACTLY ONCE across all meals on all days. Never reuse a restaurant even across different meal types.
3. **No attraction repetition**: Every attraction name must appear exactly once across all days. Use semicolons to separate multiple attractions, with a trailing semicolon.
4. **Return day completeness**: On the return day, set accommodation to `"-"`, but you MUST still fill in breakfast, lunch, dinner, attraction, and transportation with valid data from the last destination city — do NOT set them to `"-"`.
5. **Item format**: Always format restaurants, attractions, and accommodations as `"Name, City"` (name followed by comma and city name). Every item must exist in the provided reference data for that city.
6. **Accommodation min-nights**: If you use the same accommodation for N consecutive days, the accommodation's `minimum nights` in the reference data must be ≤ N.
7. **Cost calculation**: Budget check uses these exact formulas:
   - Flights: price × N_people
   - Taxi: cost_per_vehicle × ceil(N_people / 4)  (taxi capacity = 4)
   - Self-driving: cost_per_vehicle × ceil(N_people / 5)  (capacity = 5)
   - Meals: average_cost × N_people
   - Accommodation: price_per_night × ceil(N_people / max_occupancy)
   Verify total ≤ budget before finalizing.
8. **Room constraints**: If house rule constraint is given (e.g., "children under 10"), the accommodation must NOT have "No children under 10" in its house_rules. If room type is "entire room", the accommodation's room type must be "Entire home/apt" in the DB.

## Past Experiences / Lessons Learned

Pay special attention to the following experiences from past attempts on THIS specific query:

{reflections}"""


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


def load_per_question_reflection(reflection_dir: str, idx: int):
    """
    Load deduped reflection for a single question.
    Mirrors Math domain: reads {idx}.jsonl from the dedup output directory.
    """
    fpath = os.path.join(reflection_dir, f"{idx}.jsonl")
    if not os.path.exists(fpath):
        return None

    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    content = record.get('reflection_parsed')
                    if isinstance(content, str):
                        content = json.loads(content)
                    return content
    except Exception:
        pass
    return None


def format_reflection_context(reflection_data: dict, max_items: int = 50) -> str:
    """Format per-question reflections into text for the system prompt."""
    if not reflection_data:
        return "(No past experiences available for this query.)"

    lines = []

    propositions = reflection_data.get('propositions', [])
    pitfalls = reflection_data.get('pitfalls', [])

    if propositions:
        severity_order = {'critical': 0, 'important': 1, 'minor': 2, 'unknown': 3}
        sorted_props = sorted(propositions,
                              key=lambda p: severity_order.get(
                                  p.get('severity', 'unknown') if isinstance(p, dict) else 'unknown', 3))

        positive_props = [p for p in sorted_props
                          if isinstance(p, dict) and p.get('prop_type', '').lower() == 'positive']
        negative_props = [p for p in sorted_props
                          if isinstance(p, dict) and p.get('prop_type', '').lower() == 'negative']
        untyped_props = [p for p in sorted_props
                         if isinstance(p, dict) and p.get('prop_type', '').lower() not in ('positive', 'negative')]
        str_props = [p for p in sorted_props if isinstance(p, str)]

        def _format_prop(i, prop):
            if isinstance(prop, str):
                return f"{i}. {prop}"
            severity = prop.get('severity', 'important').upper()
            cat = prop.get('category', '')
            check = prop.get('eval_check', '')
            check_tag = f" (addresses: {check})" if check else ""
            stmt = prop.get('statement', str(prop))
            return f"{i}. [{severity}] [{cat}]{check_tag} {stmt}"

        if positive_props:
            lines.append("=== Correct Patterns to Reinforce ===")
            for i, prop in enumerate(positive_props[:max_items], 1):
                lines.append(_format_prop(i, prop))

        if negative_props:
            lines.append("\n=== Mistakes to Fix ===")
            for i, prop in enumerate(negative_props[:max_items], 1):
                lines.append(_format_prop(i, prop))

        if untyped_props or str_props:
            lines.append("\n=== Key Experiences / Rules ===")
            for i, prop in enumerate((untyped_props + str_props)[:max_items], 1):
                lines.append(_format_prop(i, prop))

    if pitfalls:
        lines.append("\n=== Common Pitfalls to Avoid ===")
        for i, pit in enumerate(pitfalls[:max_items], 1):
            if isinstance(pit, str):
                lines.append(f"{i}. {pit}")
            else:
                cat = pit.get('category', '')
                check = pit.get('eval_check', '')
                check_tag = f" (addresses: {check})" if check else ""
                lines.append(f"{i}. [{cat}]{check_tag} {pit.get('statement', str(pit))}")

    return "\n".join(lines) if lines else "(No past experiences available for this query.)"


def extract_thinking(text: str):
    """Extract <think>...</think> reasoning content."""
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return reasoning, final_text
    return "", text


def extract_plan_json(text: str):
    """Try to extract a JSON array from the model output."""
    m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        try:
            return ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="RSE Step 3: Plan with Reflection (per-question)")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--data-dir", type=str, default="<DATABASE_PATH>",
                        help="Path to database dir")
    parser.add_argument("--reflection-dir", type=str, required=True,
                        help="Per-question deduped reflections directory (from step2dot2)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-completions", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=38912)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-reflection-items", type=int, default=50,
                        help="Max number of reflection items to include in prompt")
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
    indices = []
    skipped = 0
    no_reflection = 0
    unique_queries = 0

    for idx in range(start_idx, end_idx):
        out_path = os.path.join(args.output_dir, f"{idx}.json")
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if existing.get('n_completions', 0) >= args.n_completions:
                    skipped += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        reflection_data = load_per_question_reflection(args.reflection_dir, idx)
        if reflection_data is None:
            no_reflection += 1
            continue

        reflection_text = format_reflection_context(reflection_data, args.max_reflection_items)
        system_prompt = SYSTEM_PROMPT_WITH_REFLECTION.replace("{reflections}", reflection_text)

        q = queries[idx]
        user_msg = f"Given Information: {json.dumps(q['reference_information'], ensure_ascii=False)}\nQuery: {q['query']}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        for dup in range(args.n_completions):
            prompts.append(prompt)
            indices.append(idx)
            prompt_sampling_params.append(SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                n=1,
                seed=args.seed + dup,
                top_p=0.95,
                top_k=20,
            ))
        unique_queries += 1

    print(f"Range [{start_idx}, {end_idx}): {skipped} already done, "
          f"{no_reflection} no reflection, "
          f"{unique_queries} queries × {args.n_completions} = {len(prompts)} requests")

    if not prompts:
        print("All queries already completed. Nothing to do.")
        return

    print(f"Running inference on {len(prompts)} requests (n=1 each)...")
    outputs = llm.generate(prompts, prompt_sampling_params)

    from collections import OrderedDict
    grouped = OrderedDict()
    for i, output in enumerate(outputs):
        idx = indices[i]
        if idx not in grouped:
            grouped[idx] = []
        grouped[idx].append(output.outputs[0])

    for idx, outs in grouped.items():
        q = queries[idx]
        completions = []
        for out in outs:
            reasoning_content, answer_text = extract_thinking(out.text)
            plan = extract_plan_json(answer_text)
            completions.append({
                "text": out.text,
                "reasoning_content": reasoning_content,
                "answer_text": answer_text,
                "tokens": len(out.token_ids),
                "finish_reason": out.finish_reason,
                "plan": plan,
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
            "reference_information": q['reference_information'],
            "completions": completions,
            "n_completions": len(completions),
        }

        out_path = os.path.join(args.output_dir, f"{idx}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
