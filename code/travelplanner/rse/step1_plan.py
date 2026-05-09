"""
RSE Step 1: Initial Plan Generation for TravelPlanner
Uses vLLM to generate travel plans from queries + reference information (offline).
Loads validation data from local CSV / JSONL, no HuggingFace dependency.

Output: one JSON per query under {output_dir}/{idx}.json
Format: {
    "index": int,
    "query": str,
    "query_data": {...},
    "reference_text": str,
    "completions": [
        {"text": str, "reasoning_content": str, "tokens": int, "finish_reason": str, "plan": list|null}
    ],
    "n_completions": int
}
"""
import argparse
import csv
import json
import os
import re
import ast
import sys
from pathlib import Path

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****

You MUST output the travel plan as a JSON array. Each element is a dict with keys:
"days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation".
Use "-" for fields that are not applicable. Attractions should be separated by semicolons with a trailing semicolon.
Output ONLY the JSON array, no other text."""


def load_validation_data(data_dir: str):
    """Load validation queries from CSV and reference info from JSONL."""
    csv_path = os.path.join(data_dir, "validation.csv")
    ref_path = os.path.join(data_dir, "validation_ref_info.jsonl")

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

    ref_infos = []
    with open(ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ref_infos.append(json.loads(line))

    assert len(queries) == len(ref_infos), \
        f"Mismatch: {len(queries)} queries vs {len(ref_infos)} ref_infos"

    return queries, ref_infos


def format_reference_info(ref_info_from_csv: list) -> str:
    """Format the reference_information list from CSV into a text block."""
    parts = []
    for item in ref_info_from_csv:
        parts.append(f"{item['Description']}:\n{item['Content']}")
    return "\n\n".join(parts)


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
    parser = argparse.ArgumentParser(description="RSE Step 1: Initial Plan Generation")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--data-dir", type=str, default="<DATABASE_PATH>",
                        help="Path to database dir containing validation.csv & validation_ref_info.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-completions", type=int, default=1, help="Number of completions per query")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=38912, help="Max output tokens")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--start-idx", type=int, default=0, help="Start query index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=-1, help="End query index (exclusive, -1=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Loading validation data from {args.data_dir}...")
    queries, ref_infos = load_validation_data(args.data_dir)
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

        q = queries[idx]

        user_msg = f"Given Information: {json.dumps(q['reference_information'], ensure_ascii=False)}\nQuery: {q['query']}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
