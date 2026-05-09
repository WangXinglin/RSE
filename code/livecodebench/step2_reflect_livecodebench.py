#!/usr/bin/env python3
"""
Offline batch reflection generation with vLLM - Robust Version

Features:
1. Offline inference using vLLM
2. Robust generation: generate N samples per call, take first valid JSON
3. Auto-retry: if all N parse failures, retry in next round (max 5)
4. Checkpoint/resume: auto-skip already processed questions
"""

import json
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install vLLM: pip install vllm")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# ==========================================
# Prompt Template
# ==========================================

REFLECTION_SYSTEM_PROMPT = """You are a Strategic Code Reasoning Distiller. Your goal is to construct a "Memory Bank" that will serve as the foundation for the student's next problem-solving iteration by extracting two specific lists:
1.  **Verified Propositions:** Reliable facts and intermediate conclusions derived correctly from the algorithmic reasoning or implementation attempt.
2.  **Critical Pitfalls:** Wrong assumptions, invalid transitions, unsafe operations, implementation bugs, and dead ends to avoid.
The student will explicitly reference this data:
- Utilizing **Verified Propositions** as established anchors to accelerate valid reasoning and implementation
- Consulting **Critical Pitfalls** to proactively avoid repeating previously identified errors, code-level bugs, logic gaps, or dead ends.

**Constraint: strict_neutrality**
You have **NO access** to the golden answer. You have **NO access** to any external feedback such as code execution, test-case results, hidden tests, or judge outcomes. You must **NOT** make any assumptions about whether the student's final code or conclusion is correct or incorrect based on imagined runtime behavior. Treat the student's work as an unverified hypothesis; verify the validity of each step strictly based on algorithmic logic, code semantics, the stated problem constraints, and the internal consistency of the attempt alone.

## Task 1: verified_propositions (List[str])

**Goal:** Extract *only* logically sound, reusable facts from the attempt (Truth Anchors).

**Strict Inclusion Rules (Filter Aggressively):**
1.  **Independent Verification:** You must be able to independently verify that the statement is valid based on the algorithm, data structure properties, control flow, complexity reasoning, boundary conditions, or strictly derived from previous valid steps.
2.  **Explicit Conditions:** Every proposition MUST state its necessary conditions (e.g., "If the array is sorted, then...", "For 0 <= i < n, ...", "Assuming all edge weights are non-negative, ..."). Do not assume global constraints apply unless stated.
3.  **Atomicity:** Break complex thoughts into the smallest reusable units.
4.  **No "Lucky Guesses":** Do not include conclusions that are merely plausible, based on a few examples, or asserted without a clear derivation in the text.
5.  **Self-Contained:** The string must be understandable without reading the original student text. Replace vague references like "this", "it", or "the loop" with explicit variables, arrays, states, transitions, or operations.

**Content to Extract:**
*   **Valid Intermediate Algorithmic Conclusions:** Concrete properties derived accurately from previous steps and useful for implementation (e.g., "After sorting nums in nondecreasing order, equal values become adjacent", "The DP state dp[i] represents the optimal value for prefix [0..i]", "The recursion terminates because the index strictly decreases in each call").
*   **Correct Equivalences / Reformulations:** Correctly transformed problem statements, invariants, recurrence relations, greedy conditions, or state definitions (e.g., "Maximizing score is equivalent to minimizing total penalty under the given transformation", "The problem can be reduced to checking whether a path exists in the constructed graph").
*   **Constraint-Driven Deductions:** Deductions regarding valid ranges, monotonicity, uniqueness, feasibility, pruning conditions, or boundary behavior (e.g., "Since indices must remain in bounds, i must satisfy 0 <= i < n", "Because the graph is a tree, there is exactly one simple path between any two nodes").
*   **Correct Application of Standard Techniques:** Standard algorithmic facts or implementation principles used where all conditions are visibly met (e.g., "Binary search is applicable because the feasibility predicate is monotonic", "Dijkstra's algorithm is valid because all edge weights are non-negative").
*   **Implementation-Relevant Invariants:** Facts that help preserve correctness during coding, iteration, recursion, or state updates.

**Format:**
*   `"<Complete Statement with Conditions>. (Source: <Derivation/Method>)"`

## Task 2: critical_pitfalls (List[str])

**Goal:** Identify "Negative Constraints" that serve as warning signs for future explorations.

**Focus on identifying these specific categories:**
1.  **Dead Ends (Strategy Failures):** Approaches that are technically possible but lead to unmanageable complexity, excessive casework, fragile implementation, or a solution path unlikely to satisfy the stated constraints.
2.  **Fatal Logic Flaws (Actual Errors):** Fundamental errors that invalidate the attempt, such as incorrect state definitions, wrong transitions, invalid greedy assumptions, broken invariants, or reasoning that does not preserve correctness.
3.  **Potential Risks (Unsafe Operations):** Correct-looking steps that lack necessary checks, such as out-of-bounds access, division by zero, incorrect initialization, integer overflow risk, recursion depth issues, misuse of library behavior, or applying an algorithm without verifying its preconditions. Only include risks that are directly supported by the attempt, the stated constraints, or the explicit algorithmic structure.
4.  **Missing Proof Obligations:** Leaps in logic where an important case, invariant, edge case, complexity condition, or correctness condition was ignored.

**Format:**
*   `"<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"`

**Explanation Requirements (The "WHY"):**
*   **Trigger:** What specific code structure, assumption, constraint pattern, or input pattern caused the issue?
*   **Invalid Action:** What did the student fail to check, or do incorrectly?
*   **Consequence:** What is the algorithmic or implementation result? (e.g., "Out-of-bounds access," "Incorrect answer on duplicate values," "State transition becomes invalid," "Time complexity exceeds the stated constraints," "Infinite recursion or non-termination," "Leaves certain edge cases unaddressed").

**Example:**
*   `"Using binary search on the answer -> Potential Risk -> Trigger: binary search requires a monotonic feasibility predicate; Invalid Action: Failed to verify monotonicity of the check function; Consequence: Search logic is not justified and may return an incorrect result."`

## Output Requirements

*   **Output ONLY a raw JSON object.**
*   No Markdown formatting (no ```json ... ```), no explanations, no chat.

**JSON Structure:**

{
    "verified_propositions": [
        "<Complete Statement with Conditions>. (Source: <Derivation/Method>)",
        "..."
    ],
    "critical_pitfalls": [
        "<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>",
        "..."
    ]
}

## Input Data

**Question:**
{{question}}

**Student's Attempt:**
{{attempt}}
"""

def load_jsonl(file_path):
    data_points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data_points.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data_points

def save_to_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def format_prompt(question: str, starter_code: str, attempt: str) -> str:
    """Apply the new Prompt template"""
    if starter_code:
        question = f"{question}\n\nStarter Code:\n```python\n{starter_code}\n```"
    return REFLECTION_SYSTEM_PROMPT.replace("{{question}}", question).replace("{{attempt}}", attempt)

def extract_and_validate_json(text: str) -> Optional[str]:
    """Attempt to extract and validate JSON from text. Returns cleaned JSON string if valid, else None."""
    cleaned_text = text

    if " response" in text:
        cleaned_text = text.split(" response")[-1].strip()

    json_match = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
    else:
        start = cleaned_text.find('{')
        end = cleaned_text.rfind('}')
        if start != -1 and end != -1:
            candidate = cleaned_text[start:end+1]
        else:
            candidate = cleaned_text

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None

def main():
    parser = argparse.ArgumentParser(description='vLLM offline Reflection generation (Robust)')
    parser.add_argument('--model', type=str, required=True, help='Reflection model path')
    parser.add_argument('--question-file', type=str, required=True, help='Original question file (jsonl)')
    parser.add_argument('--answer-dir', type=str, required=True, help='Directory containing rollouts')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')

    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size (tasks per batch)')

    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--n-samples', type=int, default=3, help='Samples per inference (for fault tolerance)')

    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)
    parser.add_argument('--answer-file-prefix', type=str, default="", help='Prefix for answer files')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading question file: {args.question_file}")
    questions = load_jsonl(args.question_file)

    for i, q in enumerate(questions):
        if 'index' not in q: q['index'] = i

    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx : end_idx]
    logger.info(f"Processing range: {args.start_idx} - {end_idx} (Total {len(questions)})")

    all_tasks = []

    logger.info("Scanning and preparing tasks...")

    for q_item in tqdm(questions):
        original_idx = q_item['index']
        q_id = q_item.get('question_id', q_item.get('id', ''))

        output_file = Path(args.output_dir) / f"{original_idx}.jsonl"
        if output_file.exists():
            continue

        answer_path = Path(args.answer_dir) / f"{args.answer_file_prefix}{original_idx}.json"
        if not answer_path.exists():
            continue

        try:
            with open(answer_path, 'r') as f:
                answer_data = json.load(f)
        except Exception:
            continue

        if answer_data.get('question_id') != q_id:
            continue

        completions = answer_data.get('completions', [])
        if not completions:
            continue

        for c_idx, completion in enumerate(completions):
            attempt_text = ""
            if completion.get("reasoning_content"):
                attempt_text += completion.get("reasoning_content", "") + "\n\n"
            if completion.get("text"):
                attempt_text += completion.get("text", "")

            if not attempt_text.strip():
                continue

            question_text = q_item.get('question_content', '')
            starter_code = q_item.get('starter_code', '')
            prompt = format_prompt(question_text, starter_code, attempt_text)

            all_tasks.append({
                'original_idx': original_idx,
                'question_id': q_id,
                'rollout_idx': c_idx,
                'prompt': prompt,
                'retries': 0
            })

    if not all_tasks:
        logger.info("No new tasks to process.")
        return

    logger.info(f"Prepared {len(all_tasks)} initial tasks")

    logger.info(f"Initializing vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=100000,
        enforce_eager=False
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    pending_tasks = all_tasks
    max_retries = 5

    final_results = defaultdict(list)

    current_round_tasks = pending_tasks

    for round_idx in range(max_retries + 1):
        if not current_round_tasks:
            break

        next_round_tasks = []

        num_batches = (len(current_round_tasks) + args.batch_size - 1) // args.batch_size

        logger.info(f"=== Round {round_idx} (Tasks: {len(current_round_tasks)}, Num_Batch: {num_batches}) ===")

        for i in range(num_batches):
            batch_tasks = current_round_tasks[i * args.batch_size : (i + 1) * args.batch_size]

            batch_prompts = []
            for t in batch_tasks:
                messages = [{"role": "user", "content": t['prompt']}]
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_prompts.append(full_prompt)

            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True)

            for j, output in enumerate(outputs):
                task = batch_tasks[j]

                valid_content = None
                valid_raw = None

                for sample in output.outputs:
                    raw_text = sample.text
                    parsed_json = extract_and_validate_json(raw_text)
                    if parsed_json:
                        valid_content = parsed_json
                        valid_raw = raw_text
                        break

                if valid_content:
                    result_entry = {
                        "question_id": task['question_id'],
                        "rollout_idx": task['rollout_idx'],
                        "reflection_raw": valid_raw,
                        "reflection_parsed": valid_content
                    }
                    final_results[task['original_idx']].append(result_entry)
                else:
                    if task['retries'] < max_retries:
                        task['retries'] += 1
                        next_round_tasks.append(task)
                    else:
                        logger.warning(f"Task failed after {max_retries} retries: QID {task['question_id']} Rollout {task['rollout_idx']}")

        current_round_tasks = next_round_tasks

    logger.info("Saving results...")
    for q_idx, results in final_results.items():
        results.sort(key=lambda x: x['rollout_idx'])
        output_file = Path(args.output_dir) / f"{q_idx}.jsonl"
        save_to_jsonl(output_file, results)

    logger.info(f"All done! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
