#!/usr/bin/env python3

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


EXPERIENCE_DISTILLATION_SYSTEM_PROMPT = """"You are a Strategic Reasoning Distiller. Your goal is to construct a "Experience Bank" that will serve as the foundation for the student's next problem-solving iteration by extracting two specific lists:
1.  **Verified Propositions:** Irrefutable truths and intermediate conclusions derived correctly.
2.  **Critical Pitfalls:** Logical fallacies, dangerous operations, and dead ends to avoid.
The student will explicitly reference this data: 
- Utilizing **Verified Propositions** as established anchors to accelerate valid reasoning
- Consulting **Critical Pitfalls** to proactively avoid repeating previously identified errors, logic gaps, or dead ends.

**Constraint: strict_neutrality**
You have **NO access** to the golden answer. You must **NOT** make any assumptions about whether the student's final conclusion is correct or incorrect. Treat the student's work as an unverified hypothesis; verify the validity of each step strictly based on logic and mathematical axioms alone.

## Task 1: verified_propositions (List[str])

**Goal:** Extract *only* mathematically sound, reusable facts (Truth Anchors).

**Strict Inclusion Rules (Filter Aggressively):**
1.  **Independent Verification:** You must be able to independently verify the statement is true based on standard mathematical axioms or strictly derived from the previous valid steps.
2.  **Explicit Conditions:** Every proposition MUST state its necessary conditions (e.g., "If $x \\neq 0$, then...", "For $a > 0$, implies..."). Do not assume global constraints apply unless stated.
3.  **Atomicity:** Break complex thoughts into the smallest reusable units.
4.  **No "Lucky Guesses":** Do not include conclusions that are "likely true" or "verified by plugging in numbers" but lack logical derivation in the text.
5.  **Self-Contained:** The string must be understandable without reading the original student text. Replace pronouns like "it" or "the equation" with specific variables or expressions.

**Content to Extract:**
*   **Valid Intermediate Calculations:** Concrete results derived accurately from previous steps (e.g., "The derivative $f'(x)$ is calculated as $3x^2 - 4$", "The discriminant $\\Delta$ equals $16$", "The roots of the auxiliary equation are $x=2, x=3$").
*   **Algebraic Equivalences:** Correctly simplified or rearranged forms of equations/expressions (e.g., "Equation (1) is equivalent to $y = 2x + 1$ under the given constraints").
*   **Logical Implications & Domain Constraints:** Deductions regarding variable ranges, inequalities, or existence conditions (e.g., "Since $x$ is a length, $x > 0$", "Therefore, $a$ must be an integer").
*   **Correct Application of Theorems/Identities:** Standard mathematical definitions or theorems used where all conditions are visibly met (e.g., "Applying Pythagorean theorem: $a^2 + b^2 = c^2$").

**Format:**
*   `"<Complete Statement with Conditions>. (Source: <Derivation/Method>)"`

## Task 2: critical_pitfalls (List[str])

**Goal:** Identify "Negative Constraints" that serve as warning signs for future explorations.

**Focus on identifying these specific categories:**
1.  **Dead Ends (Strategy Failures):** Approaches that are technically valid but lead to unmanageable complexity, circular reasoning, or an unsolvable state (e.g., expanding a high-power polynomial unnecessarily).
2.  **Fatal Logic Flaws (Actual Errors):** Fundamental errors that ruin the attempt, such as non-equivalent transformations, confusing sufficient/necessary conditions, or calculation mistakes.
3.  **Potential Risks (Unsafe Operations):** Correct-looking steps that lack necessary checks (e.g., dividing by a variable that could be zero, squaring equations without checking for extraneous roots).
4.  **Missing Proof Obligations:** Leaps in logic where a case was ignored or a theorem was applied without verifying its preconditions.

**Format:**
*   `"<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"`

**Explanation Requirements (The "WHY"):**
*   **Trigger:** What specific expression or structure caused the issue?
*   **Invalid Action:** What did the student fail to check, or do incorrectly?
*   **Consequence:** What is the mathematical result? (e.g., "Loss of valid solution x=1," "Explosion of terms making solution impossible," "False conclusion derived").

**Example:**
*   `"Dividing both sides by (x-1) -> Potential Risk -> Trigger: (x-1) in denominator; Invalid Action: Failed to verify x!=1; Consequence: Zero division error and loss of potential solution."`

## Output Requirements

*   **Output ONLY a raw JSON object.**
*   No Markdown formatting (no ```json ... ```), no explanations, no chat.
*   Ensure all LaTeX backslashes are escaped properly for JSON (e.g., `\\\\frac`).

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

def format_prompt(question: str, attempt: str) -> str:
    return EXPERIENCE_DISTILLATION_SYSTEM_PROMPT.replace("{{question}}", question).replace("{{attempt}}", attempt)

def extract_and_validate_json(text: str) -> Optional[str]:
    cleaned_text = text
    
    if "</think>" in text:
        cleaned_text = text.split("</think>")[-1].strip()
    
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
    parser = argparse.ArgumentParser(description='Offline Experience Distillation with vLLM')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--question-file', type=str, required=True, help='Original question file (jsonl)')
    parser.add_argument('--answer-dir', type=str, required=True, help='Directory containing answer rollouts')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size')
    
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--n-samples', type=int, default=3, help='Number of samples per inference (for robustness)')
    
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
    logger.info(f"Processing range: {args.start_idx} - {end_idx} (Total {len(questions)} items)")

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
                
            prompt = format_prompt(q_item['question'], attempt_text)
            
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
                        "experience_raw": valid_raw,
                        "experience_parsed": valid_content
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
