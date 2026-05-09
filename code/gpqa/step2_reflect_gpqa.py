#!/usr/bin/env python3
"""
Offline batch reflection generation with vLLM - Robust Version
GPQA-Diamond adapted

Features:
1. Offline inference using vLLM
2. Robust generation: generate N samples per call, take first valid JSON
3. Auto-retry: if all N parse failures, retry in next round (max 5)
4. Checkpoint/resume: auto-skip already processed questions
5. Adapted for GPQA-Diamond multiple-choice format, aligned with step1 interface
"""

import json
import argparse
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

# =========================================================
# GPQA Question Formatting (consistent with step1)
# =========================================================

GPQA_PROMPT_TEMPLATE = """What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}
Please show your choice in the `answer` field with only the choice letter, e.g., {{"answer": "C"}}."""

# =========================================================
# Reflection Prompt
# =========================================================

REFLECTION_SYSTEM_PROMPT = """You are a Strategic Scientific Reasoning Distiller. Your goal is to construct a "Memory Bank" that will serve as the foundation for the student's next problem-solving iteration by extracting two specific lists:
1.  **Verified Propositions:** Reliable scientific facts, intermediate conclusions, eliminations, and reasoning steps derived correctly from the attempt.
2.  **Critical Pitfalls:** Wrong assumptions, invalid eliminations, logical fallacies, unsupported scientific claims, and dead ends to avoid.
The student will explicitly reference this data:
- Utilizing **Verified Propositions** as established anchors to accelerate valid reasoning
- Consulting **Critical Pitfalls** to proactively avoid repeating previously identified errors, logic gaps, or dead ends

**Constraint: strict_neutrality**
You have **NO access** to the golden answer. You have **NO access** to any external feedback such as answer checking, execution, hidden labels, or evaluation outcomes. You must **NOT** make any assumptions about whether the student's final choice is correct or incorrect. Treat the student's work as an unverified hypothesis; verify the validity of each step strictly based on scientific reasoning, the stated question and options, and the internal consistency of the attempt alone.

## Task 1: verified_propositions (List[str])

**Goal:** Extract *only* logically sound, reusable facts from the attempt (Truth Anchors).

**Strict Inclusion Rules (Filter Aggressively):**
1.  **Independent Verification:** You must be able to independently verify that the statement is valid based on scientific knowledge, option semantics, definitional facts, internal reasoning consistency, or strictly derived from previous valid steps.
2.  **Explicit Conditions:** Every proposition MUST state its necessary conditions when relevant (e.g., "If the molecule is aromatic, then...", "Assuming the process occurs in equilibrium, ...", "Under classical mechanics, ..."). Do not assume hidden conditions.
3.  **Atomicity:** Break complex thoughts into the smallest reusable units.
4.  **No "Lucky Guesses":** Do not include conclusions that are merely plausible, pattern-matched, or based on vague intuition without a clear derivation in the text.
5.  **Self-Contained:** The string must be understandable without reading the original student text. Replace vague references like "this", "it", or "that option" with explicit scientific statements or option labels.

**Content to Extract:**
*   **Valid Intermediate Scientific Conclusions:** Concrete facts or deductions derived accurately from the attempt (e.g., "Option B is inconsistent with conservation of energy under the stated setup", "The described mechanism requires a leaving group, which is absent in option C").
*   **Correct Eliminations:** Scientifically justified reasons for ruling out one or more answer choices.
*   **Correct Equivalences / Reformulations:** Correct reinterpretations of the question, scientific definitions, domain assumptions, or comparisons between options.
*   **Constraint-Driven Deductions:** Deductions regarding units, directionality, causality, feasibility, physical plausibility, biochemical compatibility, chemical stability, or domain-specific constraints.
*   **Correct Application of Standard Scientific Principles:** Standard laws, definitions, or scientific regularities used where their conditions are visibly met.

**Format:**
*   `"<Complete Statement with Conditions>. (Source: <Derivation/Method>)"`

## Task 2: critical_pitfalls (List[str])

**Goal:** Identify "Negative Constraints" that serve as warning signs for future explorations.

**Focus on identifying these specific categories:**
1.  **Dead Ends (Strategy Failures):** Approaches that are technically possible but unproductive, overly speculative, or based on irrelevant detail rather than discriminating between answer choices.
2.  **Fatal Logic Flaws (Actual Errors):** Fundamental errors that invalidate the attempt, such as misreading the question, confusing scientific concepts, invalid option elimination, or contradiction with known principles.
3.  **Potential Risks (Unsafe Operations):** Correct-looking steps that lack necessary justification, such as eliminating an option without checking all conditions, assuming a mechanism/property without evidence, or treating a heuristic as proof.
4.  **Missing Proof Obligations:** Leaps in logic where an important comparison between choices, scientific precondition, unit check, or causal justification was ignored.

**Format:**
*   `"<Context/Step> -> <Type: Dead End / Fatal Flaw / Potential Risk> -> <Explanation: Trigger + Invalid Action + Consequence>"`

**Explanation Requirements (The "WHY"):**
*   **Trigger:** What specific concept, option comparison, scientific assumption, or wording in the attempt caused the issue?
*   **Invalid Action:** What did the student fail to check, or do incorrectly?
*   **Consequence:** What is the reasoning result? (e.g., "Eliminates the correct option without justification," "Keeps an incompatible option alive," "Confuses correlation with mechanism," "Uses a scientific principle outside its valid regime," "Leaves key option distinctions unresolved").

**Example:**
*   `"Ruling out option B because it 'seems too strong' -> Potential Risk -> Trigger: qualitative impression only; Invalid Action: eliminated an option without checking whether the underlying scientific claim is actually false under the stated conditions; Consequence: valid options may be discarded for non-scientific reasons."`

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


def format_gpqa_question(q_item: Dict[str, Any]) -> str:
    """Build multiple-choice question text from GPQA raw fields, consistent with step1."""
    question_text = q_item.get('Question', '')
    correct_answer = q_item.get('Correct Answer', '')
    incorrect_1 = q_item.get('Incorrect Answer 1', '')
    incorrect_2 = q_item.get('Incorrect Answer 2', '')
    incorrect_3 = q_item.get('Incorrect Answer 3', '')

    return GPQA_PROMPT_TEMPLATE.format(
        question=question_text,
        A=correct_answer,
        B=incorrect_1,
        C=incorrect_2,
        D=incorrect_3
    )


def format_prompt(question: str, attempt: str) -> str:
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
    parser = argparse.ArgumentParser(description='vLLM offline Reflection generation (GPQA adapted)')
    parser.add_argument('--model', type=str, required=True, help='Reflection model path')
    parser.add_argument('--question-file', type=str, required=True, help='Original question file (GPQA jsonl)')
    parser.add_argument('--answer-dir', type=str, required=True, help='Directory containing step1 rollouts')
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
        if 'index' not in q:
            q['index'] = i

    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    logger.info(f"Processing range: {args.start_idx} - {end_idx} (Total {len(questions)})")

    all_tasks = []

    logger.info("Scanning and preparing tasks...")

    for q_item in tqdm(questions):
        original_idx = q_item['index']
        q_id = q_item.get('Record ID', q_item.get('question_id', q_item.get('id', '')))

        output_file = Path(args.output_dir) / f"{original_idx}.jsonl"
        if output_file.exists():
            continue

        answer_path = Path(args.answer_dir) / f"{args.answer_file_prefix}{original_idx}.json"
        if not answer_path.exists():
            continue

        try:
            with open(answer_path, 'r', encoding='utf-8') as f:
                answer_data = json.load(f)
        except Exception:
            continue

        if answer_data.get('question_id') != q_id:
            continue

        completions = answer_data.get('completions', [])
        if not completions:
            continue

        question_text = format_gpqa_question(q_item)

        for c_idx, completion in enumerate(completions):
            attempt_text = ""
            if completion.get("reasoning_content"):
                attempt_text += completion.get("reasoning_content", "") + "\n\n"
            if completion.get("text"):
                attempt_text += completion.get("text", "")

            if not attempt_text.strip():
                continue

            prompt = format_prompt(question_text, attempt_text)

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
            batch_tasks = current_round_tasks[i * args.batch_size:(i + 1) * args.batch_size]

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
                        logger.warning(
                            f"Task failed after {max_retries} retries: "
                            f"QID {task['question_id']} Rollout {task['rollout_idx']}"
                        )

        current_round_tasks = next_round_tasks

    logger.info("Saving results...")
    for q_idx, results in final_results.items():
        results.sort(key=lambda x: x['rollout_idx'])
        output_file = Path(args.output_dir) / f"{q_idx}.jsonl"
        save_to_jsonl(output_file, results)

    logger.info(f"All done! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
