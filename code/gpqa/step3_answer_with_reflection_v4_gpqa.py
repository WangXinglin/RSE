#!/usr/bin/env python3
"""
Step 3: Rollout with RAW Reflection (Offline vLLM Version) - GPQA adapted

Features:
- Offline inference using vLLM.
- Dynamic Aggregation: Loads N raw reflections per question and aggregates them on-the-fly.
- Uses the reflection-guided prompt to guide the model.
- Resumable and efficient batching.
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

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
# Prompt Templates
# ==========================================

GPQA_PROMPT_TEMPLATE = """What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}
Please show your choice in the `answer` field with only the choice letter, e.g., {{"answer": "C"}}."""

REFLECTION_GUIDED_SYSTEM_PROMPT = """You are an advanced scientific reasoning solver.
You have access to a "Reference Report" from previous attempts, containing "Propositions" (Intermediate Results) and "Critical Pitfalls" (Past Errors).

**Core Directive: First-Principles Reasoning with Strategic Consultation**
Your primary goal is to determine the correct answer starting from the scientific content of the question, the semantic differences between the answer choices, and the underlying domain principles. Use the provided context strictly as a **navigational aid**, not as a definitive source of truth.

**Operational Guidelines:**

1.  **Proposition Handling (Structure > Surface Form):**
    - **Rule:** Treat Propositions as *structural hypotheses*, not proven facts.
    - **Priority:** Prioritize propositions that offer **scientific insights**, **correct eliminations**, **causal or mechanistic distinctions**, **constraint-driven simplifications**, or **correct reformulations of the question**.
    - **Skepticism:** Be extremely skeptical of **surface-pattern matches**, **example-specific conclusions**, **unsupported scientific claims**, or **assertions that a particular option must be correct without justification**. NEVER trust a proposed final choice unless you have independently justified why it is supported by the question and why competing options fail.
    - **Action:** If a proposition offers a shortcut, verify its *premise* immediately. If the premise holds and aligns with your reasoning, use it to accelerate. If it contradicts the wording of the question, the scientific constraints, or the option semantics, **discard it immediately**.

2.  **Pitfall Discrimination (Mechanism > Appearance):**
    - **Insight:** Pitfalls often describe a *misread condition*, a *false elimination*, a *confused scientific concept*, or an *unsupported leap from intuition to conclusion*. Do not confuse a "nontrivial but valid scientific inference" with a "logic error".
    - **Action:** When your reasoning resembles a Pitfall:
        - *Check:* Are you actually committing the specific scientific or logical error described (e.g., confusing correlation with mechanism, applying a principle outside its regime, eliminating an option without checking all relevant conditions)?
        - *Or:* Are you performing a valid step that merely *looks* similar to the pitfall?
    - **Protocol:** If it is a genuine flaw, **ABORT** the branch. If it is a valid operation, **PROCEED** but explicitly verify why your reasoning is sound.

3.  **Conflict Resolution & Robustness:**
    - **Scenario:** You encounter a contradiction (e.g., two incompatible interpretations of the question, two competing principles, or two answer choices that seem plausible for different reasons).
    - **Constraint:** Do NOT simply choose the more familiar or more intuitive option.
    - **Action:** A contradiction usually means a **foundational assumption** (e.g., what is being asked, what condition is operative, or which principle actually applies) is incorrect. **Backtrack to the question wording**, compare the options carefully, and challenge your initial setup.

4.  **Discrimination over Enumeration:**
    - **Guideline:** Before diving into vague speculation or excessive detail, pause and ask: "What is the key scientific distinction that separates the correct option from the others?"
    - **Goal:** Use the Reference Report to find these *structural discriminators* rather than complicating the problem with ad hoc or weakly supported arguments.

**Context from Previous Attempts:**
{reflection_context}

**Instruction:**
Reason step by step. Start by identifying the core scientific issue, the critical constraints or definitions, and the key distinctions among the answer choices. Consult the Reference Report critically: verify pitfalls before pruning, and use propositions only if they safely accelerate your work. Then produce your final answer as a JSON object with the format {{"answer": "A"}} where the value is a single letter from A, B, C, or D."""

# ==========================================
# Data Loading & Parsing
# ==========================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data

def format_gpqa_question(item: Dict[str, Any]) -> str:
    return GPQA_PROMPT_TEMPLATE.format(
        question=item.get("Question", ""),
        A=item.get("Correct Answer", ""),
        B=item.get("Incorrect Answer 1", ""),
        C=item.get("Incorrect Answer 2", ""),
        D=item.get("Incorrect Answer 3", ""),
    )

def load_and_aggregate_raw_reflections(
    reflection_dir: Path,
    original_idx: int,
    n_completions_to_use: int
) -> Optional[Dict[str, List[str]]]:
    """
    Load N raw reflections and aggregate their content (Propositions & Pitfalls).
    """
    reflection_file = reflection_dir / f"{original_idx}.jsonl"

    if not reflection_file.exists():
        return None

    raw_data = load_jsonl(str(reflection_file))
    if not raw_data:
        return None

    selected_data = raw_data[:n_completions_to_use]

    aggregated_pitfalls = set()
    aggregated_propositions = set()

    for record in selected_data:
        content = record.get('reflection_parsed') or record.get('reflection_raw')

        content_dict = None
        if isinstance(content, dict):
            content_dict = content
        elif isinstance(content, str):
            try:
                clean_content = content.replace("```json", "").replace("```", "").strip()
                content_dict = json.loads(clean_content)
            except json.JSONDecodeError:
                continue

        if not content_dict:
            continue

        if 'critical_pitfalls' in content_dict:
            for p in content_dict['critical_pitfalls']:
                if isinstance(p, str):
                    aggregated_pitfalls.add(p)

        if 'verified_propositions' in content_dict:
            for p in content_dict['verified_propositions']:
                if isinstance(p, str):
                    aggregated_propositions.add(p)

    return {
        "critical_pitfalls": sorted(list(aggregated_pitfalls)),
        "verified_propositions": sorted(list(aggregated_propositions))
    }

def construct_reflection_context(reflection_data: Dict[str, Any]) -> str:
    """
    Format the Aggregated Reflection data into a string for the prompt.
    """
    if not reflection_data:
        return "No prior insights available."

    context_parts = []

    pitfalls = reflection_data.get("critical_pitfalls", [])
    if pitfalls:
        context_parts.append(
            "### Critical Pitfalls (STRICTLY AVOID):\n" + "\n".join([f"- {p}" for p in pitfalls])
        )

    facts = reflection_data.get("verified_propositions", [])
    if facts:
        context_parts.append(
            "### Propositions (Verify before use):\n" + "\n".join([f"- {f}" for f in facts])
        )

    if not context_parts:
        return "No significant insights extracted."

    return "\n\n".join(context_parts)

def extract_thinking(text: str) -> Tuple[str, str]:
    """
    Extract reasoning (<think>...</think>) and final answer.
    """
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return final_text, reasoning
    else:
        return text.strip(), ""

def load_existing_output(output_file: Path) -> List[Dict[str, Any]]:
    """Load existing completions to support resume."""
    if not output_file.exists():
        return []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            valid = []
            for c in data.get('completions', []):
                if (c.get('text') or c.get('reasoning_content')):
                    valid.append(c)
            return valid
    except Exception:
        return []

def save_result(
    output_dir: Path,
    original_idx: int,
    question_data: Dict[str, Any],
    new_completions: List[Dict[str, Any]]
):
    """Save results to JSON file."""
    output_file = output_dir / f"{original_idx}.json"

    existing_completions = load_existing_output(output_file)
    all_completions = existing_completions + new_completions

    result = {
        'index': original_idx,
        'question_id': question_data.get('Record ID', question_data.get('question_id', f'q_{original_idx}')),
        'Question': question_data.get('Question', ''),
        'Correct Answer': question_data.get('Correct Answer', ''),
        'Incorrect Answer 1': question_data.get('Incorrect Answer 1', ''),
        'Incorrect Answer 2': question_data.get('Incorrect Answer 2', ''),
        'Incorrect Answer 3': question_data.get('Incorrect Answer 3', ''),
        'completions': all_completions,
        'n_completions': len(all_completions)
    }

    for k, v in question_data.items():
        if k not in result:
            result[k] = v

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ==========================================
# Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Step 3: Rollout with RAW Reflection (Offline, GPQA adapted)')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file (questions)')
    parser.add_argument('--reflection-dir', type=str, required=True, help='Directory containing RAW reflection JSONL files')
    parser.add_argument('--output', type=str, required=True, help='Output directory')

    parser.add_argument('--n-reflection-completions', type=int, default=5,
                        help='Number of raw reflections to aggregate per question')
    parser.add_argument('--n-completions', type=int, default=1, help='Number of new rollouts to generate per question')

    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size for vLLM processing (number of questions)')

    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=2048)

    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    reflection_path = Path(args.reflection_dir)

    logger.info(f"Loading questions from {args.input}")
    questions = load_jsonl(args.input)

    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]

    logger.info(f"Processing range: {args.start_idx} to {end_idx} (Total {len(questions)})")

    pending_items = []

    for i, item in enumerate(questions):
        original_idx = args.start_idx + i
        item['index'] = original_idx

        output_file = output_path / f"{original_idx}.json"
        existing = load_existing_output(output_file)
        if len(existing) >= args.n_completions:
            continue

        pending_items.append(item)

    if not pending_items:
        logger.info("All questions completed!")
        return

    logger.info(f"Pending questions: {len(pending_items)}")

    logger.info(f"Initializing vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=262144,
        enforce_eager=False
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=args.n_completions,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    total_batches = (len(pending_items) + args.batch_size - 1) // args.batch_size

    for i in range(total_batches):
        batch_items = pending_items[i * args.batch_size : (i + 1) * args.batch_size]
        logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch_items)} questions)...")

        prompts = []
        batch_metadata = []

        for item in batch_items:
            original_idx = item['index']
            question_text = format_gpqa_question(item)

            reflection_data = load_and_aggregate_raw_reflections(
                reflection_path,
                original_idx,
                args.n_reflection_completions
            )

            if reflection_data:
                reflection_context_str = construct_reflection_context(reflection_data)
                system_content = REFLECTION_GUIDED_SYSTEM_PROMPT.format(
                    reflection_context=reflection_context_str
                )
            else:
                raise ValueError(f"No reflection data found for question index {original_idx}")

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question_text}
            ]

            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts.append(full_prompt)
            batch_metadata.append((original_idx, item))

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        for output, (original_idx, q_item) in zip(outputs, batch_metadata):
            new_completions = []

            for sample_out in output.outputs:
                text_raw = sample_out.text
                final_text, reasoning = extract_thinking(text_raw)

                new_completions.append({
                    'text': final_text,
                    'reasoning_content': reasoning,
                    'tokens': len(sample_out.token_ids),
                    'finish_reason': sample_out.finish_reason
                })

            save_result(output_path, original_idx, q_item, new_completions)

    logger.info(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()