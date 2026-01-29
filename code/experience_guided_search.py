#!/usr/bin/env python3

import json
import argparse
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

# Import vLLM
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

EXPERIENCE_GUIDED_SYSTEM_PROMPT = """You are an advanced mathematical solver augmented with **Experience Bank **. 
You are currently in a **Test-Time Scaling** loop. Previous attempts on this specific problem have been analyzed to extract useful "Propositions" (Intermediate Results) and "Critical Pitfalls" (Past Errors).
 
Your goal is to solve the problem by starting from the definitions of the problem. Use previous memories strictly as a **navigational aid**. 
 
 
**Operational Guidelines:**
 
1.  **Accelerate via Verified Propositions (The Anchor):**
    - **Rule:** Treat Propositions as *structural hypotheses*, not proven facts.
    - **Priority:** Prioritize propositions that offer **abstract insights**, **simplifications**, or **identities** (e.g., algebraic simplifications, geometric invariants, combinatorial symmetries).
    - **Skepticism:** Be extremely skeptical of **raw numerical propositions** or unverified final answers. NEVER use a specific number from the report unless you have independently derived the logic that produces it.
    - **Action:** If a proposition offers a shortcut, verify its *premise* instantly. If the premise holds and aligns with your logic, use it to accelerate. If it contradicts your intuition or derivation, **discard it immediately**.
 
2.  **Navigate via Critical Pitfalls:**
    - The provided "Critical Pitfalls" describe specific logical errors or dead-ends encountered in previous failures.
    - **You are STRICTLY FORBIDDEN** from repeating the Critical Pitfalls.
    - If you approach a decision point mentioned in a pitfall, you MUST actively choose an alternative strategy/path.
    
3.  **Conflict Resolution & Robustness:**
    - **Scenario:** You encounter a contradiction (e.g., deriving two conflicting values for the same variable from different constraints).
    - **Constraint:** Do NOT simply choose the "easier" or "more common" value.
    - **Action:** A contradiction usually means a **foundational assumption** (e.g., geometric configuration, variable definition) is incorrect. **Backtrack to the very beginning**, re-read the problem statement, and challenge your initial setup.
 
 
**Context from Previous Attempts:**
{experience_context}
 
**Instruction:**
Reason step by step. Consult the Experience Bank critically: Avoiding the previous error with pitfalls, and use propositions only if they accelerate your work. Put your final answer within \\boxed{{}}.
"""

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

def load_and_aggregate_raw_experiences(
    experience_dir: Path, 
    original_idx: int,
    n_completions_to_use: int
) -> Optional[Dict[str, List[str]]]:
    """
    Load N raw experiences and aggregate their content (Propositions & Pitfalls).
    """
    experience_file = experience_dir / f"{original_idx}.jsonl"
    
    if not experience_file.exists():
        return None

    # Load all raw experiences
    raw_data = load_jsonl(str(experience_file))
    if not raw_data:
        return None
    
    # Slice the top N
    selected_data = raw_data[:n_completions_to_use]
    
    aggregated_pitfalls = set()
    aggregated_propositions = set()
    
    for record in selected_data:
        # Extract the JSON content (parsed or raw string)
        content = record.get('experience_parsed') or record.get('experience_raw')
        
        content_dict = None
        if isinstance(content, dict):
            content_dict = content
        elif isinstance(content, str):
            try:
                # Try to clean markdown code blocks if present
                clean_content = content.replace("```json", "").replace("```", "").strip()
                content_dict = json.loads(clean_content)
            except json.JSONDecodeError:
                continue
        
        if not content_dict:
            continue
            
        # Collect Pitfalls
        if 'critical_pitfalls' in content_dict:
            for p in content_dict['critical_pitfalls']:
                if isinstance(p, str):
                    aggregated_pitfalls.add(p)
                    
        # Collect Propositions
        if 'verified_propositions' in content_dict:
            for p in content_dict['verified_propositions']:
                if isinstance(p, str):
                    aggregated_propositions.add(p)

    return {
        "critical_pitfalls": sorted(list(aggregated_pitfalls)),
        "verified_propositions": sorted(list(aggregated_propositions))
    }

def construct_experience_context(experience_data: Dict[str, Any]) -> str:
    """
    Format the Aggregated experience data into a string for the prompt.
    """
    if not experience_data:
        return "No prior insights available."

    context_parts = []
    
    # 1. Critical Pitfalls
    pitfalls = experience_data.get("critical_pitfalls", [])
    if pitfalls:
        context_parts.append("### Critical Pitfalls (STRICTLY AVOID):\n" + "\n".join([f"- {p}" for p in pitfalls]))

    # 2. Verified Propositions
    facts = experience_data.get("verified_propositions", [])
    if facts:
        context_parts.append("### Propositions (Verify before use):\n" + "\n".join([f"- {f}" for f in facts]))

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
            # Basic validation
            valid = []
            for c in data.get('completions', []):
                if (c.get('text') or c.get('reasoning_content')):
                    valid.append(c)
            return valid
    except:
        return []

def save_result(
    output_dir: Path, 
    original_idx: int, 
    question_data: Dict[str, Any], 
    new_completions: List[Dict[str, Any]]
):
    """Save results to JSON file."""
    output_file = output_dir / f"{original_idx}.json"
    
    # Load existing to append
    existing_completions = load_existing_output(output_file)
    all_completions = existing_completions + new_completions
    
    result = {
        'index': original_idx,
        'question_id': question_data.get('question_id', f'q_{original_idx}'),
        'question': question_data.get('question', ''),
        'answer': question_data.get('answer', ''),
        'completions': all_completions,
        'n_completions': len(all_completions)
    }
    
    # Preserve other metadata
    for k, v in question_data.items():
        if k not in result:
            result[k] = v
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ==========================================
# Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Step 2: Rollout with RAW experience (Offline)')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file (questions)')
    parser.add_argument('--experience-dir', type=str, required=True, help='Directory containing RAW experience JSONL files')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    # NEW PARAMETER: Control how many raw experiences to aggregate
    parser.add_argument('--n-experience-completions', type=int, default=5, 
                        help='Number of raw experiences to aggregate per question')

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

    # Setup directories
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    experience_path = Path(args.experience_dir)

    # Load Questions
    logger.info(f"Loading questions from {args.input}")
    questions = load_jsonl(args.input)
    
    # Determine range
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions = questions[args.start_idx:end_idx]
    
    logger.info(f"Processing range: {args.start_idx} to {end_idx} (Total {len(questions)})")
    
    # Filter pending questions
    pending_items = [] # List of (original_idx, question_item)
    
    for i, item in enumerate(questions):
        original_idx = args.start_idx + i
        item['index'] = original_idx # Ensure index is set
        
        output_file = output_path / f"{original_idx}.json"
        
        # Check if already done enough completions
        existing = load_existing_output(output_file)
        if len(existing) >= args.n_completions:
            continue
            
        pending_items.append(item)

    if not pending_items:
        logger.info("All questions completed!")
        return

    logger.info(f"Pending questions: {len(pending_items)}")

    # Initialize vLLM
    logger.info(f"Initializing vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
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

    # Batch Processing
    total_batches = (len(pending_items) + args.batch_size - 1) // args.batch_size
    
    for i in range(total_batches):
        batch_items = pending_items[i * args.batch_size : (i + 1) * args.batch_size]
        logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch_items)} questions)...")
        
        prompts = []
        batch_metadata = [] # Stores (original_idx, item) for saving later
        
        for item in batch_items:
            original_idx = item['index']
            question_text = item['question']
            
            # 1. Load and Dynamically Aggregate Raw experiences
            experience_data = load_and_aggregate_raw_experiences(
                experience_path, 
                original_idx, 
                args.n_experience_completions
            )
            
            # 2. Construct experience Context String
            if experience_data:
                experience_context_str = construct_experience_context(experience_data)
                system_content = EXPERIENCE_GUIDED_SYSTEM_PROMPT.format(
                    experience_context=experience_context_str
                )
            else:
                # Fallback if no experience found
                raise ValueError
                # system_content = "Please reason step by step, and put your final answer within \\boxed{}."
            
            # 3. Build Full Prompt
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
            
        # 4. Generate
        if not prompts:
            continue
            
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        # 5. Process and Save
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
            
            # Save immediately
            save_result(output_path, original_idx, q_item, new_completions)

    logger.info(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()