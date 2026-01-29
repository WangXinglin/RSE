#!/usr/bin/env python3

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging
from collections import defaultdict

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install vLLM: pip install vllm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def format_prompt(question: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = DEFAULT_MATH_PROMPT
    return f"{system_prompt}\n\n{question}"

def extract_thinking(text: str) -> Tuple[str, str]:
    end_tag = "</think>"
    
    if end_tag in text:
        parts = text.split(end_tag, 1)
        
        reasoning_content = parts[0].strip()
        reasoning_content = reasoning_content.replace("<think>", "").strip()
        
        final_text = parts[1].strip() if len(parts) > 1 else ""
        
        return final_text, reasoning_content
    else:
        return text.strip(), ""

def load_existing_output(output_file: str) -> Dict[str, Any]:
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_completions = []
        total_completions = data.get('completions', [])
        for completion in total_completions:
            text = completion.get('text', '')
            reasoning = completion.get('reasoning_content', '')
            if (text.strip() or reasoning.strip()) and "API call failed" not in text:
                valid_completions.append(completion)
        return {
            'data': data,
            'total_completions': len(total_completions),
            'valid_completions': len(valid_completions),
            'valid_completion_list': valid_completions
        }
    except Exception:
        return {
            'data': None,
            'total_completions': 0,
            'valid_completions': 0,
            'valid_completion_list': []
        }

def save_completed_questions(
    questions_map: Dict[int, Dict[str, Any]], 
    completion_results: Dict[int, List[Dict[str, Any]]],
    output_dir: str
) -> List[int]:
    output_path = Path(output_dir)
    saved_questions = []
    
    for original_idx, new_completions in completion_results.items():
        if not new_completions:
            continue
            
        item = questions_map.get(original_idx)
        if not item:
            continue

        question_id = item.get('question_id', item.get('id', f'q_{original_idx}'))
        output_file = output_path / f"{original_idx}.json"
        
        existing_valid_completions = []
        if output_file.exists():
            existing_data = load_existing_output(str(output_file))
            existing_valid_completions = existing_data['valid_completion_list']
        
        all_completions = existing_valid_completions + new_completions
        
        result = {
            'index': original_idx,
            'question_id': question_id,
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'completions': all_completions,
            'n_completions': len(all_completions),
        }
        
        for key, value in item.items():
            if key not in ['question_id', 'id', 'question', 'answer']:
                result[key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        saved_questions.append(original_idx)
        logger.info(f"Saved question {original_idx}: Total {len(all_completions)} (New: {len(new_completions)})")
    
    return saved_questions

def analyze_completion_status(
    data: List[Dict[str, Any]],
    output_dir: str,
    n_completions: int,
    start_idx: int = 0
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[int, int]]:
    output_path = Path(output_dir)
    pending_questions = []
    completion_needed = {}
    
    for idx, item in enumerate(data):
        original_idx = start_idx + idx
        output_file = output_path / f"{original_idx}.json"
        
        needed = n_completions
        if output_file.exists():
            result = load_existing_output(str(output_file))
            valid_count = result['valid_completions']
            if valid_count >= n_completions:
                continue 
            needed = n_completions - valid_count
            logger.info(f"Question {original_idx} needs: {needed} (Existing: {valid_count})")
        
        completion_needed[original_idx] = needed
        pending_questions.append((original_idx, item))
    
    return pending_questions, completion_needed

def batch_inference(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int = 64,
    batch_size: int = 8,
    tensor_parallel_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    system_prompt: str = None,
    start_idx: int = 0,
    end_idx: int = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data: {input_file}")
    data = load_jsonl(input_file)
    
    if end_idx is None:
        end_idx = len(data)
    
    current_data_slice = data[start_idx:end_idx]
    
    print(f"Analyzing existing completions status...")
    pending_questions, completion_needed = analyze_completion_status(
        current_data_slice, output_dir, n_completions, start_idx
    )
    
    if not pending_questions:
        print("All questions completed!")
        return

    questions_map = {idx: item for idx, item in pending_questions}
    
    print(f"Initializing vLLM engine (TP={tensor_parallel_size})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    print(f"Starting inference: {len(pending_questions)} questions pending")

    total_chunks = (len(pending_questions) + batch_size - 1) // batch_size
    
    for i in range(total_chunks):
        chunk_start = i * batch_size
        chunk_end = min(chunk_start + batch_size, len(pending_questions))
        current_chunk = pending_questions[chunk_start:chunk_end]
        
        print(f"Processing batch {i+1}/{total_chunks} (Questions: {len(current_chunk)})...")
        
        prompts = []
        prompt_metadata = []
        
        for original_idx, item in current_chunk:
            question = item.get('question', '')
            messages = [
                {"role": "system", "content": DEFAULT_MATH_PROMPT if system_prompt is None else system_prompt},
                {"role": "user", "content": question}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            needed = completion_needed.get(original_idx, 0)
            for _ in range(needed):
                prompts.append(full_prompt)
                prompt_metadata.append(original_idx)
        
        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        batch_results = defaultdict(list)
        
        for output, q_idx in zip(outputs, prompt_metadata):
            generated_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            
            final_text, reasoning_content = extract_thinking(generated_text)
            
            result = {
                'text': final_text,
                'reasoning_content': reasoning_content,
                'tokens': len(output.outputs[0].token_ids),
                'finish_reason': finish_reason
            }
            batch_results[q_idx].append(result)
        
        save_completed_questions(questions_map, batch_results, output_dir)
        
        del outputs
        del batch_results

    print(f"\nInference completed! Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='vLLM Offline Batch Inference')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model path')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--n-completions', '-n', type=int, default=64)
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Save every N questions')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--temperature', '-t', type=float, default=0.7)
    parser.add_argument('--top-p', '-p', type=float, default=0.95)
    parser.add_argument('--top-k', '-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)
    
    args = parser.parse_args()
    
    batch_inference(
        model_name=args.model,
        input_file=args.input,
        output_dir=args.output,
        n_completions=args.n_completions,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

if __name__ == '__main__':
    main()
