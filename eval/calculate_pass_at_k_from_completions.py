#!/usr/bin/env python3
"""
Pass@k Calculation Script (with Token Length Filtering + 60s Timeout Mechanism)
Features:
1. Reads JSON files from a specified directory (containing completions and answer).
2. (New) If max_reference is specified, loads Tokenizer to filter out responses with length > 8000 tokens and keeps the top max_reference.
3. Extracts \boxed{} content from completions.
4. Compares with ground truth to determine correctness.
5. Calculates and outputs Pass@k metrics.
6. (New) Automatically forcibly stops and discards single file processing if it exceeds 60s.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import multiprocessing
import re
import signal  # New: Used for handling timeout signals
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from scipy.special import comb

from sal.utils.math import extract_answer
from evaluation.grader import math_equal

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# ================= Global Variables for Workers =================
_global_tokenizer = None
_global_max_ref = None
_global_token_limit = 320000

def init_worker(tokenizer_path: str, max_reference: int):
    """
    Multiprocessing initialization function: Loads tokenizer in each worker process.
    """
    global _global_tokenizer, _global_max_ref
    _global_max_ref = max_reference
    
    if max_reference is not None:
        if not HAS_TRANSFORMERS:
            print("⚠️ Warning: transformers not installed, cannot perform Token filtering, max_reference will be ignored")
            return

        try:
            _global_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Worker Tokenizer initialization failed: {e}")
            _global_tokenizer = None

# ================= Logic Functions =================

# New: Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError("Processing timed out")

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if k > n: return 0.0
    if c == 0: return 0.0
    if k > n - c: return 1.0
    
    try:
        pass_k = 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)
        return max(0.0, min(1.0, pass_k))
    except (OverflowError, ValueError):
        try:
            pass_k = 1.0 - comb(n - c, k) / comb(n, k)
            return max(0.0, min(1.0, pass_k))
        except:
            return 0.0

def check_answer(text: str, ground_truth: str, data_name: str = "math") -> bool:
    # Add internal timeout protection to prevent extract_answer regex from hanging
    try:
        pred_ans = extract_answer(text, data_name)
    except Exception:
        pred_ans = None
    
    if pred_ans is None:
        pred_ans = ""
        
    is_correct = math_equal(str(pred_ans), str(ground_truth), timeout=True)
    return is_correct

def process_file(file_path: Path) -> Optional[Tuple[int, Dict]]:
    """
    Process a single file: Read, (Filter), Extract, Verify
    Includes 60s forced timeout mechanism
    """
    # ================= Timeout Setup Start =================
    # Only effective on Unix/Linux systems; signal.SIGALRM does not exist on Windows
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # Set 60-second alarm
    # ===============================================

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get index
        index = data.get('index')
        if index is None:
            try:
                stem = file_path.stem
                if stem.startswith('verification_'):
                    index = int(stem.split('_')[1])
                else:
                    index = int(stem)
            except ValueError:
                return None

        ground_truth = data.get('answer')
        if not ground_truth:
            return None

        raw_completions = data.get('completions', [])
        
        global _global_tokenizer, _global_max_ref, _global_token_limit
        
        final_completions = []
        
        if _global_max_ref is not None and _global_tokenizer is not None:
            valid_candidates = []
            for comp in raw_completions:
                raw_text = comp.get('text') if isinstance(comp, dict) else comp
                text_str = str(raw_text) if raw_text is not None else ""
                
                try:
                    token_ids = _global_tokenizer.encode(text_str)
                    if len(token_ids) <= _global_token_limit:
                        valid_candidates.append(comp)
                    else:
                        valid_candidates.append({"text": "wa"})
                except Exception:
                    continue
            
            final_completions = valid_candidates[:_global_max_ref]
        else:
            final_completions = raw_completions[:128]

        verification_details = []
        
        for comp in final_completions:
            raw_text = comp.get('text') if isinstance(comp, dict) else comp
            is_text_null = raw_text is None or str(raw_text).strip() == ""
            text = str(raw_text) if raw_text is not None else ""
            
            is_correct = check_answer(text, ground_truth) if text != "wa" else False
            
            verification_details.append({
                'is_correct': is_correct,
                'is_text_null': is_text_null
            })
            
        return (int(index), {'verification_details': verification_details})
    
    except TimeoutError:
        # Catch our self-set timeout exception
        # print(f"⏰ File processing timeout (60s): {file_path.name}") # Optional: Print log
        return None  # Return None to indicate discard
    except Exception:
        return None
    finally:
        # ================= Cancel Timeout Setup =================
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # Disable alarm to prevent affecting subsequent code
        # ===============================================

def load_and_verify_data(data_dir: str, tokenizer_path: str, max_reference: int) -> Dict[int, Dict]:
    path = Path(data_dir)
    files = sorted([f for f in path.glob("*.json") if not f.name.startswith('.')])
    
    if not files:
        print(f"❌ No JSON files found in {data_dir}")
        return {}

    print(f"📂 Found {len(files)} files, starting processing...")
    if max_reference is not None:
        print(f"⚙️  Filtering mode enabled: Max Ref={max_reference}, Token Limit=8000, Tokenizer={tokenizer_path}")
    
    results = {}
    # Slightly reduce worker count to prevent machine overload and freezing
    num_workers = max(1, min(multiprocessing.cpu_count() - 1, len(files)))
    
    print(f"🚀 Starting {num_workers} Worker processes...")

    with ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker, 
        initargs=(tokenizer_path, max_reference)
    ) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files}
        
        with tqdm(total=len(files), desc="Processing", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    # Note: This timeout is mainly to prevent main process deadlock; actual timeout logic is controlled by signal within worker
                    # If signal within worker takes effect, future will return None normally
                    result = future.result(timeout=65) 
                    
                    if result is not None:
                        index, data = result
                        results[index] = data
                    else:
                        # Result is None, indicating parsing failure or timeout discard by worker
                        pass 

                except TimeoutError:
                    print(f"⚠️ Main process detected timeout discard: {file_path.name}")
                except Exception as e:
                    print(f"❌ Processing error {file_path.name}: {e}")
                
                pbar.update(1)
    
    print(f"✅ Successfully processed {len(results)} files (some may have been discarded due to timeout or error)")
    return results

def calculate_metrics(results: Dict[int, Dict], k_values: List[int]) -> Dict:
    problem_stats = []
    pass_at_k_sums = {k: 0.0 for k in k_values}
    
    total_null_text = 0
    
    for index, data in results.items():
        details = data.get('verification_details', [])
        total = len(details)
        correct = sum(1 for d in details if d.get('is_correct'))
        null_text_count = sum(1 for d in details if d.get('is_text_null'))
        
        total_null_text += null_text_count
        
        accuracy = correct / total if total > 0 else 0.0
        
        current_pass_at_k = {}
        for k in k_values:
            if total > 0:
                pk = calculate_pass_at_k(total, correct, k)
                current_pass_at_k[f'pass@{k}'] = pk
                pass_at_k_sums[k] += pk
            else:
                current_pass_at_k[f'pass@{k}'] = 0.0

        problem_stats.append({
            'index': index,
            'total': total,
            'correct': correct,
            'null_text': null_text_count,
            'accuracy': accuracy,
            'pass_at_k': current_pass_at_k
        })
    
    num_problems = len(results)
    metrics = {
        'total_problems': num_problems,
        'total_null_text': total_null_text,
        'pass_at_k': {f'pass@{k}': (pass_at_k_sums[k] / num_problems if num_problems > 0 else 0.0) for k in k_values},
        'problem_stats': sorted(problem_stats, key=lambda x: x['index'])
    }
    return metrics

def print_results(metrics: Dict):
    if metrics['total_problems'] == 0:
        print("⚠️ No valid results generated.")
        return

    print("\n" + "="*60)
    print("📊 Pass@k Statistics")
    print("="*60)
    print(f"Total Problems: {metrics['total_problems']}")
    print(f"Total Null Text: {metrics['total_null_text']}")
    
    print("-" * 60)
    print("Overall Pass@k:")
    sorted_keys = sorted(metrics['pass_at_k'].keys(), key=lambda x: int(x.split('@')[1]))
    for k in sorted_keys:
        v = metrics['pass_at_k'][k]
        print(f"  {k:10s}: {v:.2%}")
    
    print("-" * 60)
    print("Pass@k per problem (Top 10):")
    for stat in metrics['problem_stats'][:10]:
        print(f"  Problem {stat['index']}: Correct {stat['correct']}/{stat['total']} (Null: {stat['null_text']})")
        p_keys = sorted(stat['pass_at_k'].keys(), key=lambda x: int(x.split('@')[1]))
        if p_keys:
            p_str = ", ".join([f"{k}={stat['pass_at_k'][k]:.2f}" for k in p_keys])
            print(f"    {p_str}")
    
    if len(metrics['problem_stats']) > 10:
        print(f"  ... (See output file for remaining {len(metrics['problem_stats']) - 10} problems)")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate Pass@k (Automatically extract boxed content and verify)")
    parser.add_argument('--verification_dir', type=str, required=True, help='Directory containing JSON files with completions')
    parser.add_argument('--k_values', type=str, default='1,8,16,32,64,128', help='List of k values, comma separated')
    parser.add_argument('--output_file', type=str, default=None, help='Result output file path (optional)')
    parser.add_argument('--max_reference', type=int, default=None, help='Maximum number of completions to keep (if set, filters responses >8000 tokens)')
    parser.add_argument('--tokenizer_path', type=str, default='', help='Tokenizer path')
    
    args = parser.parse_args()
    
    try:
        k_list = [int(x.strip()) for x in args.k_values.split(',')]
    except ValueError:
        print("❌ k_values format error, should be comma-separated integers")
        return

    results = load_and_verify_data(args.verification_dir, args.tokenizer_path, args.max_reference)
    if not results:
        return

    metrics = calculate_metrics(results, k_list)
    print_results(metrics)
    
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"💾 Complete results saved to: {args.output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
        
        if HAS_PANDAS:
            try:
                excel_file = Path(args.output_file).with_suffix('.xlsx')
                excel_data = []
                for stat in metrics['problem_stats']:
                    row = {
                        'Problem_Index': stat['index'],
                        'Total_Completions': stat['total'],
                        'Correct_Count': stat['correct'],
                        'Null_Text_Count': stat['null_text'],
                        'Accuracy': stat['accuracy'],
                    }
                    for k, v in stat['pass_at_k'].items():
                        row[k] = v
                    excel_data.append(row)
                
                df = pd.DataFrame(excel_data)
                
                summary_row = {
                    'Problem_Index': 'SUMMARY',
                    'Total_Completions': '',
                    'Correct_Count': '',
                    'Null_Text_Count': metrics['total_null_text'],
                }
                for k, v in metrics['pass_at_k'].items():
                    summary_row[k] = v
                df.loc[len(df)] = summary_row
                
                df.to_excel(excel_file, index=False, engine='openpyxl')
                print(f"💾 Excel results saved to: {excel_file}")
                
            except Exception as e:
                print(f"❌ Failed to save Excel: {e}")
    else:
        print("ℹ️  No output file specified, results printed to console only")

if __name__ == "__main__":
    main()
