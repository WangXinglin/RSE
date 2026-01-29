import json
import argparse
import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import torch

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PID %(process)d] - %(message)s'
)
logger = logging.getLogger(__name__)


_worker_model = None

def init_worker(model_name_or_path: str, use_cpu_only: bool = True):
    global _worker_model
    torch.set_num_threads(1) 
    os.environ["OMP_NUM_THREADS"] = "1"
    
    device = "cpu"
    if not use_cpu_only and torch.cuda.is_available():
        device = "cuda"

    try:
        _worker_model = SentenceTransformer(model_name_or_path, device=device)
    except Exception as e:
        logger.error(f"Worker Model load failed: {e}")
        _worker_model = None

def is_subset_string(short_str: str, long_str: str) -> bool:
    if len(short_str) >= len(long_str):
        return False
    return short_str in long_str

def deduplicate_logic_with_trace(items: list[dict], threshold: float, keep_order: bool = False):
    global _worker_model
    if not items or _worker_model is None:
        return [], []

    unique_text_map = {}
    
    for idx, item in enumerate(items):
        t = item['text']
        source = item['source']
        clean = t.strip()
        if clean:
            item_with_idx = item.copy()
            item_with_idx['original_index'] = idx
            
            if clean not in unique_text_map:
                unique_text_map[clean] = item_with_idx
            elif source == 'current':
                unique_text_map[clean] = item_with_idx
    
    unique_items = list(unique_text_map.values())
    unique_texts = [i['text'] for i in unique_items]
    
    if not unique_texts:
        return [], []

    try:
        embeddings = _worker_model.encode(unique_texts, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        return unique_texts, []

    candidates = []
    for idx, item in enumerate(unique_items):
        candidates.append({
            "text": item['text'],
            "source": item['source'],
            "original_index": item['original_index'],
            "emb": embeddings[idx],
            "len": len(item['text'])
        })
    
    candidates.sort(key=lambda x: (1 if x['source'] == 'current' else 0, x["len"]), reverse=True)

    clusters = []
    
    kept_emb_stack = None

    for cand in candidates:
        cand_text = cand["text"]
        cand_emb = cand["emb"]
        
        matched_cluster_idx = -1
        match_reason = None
        match_score = 0.0

        for i, cluster in enumerate(clusters):
            if is_subset_string(cand_text, cluster["head"]["text"]):
                matched_cluster_idx = i
                match_reason = "subset"
                match_score = 1.0
                break
        
        if matched_cluster_idx == -1 and clusters:
            if kept_emb_stack is None:
                kept_emb_stack = torch.stack([c["head"]["emb"] for c in clusters])
            
            sim_scores = util.cos_sim(cand_emb, kept_emb_stack)[0]
            
            max_val, max_idx = torch.max(sim_scores, dim=0)
            score = max_val.item()
            
            if score > threshold:
                matched_cluster_idx = max_idx.item()
                match_reason = "similarity"
                match_score = score
        
        if matched_cluster_idx != -1:
            clusters[matched_cluster_idx]["children"].append({
                "text": cand_text,
                "reason": match_reason,
                "score": round(match_score, 4)
            })
        else:
            clusters.append({
                "head": cand,
                "children": []
            })
            if kept_emb_stack is None:
                kept_emb_stack = cand_emb.unsqueeze(0)
            else:
                kept_emb_stack = torch.cat((kept_emb_stack, cand_emb.unsqueeze(0)), dim=0)

    if keep_order:
        clusters.sort(key=lambda c: c["head"]["original_index"])

    kept_texts = [c["head"]["text"] for c in clusters]
    
    debug_info = []
    for c in clusters:
        if c["children"]:
            debug_info.append({
                "kept_content": c["head"]["text"],
                "merged_items": c["children"]
            })

    return kept_texts, debug_info


def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: data.append(json.loads(line.strip()))
                    except: continue
    except: pass
    return data

def extract_content_from_records(records, n_completions=None):
    if n_completions is not None:
        records.sort(key=lambda x: x.get('rollout_idx', 0))
        target_records = records[:n_completions]
    else:
        target_records = records
    
    raw_pitfalls = []
    raw_propositions = []
    
    for record in target_records:
        content = record.get('experience_parsed')
        content_dict = {}
        if isinstance(content, dict): content_dict = content
        elif isinstance(content, str):
            try: content_dict = json.loads(content.replace("```json", "").replace("```", "").strip())
            except: continue
        
        if isinstance(content_dict, dict):
            for p in content_dict.get('critical_pitfalls', []):
                if isinstance(p, str): raw_pitfalls.append(p)
                elif isinstance(p, dict):
                    if 'error_location' in p and 'reasoning_flaw' in p:
                        raw_pitfalls.append(f"{p['error_location']}: {p['reasoning_flaw']}")
                    elif 'content' in p: raw_pitfalls.append(p['content'])
            for p in content_dict.get('verified_propositions', []):
                if isinstance(p, str): raw_propositions.append(p)
                elif isinstance(p, dict) and 'content' in p: raw_propositions.append(p['content'])
                
    return raw_pitfalls, raw_propositions

def process_single_file(file_path: Path, n_completions: int, threshold: float, debug_dir: Path, previous_dir: Path = None, keep_order: bool = False):
    try:
        combined_pits = []
        combined_props = []

        if previous_dir:
            prev_file = previous_dir / file_path.name
            if prev_file.exists():
                prev_records = load_jsonl(str(prev_file))
                prev_pits, prev_props = extract_content_from_records(prev_records, n_completions=None)
                
                combined_pits.extend([{'text': p, 'source': 'previous'} for p in prev_pits])
                combined_props.extend([{'text': p, 'source': 'previous'} for p in prev_props])

        records = load_jsonl(str(file_path))
        if records:
            cur_pits, cur_props = extract_content_from_records(records, n_completions)
            combined_pits.extend([{'text': p, 'source': 'current'} for p in cur_pits])
            combined_props.extend([{'text': p, 'source': 'current'} for p in cur_props])
        
        if not combined_pits and not combined_props:
            return None

        unique_pits, debug_pits = deduplicate_logic_with_trace(combined_pits, threshold, keep_order=keep_order)
        unique_props, debug_props = deduplicate_logic_with_trace(combined_props, threshold, keep_order=keep_order)
        
        try: q_idx = int(file_path.stem)
        except: q_idx = records[0].get('question_id', -1)
            
        result_record = {
            "question_id": q_idx,
            "rollout_idx": -1,
            "experience_parsed": json.dumps({
                "critical_pitfalls": unique_pits,
                "verified_propositions": unique_props
            }, ensure_ascii=False)
        }

        if debug_dir:
            debug_report = {
                "question_id": q_idx,
                "threshold_used": threshold,
                "pitfalls_analysis": debug_pits,
                "propositions_analysis": debug_props
            }
            debug_file = debug_dir / f"{file_path.stem}_debug.json"
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_report, f, indent=2, ensure_ascii=False)
        
        return file_path.name, result_record
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experience-dir', type=str, required=True, help='Raw experience JSONL folder')
    parser.add_argument('--output-dir', type=str, required=True, help='Output folder')
    parser.add_argument('--previous-experience-dir', type=str, default=None, help='Optional: Previous memory bank location. Merges and deduplicates with current experiences.')
    parser.add_argument('--debug-dir', type=str, help='If set, outputs similarity comparison reports')
    parser.add_argument('--keep-order', action='store_true', default=False, help='If True, restore original order of propositions/pitfalls after deduplication.')
    parser.add_argument('--model-path', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--n-experience', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--workers', type=int, default=8)
    
    args = parser.parse_args()

    input_path = Path(args.experience_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    previous_path = None
    if args.previous_experience_dir:
        previous_path = Path(args.previous_experience_dir)
        if not previous_path.exists():
            logger.warning(f"Previous experience dir provided but does not exist: {previous_path}")
    
    debug_dir_path = None
    if args.debug_dir:
        debug_dir_path = Path(args.debug_dir)
        debug_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode ON. Reports will be saved to {debug_dir_path}")

    files = list(input_path.glob("*.jsonl"))
    
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.model_path, True)
    ) as executor:
        
        worker_func = partial(
            process_single_file,
            n_completions=args.n_experience,
            threshold=args.threshold,
            debug_dir=debug_dir_path,
            previous_dir=previous_path,
            keep_order=args.keep_order
        )
        
        future_to_file = {executor.submit(worker_func, f): f for f in files}
        pbar = tqdm(total=len(files))
        
        for future in future_to_file:
            try:
                result = future.result()
                if result:
                    filename, record = result
                    out_file = output_path / filename
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
            except Exception as e:
                logger.error(f"Error: {e}")
            finally:
                pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    main()
