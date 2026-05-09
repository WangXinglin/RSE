"""
RSE Step 2.2: Per-Question Reflection Deduplication for TravelPlanner

Mirrors the Math domain's step2dot2_reflection_dedup_by_emb.py logic exactly:
  - Per-question processing: each question's reflections are deduped independently
  - Output: one file per question ({idx}.jsonl) in the output directory
  - Optional: merge with previous iteration's deduped reflections (--previous-reflection-dir)

Input:  step2 reflection directory containing {idx}.json files
        (each with "reflections" list — one per completion)
Output: per-question deduped files in output directory

Uses sentence-transformers for embedding-based similarity dedup.
"""
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
    raise ImportError("pip install sentence-transformers")

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


def deduplicate_logic_with_trace(items: list, threshold: float, keep_order: bool = False):
    """
    Dedup logic with trace — identical to Math domain's version.
    Input items: [{'text': str, 'source': str}, ...]
    Returns: (kept_texts, debug_clusters)
    """
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
            "len": len(item['text']),
            "meta": item.get('meta', {}),
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

    kept_items = []
    for c in clusters:
        kept_items.append({
            "text": c["head"]["text"],
            "meta": c["head"].get("meta", {}),
        })

    debug_info = []
    for c in clusters:
        if c["children"]:
            debug_info.append({
                "kept_content": c["head"]["text"],
                "merged_items": c["children"]
            })

    return kept_items, debug_info


def extract_content_from_reflection_file(file_path: str):
    """
    Extract propositions and pitfalls from a step2_self_reflect output file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return [], []

    raw_propositions = []
    raw_pitfalls = []

    for refl in data.get('reflections', []):
        reflection_obj = refl.get('reflection')
        if not reflection_obj or not isinstance(reflection_obj, dict):
            continue

        for prop in reflection_obj.get('propositions', []):
            stmt = prop.get('statement', '').strip()
            if stmt:
                raw_propositions.append({
                    'text': stmt,
                    'meta': {
                        'category': prop.get('category', 'unknown'),
                        'severity': prop.get('severity', 'unknown'),
                        'eval_check': prop.get('eval_check', ''),
                        'prop_type': prop.get('type', ''),
                    }
                })

        for pit in reflection_obj.get('pitfalls', []):
            stmt = pit.get('statement', '').strip()
            if stmt:
                raw_pitfalls.append({
                    'text': stmt,
                    'meta': {
                        'category': pit.get('category', 'unknown'),
                        'eval_check': pit.get('eval_check', ''),
                    }
                })

    return raw_propositions, raw_pitfalls


def load_previous_deduped(prev_file: str):
    """
    Load a previously-deduped output file (from a prior iteration).
    """
    try:
        with open(prev_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    content = record.get('reflection_parsed')
                    if isinstance(content, str):
                        content = json.loads(content)
                    if isinstance(content, dict):
                        props = []
                        for p in content.get('propositions', []):
                            if isinstance(p, dict) and p.get('statement'):
                                props.append({
                                    'text': p['statement'],
                                    'meta': {
                                        'category': p.get('category', 'unknown'),
                                        'severity': p.get('severity', 'unknown'),
                                        'eval_check': p.get('eval_check', ''),
                                        'prop_type': p.get('prop_type', p.get('type', '')),
                                    }
                                })
                            elif isinstance(p, str):
                                props.append({'text': p, 'meta': {}})
                        pits = []
                        for p in content.get('pitfalls', []):
                            if isinstance(p, dict) and p.get('statement'):
                                pits.append({
                                    'text': p['statement'],
                                    'meta': {
                                        'category': p.get('category', 'unknown'),
                                        'eval_check': p.get('eval_check', ''),
                                    }
                                })
                            elif isinstance(p, str):
                                pits.append({'text': p, 'meta': {}})
                        return props, pits
    except Exception:
        pass
    return [], []


def process_single_file(
    file_path: Path,
    threshold: float,
    debug_dir: Path,
    previous_dir: Path = None,
    keep_order: bool = False
):
    """
    Worker function: process one question's reflections.
    """
    try:
        combined_props = []
        combined_pits = []

        if previous_dir:
            prev_file = previous_dir / file_path.name
            if prev_file.exists():
                prev_props, prev_pits = load_previous_deduped(str(prev_file))
                combined_props.extend([{'text': p['text'], 'source': 'previous', 'meta': p.get('meta', {})} for p in prev_props])
                combined_pits.extend([{'text': p['text'], 'source': 'previous', 'meta': p.get('meta', {})} for p in prev_pits])

        cur_props, cur_pits = extract_content_from_reflection_file(str(file_path))
        combined_props.extend([{'text': p['text'], 'source': 'current', 'meta': p.get('meta', {})} for p in cur_props])
        combined_pits.extend([{'text': p['text'], 'source': 'current', 'meta': p.get('meta', {})} for p in cur_pits])

        if not combined_props and not combined_pits:
            return None

        unique_props, debug_props = deduplicate_logic_with_trace(combined_props, threshold, keep_order=keep_order)
        unique_pits, debug_pits = deduplicate_logic_with_trace(combined_pits, threshold, keep_order=keep_order)

        try:
            q_idx = int(file_path.stem)
        except ValueError:
            q_idx = -1

        out_propositions = []
        for item in unique_props:
            entry = {"statement": item['text']}
            meta = item.get('meta', {})
            if meta.get('category'):
                entry['category'] = meta['category']
            if meta.get('severity'):
                entry['severity'] = meta['severity']
            if meta.get('eval_check'):
                entry['eval_check'] = meta['eval_check']
            if meta.get('prop_type'):
                entry['prop_type'] = meta['prop_type']
            out_propositions.append(entry)

        out_pitfalls = []
        for item in unique_pits:
            entry = {"statement": item['text']}
            meta = item.get('meta', {})
            if meta.get('category'):
                entry['category'] = meta['category']
            if meta.get('eval_check'):
                entry['eval_check'] = meta['eval_check']
            out_pitfalls.append(entry)

        result_record = {
            "question_id": q_idx,
            "rollout_idx": -1,
            "reflection_parsed": json.dumps({
                "propositions": out_propositions,
                "pitfalls": out_pitfalls,
            }, ensure_ascii=False)
        }

        if debug_dir:
            debug_report = {
                "question_id": q_idx,
                "threshold_used": threshold,
                "propositions_analysis": debug_props,
                "pitfalls_analysis": debug_pits,
            }
            debug_file = debug_dir / f"{file_path.stem}_debug.json"
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_report, f, indent=2, ensure_ascii=False)

        return file_path.name, result_record

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="RSE Step 2.2: Per-Question Reflection Dedup (mirrors Math domain)")
    parser.add_argument('--reflection-dir', type=str, required=True,
                        help='Step 2 reflection output directory (with {idx}.json files)')
    parser.add_argument('--previous-reflection-dir', type=str, default=None,
                        help='Optional: previous iteration deduped output directory for merging')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory (one .jsonl per question)')
    parser.add_argument('--debug-dir', type=str, default=None,
                        help='If set, save similarity debug reports here')
    parser.add_argument('--keep-order', action='store_true', default=False,
                        help='Restore original order after dedup')
    parser.add_argument('--model-path', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name or path')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Similarity threshold for dedup')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')

    args = parser.parse_args()

    input_path = Path(args.reflection_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    previous_path = None
    if args.previous_reflection_dir:
        previous_path = Path(args.previous_reflection_dir)
        if not previous_path.exists():
            logger.warning(f"Previous dir does not exist: {previous_path}")

    debug_dir_path = None
    if args.debug_dir:
        debug_dir_path = Path(args.debug_dir)
        debug_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode ON. Reports → {debug_dir_path}")

    files = sorted(input_path.glob("*.json"))
    logger.info(f"Found {len(files)} reflection files in {input_path}")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.model_path, True)
    ) as executor:

        worker_func = partial(
            process_single_file,
            threshold=args.threshold,
            debug_dir=debug_dir_path,
            previous_dir=previous_path,
            keep_order=args.keep_order
        )

        future_to_file = {executor.submit(worker_func, f): f for f in files}
        pbar = tqdm(total=len(files))

        saved = 0
        for future in future_to_file:
            try:
                result = future.result()
                if result:
                    filename, record = result
                    out_name = Path(filename).stem + ".jsonl"
                    out_file = output_path / out_name
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
                    saved += 1
            except Exception as e:
                logger.error(f"Error: {e}")
            finally:
                pbar.update(1)
        pbar.close()

    logger.info(f"Done! Saved {saved} deduped files to {output_path}")


if __name__ == "__main__":
    main()
