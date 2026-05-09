#!/usr/bin/env python3
"""
Convert custom output format to LiveCodeBench official evaluation format.

Key points:
1. Ensures the number of converted samples strictly matches the original completions (no sample loss).
2. Even if no code can be extracted from a completion, it is kept as an empty string to avoid length mismatch in the evaluator.
3. Includes comprehensive statistics and warnings for empty/abnormal samples.
4. Automatically validates code_list length for each question after conversion.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def extract_code_from_text(text: str, platform: str = "") -> str:
    """
    Extract code from completion text.

    Strategy:
    1. Try to extract ```python ... ``` or ```...``` code blocks.
    2. If no code block found, try to extract from the first import/from/def/class statement.
    3. If all fail, return the original text.strip().

    Note:
    - This function does not filter samples.
    - Even if an empty string is returned, the caller must keep the sample to ensure length alignment.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Strategy 1: Extract markdown code blocks
    code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if "class Solution" in code or "def " in code or "import " in code or "from " in code:
            return code
        # Even without obvious def/class/import, prefer the last code block
        return code

    # Strategy 2: Find the first import/from/def/class statement
    code_start_pattern = r"(?:^|\n)(import\s+|from\s+|def\s+\w+|class\s+\w+)"
    match = re.search(code_start_pattern, text)
    if match:
        return text[match.start():].strip()

    return text.strip()


def convert_single_file(input_path: str, verbose_empty: bool = False) -> Tuple[Dict, Dict]:
    """
    Convert a single question file.

    Returns:
    - result: {"question_id": ..., "code_list": [...]}
    - stats:  statistics
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_id = data.get("question_id")
    platform = data.get("platform", "").lower()
    completions = data.get("completions", [])

    if not isinstance(completions, list):
        raise ValueError(f"{input_path}: completions is not a list, but {type(completions).__name__}")

    code_list: List[str] = []
    empty_indices: List[int] = []
    non_dict_indices: List[int] = []

    for idx, comp in enumerate(completions):
        # Handle abnormal format: if completion is not a dict, try to handle gracefully
        if isinstance(comp, dict):
            text = comp.get("text", "")
        else:
            non_dict_indices.append(idx)
            text = ""

        code = extract_code_from_text(text, platform)

        if not isinstance(code, str):
            code = str(code) if code is not None else ""

        code = code.strip()

        if code == "":
            empty_indices.append(idx)
            if verbose_empty:
                print(
                    f"[WARN] empty extracted code | "
                    f"question_id={question_id} | completion_idx={idx} | file={input_path}"
                )

        # Key: always append regardless of whether it is empty, to ensure strict length consistency
        code_list.append(code)

    if len(code_list) != len(completions):
        raise RuntimeError(
            f"{input_path}: length mismatch after conversion, "
            f"len(completions)={len(completions)}, len(code_list)={len(code_list)}"
        )

    result = {
        "question_id": question_id,
        "code_list": code_list,
    }

    stats = {
        "question_id": question_id,
        "platform": platform,
        "input_file": input_path,
        "original_count": len(completions),
        "converted_count": len(code_list),
        "empty_count": len(empty_indices),
        "empty_indices": empty_indices,
        "non_dict_count": len(non_dict_indices),
        "non_dict_indices": non_dict_indices,
    }

    return result, stats


def convert_directory(
    input_dir: str,
    output_file: str,
    report_file: str = "",
    verbose_empty: bool = False,
):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(
        [f for f in input_path.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )

    print(f"Found {len(json_files)} JSON files")

    results: List[Dict] = []
    platform_stats: Dict[str, int] = {}

    total_original = 0
    total_converted = 0
    total_empty = 0
    total_non_dict = 0

    bad_length_files = []
    empty_problem_stats = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            platform = raw_data.get("platform", "unknown")
            platform_stats[platform] = platform_stats.get(platform, 0) + 1

            result, stats = convert_single_file(str(json_file), verbose_empty=verbose_empty)
            results.append(result)

            total_original += stats["original_count"]
            total_converted += stats["converted_count"]
            total_empty += stats["empty_count"]
            total_non_dict += stats["non_dict_count"]

            if stats["original_count"] != stats["converted_count"]:
                bad_length_files.append(
                    {
                        "input_file": stats["input_file"],
                        "question_id": stats["question_id"],
                        "original_count": stats["original_count"],
                        "converted_count": stats["converted_count"],
                    }
                )

            if stats["empty_count"] > 0 or stats["non_dict_count"] > 0:
                empty_problem_stats.append(stats)

        except Exception as e:
            print(f"[ERROR] Error processing {json_file}: {e}")

    # Write main results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Secondary check: verify code_list length for each question
    final_bad = []
    for i, item in enumerate(results):
        code_list = item.get("code_list", [])
        if not isinstance(code_list, list):
            final_bad.append((i, item.get("question_id"), "code_list_not_list"))
        elif len(code_list) == 0:
            final_bad.append((i, item.get("question_id"), 0))

    print(f"\nConversion complete: {output_file}")
    print(f"Total {len(results)} questions")
    print("\nPlatform distribution:")
    for p, c in sorted(platform_stats.items()):
        print(f"  {p}: {c}")

    print("\nOverall statistics:")
    print(f"  Total original completions: {total_original}")
    print(f"  Total converted code_list: {total_converted}")
    print(f"  Total empty code samples: {total_empty}")
    print(f"  Total non-dict completions: {total_non_dict}")

    if total_original != total_converted:
        print("\n[ERROR] Total length mismatch!")
        print(f"  total_original={total_original}, total_converted={total_converted}")
    else:
        print("\n[OK] Total length consistent")

    if bad_length_files:
        print("\n[ERROR] The following files have length mismatch between original and converted:")
        for item in bad_length_files[:20]:
            print(
                f"  file={item['input_file']} | "
                f"question_id={item['question_id']} | "
                f"original={item['original_count']} | converted={item['converted_count']}"
            )
    else:
        print("[OK] All files maintain original length")

    if final_bad:
        print("\n[WARN] The following questions have abnormal final code_list:")
        for item in final_bad[:20]:
            print(f"  result_idx={item[0]} | question_id={item[1]} | issue={item[2]}")
    else:
        print("[OK] Final result structure check passed")

    if empty_problem_stats:
        print("\nQuestions with empty code / abnormal completions (showing up to 20):")
        for s in empty_problem_stats[:20]:
            print(
                f"  question_id={s['question_id']} | "
                f"platform={s['platform']} | "
                f"original={s['original_count']} | "
                f"empty={s['empty_count']} | "
                f"non_dict={s['non_dict_count']}"
            )

    # Optional: write detailed report
    if report_file:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": {
                "num_questions": len(results),
                "total_original": total_original,
                "total_converted": total_converted,
                "total_empty": total_empty,
                "total_non_dict": total_non_dict,
                "length_match": total_original == total_converted,
                "platform_stats": platform_stats,
            },
            "bad_length_files": bad_length_files,
            "empty_problem_stats": empty_problem_stats,
            "final_bad": final_bad,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report written to: {report_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--verbose_empty",
        action="store_true",
        help="Print the location of each empty code sample",
    )
    args = parser.parse_args()

    convert_directory(
        input_dir=args.input_dir,
        output_file=args.output_file,
        report_file=args.report_file,
        verbose_empty=args.verbose_empty,
    )


if __name__ == "__main__":
    main()
