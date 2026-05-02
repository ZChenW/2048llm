#!/usr/bin/env python3
"""Offline generation evaluation for 2048 SFT best-action models."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = ["up", "down", "left", "right"]
ACTION_SET = set(ACTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--format", choices=["raw_move", "answer_tag"], required=True)
    parser.add_argument("--task", choices=["best_action"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bad-cases-jsonl")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def parse_prediction(text: str, fmt: str) -> str | None:
    cleaned = text.strip().lower()
    if fmt == "answer_tag":
        match = re.search(r"<answer>\s*(up|down|left|right)\s*</answer>", cleaned)
        return match.group(1) if match else None
    tokens = re.findall(r"\b(up|down|left|right)\b", cleaned)
    return tokens[0] if tokens else None


def margin_bucket(value: Any) -> str:
    if value is None:
        return "null"
    value = float(value)
    if value == 0:
        return "0"
    if value < 0.001:
        return "(0,0.001)"
    if value < 0.01:
        return "[0.001,0.01)"
    if value < 0.1:
        return "[0.01,0.1)"
    if value < 1:
        return "[0.1,1)"
    if value < 10:
        return "[1,10)"
    if value < 100:
        return "[10,100)"
    if value < 1000:
        return "[100,1000)"
    return ">=1000"


def load_rows(path: str, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def expected_answer(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if "teacher_action" in metadata:
        return metadata["teacher_action"]
    content = row["messages"][1]["content"].strip().lower()
    parsed = parse_prediction(content, "answer_tag" if content.startswith("<answer>") else "raw_move")
    if parsed is None:
        raise ValueError("could not parse expected assistant answer")
    return parsed


def prompt_text(row: dict[str, Any], tokenizer: Any) -> str:
    user = row["messages"][0]
    return tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True)


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def nested_counter_to_dict(counter: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {str(key): counter_to_dict(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def ratio(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def main() -> int:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("choose at most one of --bf16 and --fp16")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = load_rows(args.val_file, args.limit)
    total = exact = parse_fail = invalid = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    by_max_tile: dict[str, Counter] = defaultdict(Counter)
    by_empty: dict[str, Counter] = defaultdict(Counter)
    by_depth: dict[str, Counter] = defaultdict(Counter)
    by_margin: dict[str, Counter] = defaultdict(Counter)
    bad_cases = []

    for index, row in enumerate(rows, 1):
        expected = expected_answer(row)
        metadata = row.get("metadata", {})
        valid_moves = set(metadata.get("valid_moves", ACTIONS))
        inputs = tokenizer(prompt_text(row, tokenizer), return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        pred = parse_prediction(generated_text, args.format)
        total += 1
        ok = pred == expected
        if ok:
            exact += 1
        if pred is None:
            parse_fail += 1
            pred_key = "__parse_fail__"
        else:
            pred_key = pred
            if pred not in valid_moves:
                invalid += 1
        confusion[expected][pred_key] += 1
        for bucket_map, key in [
            (by_max_tile, str(metadata.get("max_tile", "unknown"))),
            (by_empty, str(metadata.get("empty_cells", "unknown"))),
            (by_depth, str(metadata.get("search_depth", "unknown"))),
            (by_margin, margin_bucket(metadata.get("score_margin"))),
        ]:
            bucket_map[key]["total"] += 1
            bucket_map[key]["correct"] += int(ok)
        if not ok:
            bad_cases.append(
                {
                    "index": index,
                    "expected": expected,
                    "prediction": pred,
                    "generated_text": generated_text,
                    "metadata": metadata,
                    "messages": row["messages"],
                }
            )

    def accuracy_table(buckets: dict[str, Counter]) -> dict[str, dict[str, float | int]]:
        result = {}
        for key, values in sorted(buckets.items(), key=lambda item: str(item[0])):
            result[key] = {
                "total": int(values["total"]),
                "correct": int(values["correct"]),
                "accuracy": ratio(values["correct"], values["total"]),
            }
        return result

    result = {
        "total": total,
        "exact_match": exact,
        "exact_match_rate": ratio(exact, total),
        "parse_fail_count": parse_fail,
        "parse_fail_rate": ratio(parse_fail, total),
        "invalid_action_count": invalid,
        "invalid_action_rate": ratio(invalid, total),
        "confusion_matrix": nested_counter_to_dict(confusion),
        "accuracy_by_max_tile": accuracy_table(by_max_tile),
        "accuracy_by_empty_cells": accuracy_table(by_empty),
        "accuracy_by_search_depth": accuracy_table(by_depth),
        "accuracy_by_score_margin_bucket": accuracy_table(by_margin),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    bad_path = Path(args.bad_cases_jsonl) if args.bad_cases_jsonl else output_path.with_suffix(".bad_cases.jsonl")
    with bad_path.open("w", encoding="utf-8") as f:
        for case in bad_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"bad cases: {bad_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
