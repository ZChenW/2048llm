#!/usr/bin/env python3
"""Offline diagnostics for SFT action-selection eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_OUTPUT_ROOT = Path("/mnt/cun/2048llm_outputs/qwen2p5_7b_best_action_raw_move_lora_v1")
DEFAULT_EVAL_JSON = DEFAULT_OUTPUT_ROOT / "offline_eval_val_10000.json"
DEFAULT_BAD_CASES = DEFAULT_OUTPUT_ROOT / "offline_eval_val_10000.bad_cases.jsonl"
DEFAULT_OUT_DIR = DEFAULT_OUTPUT_ROOT / "diagnostics_10000"
DEFAULT_VAL_FILE = Path("data/sft/best_action_raw_move_val.jsonl")

ACTION_ORDER = ["up", "down", "left", "right"]
SCORE_MARGIN_BUCKET_ORDER = [
    "0",
    "(0,0.001)",
    "[0.001,0.01)",
    "[0.01,0.1)",
    "[0.1,1)",
    "[1,10)",
    "[10,100)",
    "[100,1000)",
    ">=1000",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--bad-cases", type=Path, default=DEFAULT_BAD_CASES)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=200)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def assistant_action(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("teacher_action"):
        return str(metadata["teacher_action"])
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    raise ValueError("SFT row has no assistant action or metadata.teacher_action")


def score_margin_bucket(value: Any) -> str:
    margin = float(value or 0.0)
    if margin == 0:
        return "0"
    if margin < 0.001:
        return "(0,0.001)"
    if margin < 0.01:
        return "[0.001,0.01)"
    if margin < 0.1:
        return "[0.01,0.1)"
    if margin < 1:
        return "[0.1,1)"
    if margin < 10:
        return "[1,10)"
    if margin < 100:
        return "[10,100)"
    if margin < 1000:
        return "[100,1000)"
    return ">=1000"


def bucket_sort_key(value: str) -> tuple[int, Any]:
    if value in SCORE_MARGIN_BUCKET_ORDER:
        return (0, SCORE_MARGIN_BUCKET_ORDER.index(value))
    try:
        return (1, int(value))
    except ValueError:
        try:
            return (1, float(value))
        except ValueError:
            return (2, value)


def pct(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def empty_stats() -> dict[str, int | float]:
    return {"total": 0, "correct": 0, "errors": 0, "accuracy": 0.0, "error_rate": 0.0}


def finalize_stats(stats: dict[str, int | float]) -> dict[str, int | float]:
    total = int(stats["total"])
    correct = int(stats["correct"])
    errors = total - correct
    stats["errors"] = errors
    stats["accuracy"] = pct(correct, total)
    stats["error_rate"] = pct(errors, total)
    return stats


def metadata_match_score(bad_case: dict[str, Any], val_row: dict[str, Any]) -> int:
    bad_meta = bad_case.get("metadata") or {}
    val_meta = val_row.get("metadata") or {}
    score = 0
    if bad_case.get("expected") == assistant_action(val_row):
        score += 4
    for key in ("search_depth", "max_tile", "empty_cells", "legal_move_count"):
        if key in bad_meta and bad_meta.get(key) == val_meta.get(key):
            score += 1
    if "score_margin" in bad_meta and "score_margin" in val_meta:
        if abs(float(bad_meta["score_margin"]) - float(val_meta["score_margin"])) < 1e-6:
            score += 2
    return score


def resolve_bad_case_indices(
    bad_cases: list[dict[str, Any]], val_rows: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    by_index: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    for ordinal, bad_case in enumerate(bad_cases):
        raw_index = bad_case.get("index")
        candidates: list[int] = []
        if isinstance(raw_index, int):
            candidates.extend([raw_index, raw_index - 1])
        elif isinstance(raw_index, str) and raw_index.isdigit():
            idx = int(raw_index)
            candidates.extend([idx, idx - 1])

        best_index = None
        best_score = -1
        for candidate in candidates:
            if 0 <= candidate < len(val_rows):
                score = metadata_match_score(bad_case, val_rows[candidate])
                if score > best_score:
                    best_index = candidate
                    best_score = score

        if best_index is None:
            warnings.append(f"bad case {ordinal} has unresolved index {raw_index!r}")
            continue
        if best_index in by_index:
            warnings.append(f"duplicate bad case index resolved to {best_index}")
        enriched = dict(bad_case)
        enriched["_zero_based_index"] = best_index
        by_index[best_index] = enriched
    return by_index, warnings


def build_records(
    val_rows: list[dict[str, Any]], bad_cases: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    bad_by_index, warnings = resolve_bad_case_indices(bad_cases, val_rows)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(val_rows):
        metadata = row.get("metadata") or {}
        expected = assistant_action(row)
        bad_case = bad_by_index.get(idx)
        prediction = expected if bad_case is None else str(bad_case.get("prediction", "")).strip()
        correct = bad_case is None
        records.append(
            {
                "index": idx,
                "expected": expected,
                "prediction": prediction,
                "correct": correct,
                "search_depth": str(metadata.get("search_depth", "")),
                "score_margin": float(metadata.get("score_margin", 0.0) or 0.0),
                "score_margin_bucket": score_margin_bucket(metadata.get("score_margin", 0.0)),
                "max_tile_bucket": str(metadata.get("max_tile", "")),
                "empty_cells_bucket": str(metadata.get("empty_cells", "")),
                "legal_move_count": str(metadata.get("legal_move_count", "")),
                "bad_case": bad_case,
            }
        )
    return records, warnings


def stats_by(records: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, dict[str, int | float]] = defaultdict(empty_stats)
    for record in records:
        key = key_fn(record)
        grouped[key]["total"] = int(grouped[key]["total"]) + 1
        grouped[key]["correct"] = int(grouped[key]["correct"]) + int(bool(record["correct"]))
    return {key: finalize_stats(grouped[key]) for key in sorted(grouped, key=bucket_sort_key)}


def cross_stats(
    records: list[dict[str, Any]],
    row_key_fn: Callable[[dict[str, Any]], str],
    col_key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, dict[str, int | float]]]:
    grouped: dict[str, dict[str, dict[str, int | float]]] = defaultdict(lambda: defaultdict(empty_stats))
    for record in records:
        row_key = row_key_fn(record)
        col_key = col_key_fn(record)
        cell = grouped[row_key][col_key]
        cell["total"] = int(cell["total"]) + 1
        cell["correct"] = int(cell["correct"]) + int(bool(record["correct"]))
    return {
        row_key: {
            col_key: finalize_stats(grouped[row_key][col_key])
            for col_key in sorted(grouped[row_key], key=bucket_sort_key)
        }
        for row_key in sorted(grouped, key=bucket_sort_key)
    }


def confusion_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    labels = sorted({r["expected"] for r in records} | {r["prediction"] for r in records}, key=bucket_sort_key)
    matrix = {label: {pred: 0 for pred in labels} for label in labels}
    for record in records:
        matrix[str(record["expected"])][str(record["prediction"])] += 1
    return matrix


def write_stats_csv(path: Path, stats: dict[str, dict[str, Any]], key_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[key_name, "total", "correct", "errors", "accuracy", "error_rate"])
        writer.writeheader()
        for key, values in stats.items():
            writer.writerow({key_name: key, **values})


def write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    labels = sorted(matrix, key=bucket_sort_key)
    pred_labels = sorted({pred for row in matrix.values() for pred in row}, key=bucket_sort_key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gold_action", *pred_labels])
        writer.writeheader()
        for label in labels:
            writer.writerow({"gold_action": label, **{pred: matrix[label].get(pred, 0) for pred in pred_labels}})


def write_cross_csv(path: Path, data: dict[str, dict[str, dict[str, Any]]], row_name: str, col_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[row_name, col_name, "total", "correct", "errors", "accuracy", "error_rate"],
        )
        writer.writeheader()
        for row_key, columns in data.items():
            for col_key, values in columns.items():
                writer.writerow({row_name: row_key, col_name: col_key, **values})


def markdown_stats_table(title: str, stats: dict[str, dict[str, Any]], key_name: str) -> str:
    lines = [
        f"### {title}",
        "",
        f"| {key_name} | Total | Correct | Errors | Accuracy | Error rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in stats.items():
        lines.append(
            f"| {key} | {values['total']} | {values['correct']} | {values['errors']} | "
            f"{values['accuracy']:.10f} | {values['error_rate']:.10f} |"
        )
    return "\n".join(lines)


def sample_bad_cases(records: list[dict[str, Any]], sample_limit: int) -> dict[str, list[dict[str, Any]]]:
    error_records = [r for r in records if not r["correct"] and r.get("bad_case")]

    def raw(record: dict[str, Any]) -> dict[str, Any]:
        item = dict(record["bad_case"])
        item["_diagnostic"] = {
            "zero_based_index": record["index"],
            "score_margin_bucket": record["score_margin_bucket"],
            "max_tile_bucket": record["max_tile_bucket"],
            "empty_cells_bucket": record["empty_cells_bucket"],
            "legal_move_count": record["legal_move_count"],
        }
        return item

    filters = {
        "high_margin_errors": lambda r: r["score_margin"] >= 100,
        "very_high_margin_errors": lambda r: r["score_margin"] >= 1000,
        "left_right_to_up_down_errors": lambda r: r["expected"] in {"left", "right"}
        and r["prediction"] in {"up", "down"},
        "depth2_errors": lambda r: r["search_depth"] == "2",
        "low_empty_cells_errors": lambda r: r["empty_cells_bucket"].isdigit() and int(r["empty_cells_bucket"]) <= 4,
        "low_legal_move_count_errors": lambda r: r["legal_move_count"].isdigit() and int(r["legal_move_count"]) <= 2,
    }
    return {
        name: [raw(record) for record in error_records if predicate(record)][:sample_limit]
        for name, predicate in filters.items()
    }


def write_markdown_summary(path: Path, diagnostics: dict[str, Any], sample_counts: dict[str, int]) -> None:
    overall = diagnostics["overall_metrics"]
    lines = [
        "# 10000-step SFT Eval Diagnostics",
        "",
        "## Overall metrics",
        "",
        f"- total: `{overall.get('total')}`",
        f"- exact_match: `{overall.get('exact_match')}`",
        f"- exact_match_rate: `{overall.get('exact_match_rate')}`",
        f"- parse_fail_rate: `{overall.get('parse_fail_rate')}`",
        f"- invalid_action_rate: `{overall.get('invalid_action_rate')}`",
        f"- bad_cases: `{diagnostics['bad_case_count']}`",
        "",
        markdown_stats_table("Per-action accuracy", diagnostics["per_action_accuracy"], "gold_action"),
        "",
        markdown_stats_table("Accuracy by search_depth", diagnostics["accuracy_by_search_depth"], "search_depth"),
        "",
        markdown_stats_table(
            "Accuracy by score_margin bucket",
            diagnostics["accuracy_by_score_margin_bucket"],
            "score_margin_bucket",
        ),
        "",
        markdown_stats_table("Accuracy by max_tile bucket", diagnostics["accuracy_by_max_tile_bucket"], "max_tile"),
        "",
        markdown_stats_table(
            "Accuracy by empty_cells bucket",
            diagnostics["accuracy_by_empty_cells_bucket"],
            "empty_cells",
        ),
        "",
        markdown_stats_table(
            "Accuracy by legal_move_count",
            diagnostics["accuracy_by_legal_move_count"],
            "legal_move_count",
        ),
        "",
        "## Cross Statistics",
        "",
        "The CSV files contain total, correct, error count, accuracy, and error rate for each cell:",
        "",
        "- `cross_gold_action_by_search_depth.csv`",
        "- `cross_gold_action_by_score_margin_bucket.csv`",
        "- `cross_search_depth_by_score_margin_bucket.csv`",
        "- `cross_gold_action_by_empty_cells_bucket.csv`",
        "- `cross_gold_action_by_legal_move_count.csv`",
        "",
        "## Sampled Error Files",
        "",
        "| Sample | Rows written |",
        "| --- | ---: |",
    ]
    for name, count in sorted(sample_counts.items()):
        lines.append(f"| `{name}.jsonl` | {count} |")
    if diagnostics["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in diagnostics["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(eval_json: Path, bad_cases_path: Path, val_file: Path, out_dir: Path, sample_limit: int) -> dict[str, Any]:
    if sample_limit <= 0:
        raise ValueError("--sample-limit must be positive")
    eval_metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    val_rows = read_jsonl(val_file)
    bad_cases = read_jsonl(bad_cases_path)
    records, warnings = build_records(val_rows, bad_cases)

    if len(records) != int(eval_metrics.get("total", len(records))):
        warnings.append(f"val row count {len(records)} differs from eval total {eval_metrics.get('total')}")
    if len(bad_cases) != int(eval_metrics.get("total", 0)) - int(eval_metrics.get("exact_match", 0)):
        warnings.append(
            "bad case count "
            f"{len(bad_cases)} differs from total-exact_match "
            f"{int(eval_metrics.get('total', 0)) - int(eval_metrics.get('exact_match', 0))}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    diagnostics = {
        "inputs": {
            "eval_json": str(eval_json),
            "bad_cases": str(bad_cases_path),
            "val_file": str(val_file),
        },
        "overall_metrics": {
            key: eval_metrics.get(key)
            for key in (
                "total",
                "exact_match",
                "exact_match_rate",
                "parse_fail_count",
                "parse_fail_rate",
                "invalid_action_count",
                "invalid_action_rate",
            )
        },
        "bad_case_count": len(bad_cases),
        "confusion_matrix": eval_metrics.get("confusion_matrix") or confusion_matrix(records),
        "computed_confusion_matrix": confusion_matrix(records),
        "per_action_accuracy": stats_by(records, lambda r: str(r["expected"])),
        "accuracy_by_search_depth": stats_by(records, lambda r: str(r["search_depth"])),
        "accuracy_by_score_margin_bucket": stats_by(records, lambda r: str(r["score_margin_bucket"])),
        "accuracy_by_max_tile_bucket": stats_by(records, lambda r: str(r["max_tile_bucket"])),
        "accuracy_by_empty_cells_bucket": stats_by(records, lambda r: str(r["empty_cells_bucket"])),
        "accuracy_by_legal_move_count": stats_by(records, lambda r: str(r["legal_move_count"])),
        "cross_gold_action_by_search_depth": cross_stats(
            records, lambda r: str(r["expected"]), lambda r: str(r["search_depth"])
        ),
        "cross_gold_action_by_score_margin_bucket": cross_stats(
            records, lambda r: str(r["expected"]), lambda r: str(r["score_margin_bucket"])
        ),
        "cross_search_depth_by_score_margin_bucket": cross_stats(
            records, lambda r: str(r["search_depth"]), lambda r: str(r["score_margin_bucket"])
        ),
        "cross_gold_action_by_empty_cells_bucket": cross_stats(
            records, lambda r: str(r["expected"]), lambda r: str(r["empty_cells_bucket"])
        ),
        "cross_gold_action_by_legal_move_count": cross_stats(
            records, lambda r: str(r["expected"]), lambda r: str(r["legal_move_count"])
        ),
        "warnings": warnings,
    }

    write_json(out_dir / "diagnostics_summary.json", diagnostics)
    write_json(out_dir / "overall_metrics.json", diagnostics["overall_metrics"])
    write_confusion_csv(out_dir / "confusion_matrix.csv", diagnostics["confusion_matrix"])
    write_stats_csv(out_dir / "per_action_accuracy.csv", diagnostics["per_action_accuracy"], "gold_action")
    write_stats_csv(out_dir / "accuracy_by_search_depth.csv", diagnostics["accuracy_by_search_depth"], "search_depth")
    write_stats_csv(
        out_dir / "accuracy_by_score_margin_bucket.csv",
        diagnostics["accuracy_by_score_margin_bucket"],
        "score_margin_bucket",
    )
    write_stats_csv(out_dir / "accuracy_by_max_tile_bucket.csv", diagnostics["accuracy_by_max_tile_bucket"], "max_tile")
    write_stats_csv(
        out_dir / "accuracy_by_empty_cells_bucket.csv",
        diagnostics["accuracy_by_empty_cells_bucket"],
        "empty_cells",
    )
    write_stats_csv(
        out_dir / "accuracy_by_legal_move_count.csv",
        diagnostics["accuracy_by_legal_move_count"],
        "legal_move_count",
    )

    write_cross_csv(
        out_dir / "cross_gold_action_by_search_depth.csv",
        diagnostics["cross_gold_action_by_search_depth"],
        "gold_action",
        "search_depth",
    )
    write_cross_csv(
        out_dir / "cross_gold_action_by_score_margin_bucket.csv",
        diagnostics["cross_gold_action_by_score_margin_bucket"],
        "gold_action",
        "score_margin_bucket",
    )
    write_cross_csv(
        out_dir / "cross_search_depth_by_score_margin_bucket.csv",
        diagnostics["cross_search_depth_by_score_margin_bucket"],
        "search_depth",
        "score_margin_bucket",
    )
    write_cross_csv(
        out_dir / "cross_gold_action_by_empty_cells_bucket.csv",
        diagnostics["cross_gold_action_by_empty_cells_bucket"],
        "gold_action",
        "empty_cells",
    )
    write_cross_csv(
        out_dir / "cross_gold_action_by_legal_move_count.csv",
        diagnostics["cross_gold_action_by_legal_move_count"],
        "gold_action",
        "legal_move_count",
    )

    samples = sample_bad_cases(records, sample_limit)
    sample_counts: dict[str, int] = {}
    for name, rows in samples.items():
        write_jsonl(samples_dir / f"{name}.jsonl", rows)
        sample_counts[name] = len(rows)
    write_json(out_dir / "sample_counts.json", sample_counts)
    write_markdown_summary(out_dir / "diagnosis_summary.md", diagnostics, sample_counts)
    return diagnostics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    diagnostics = run(args.eval_json, args.bad_cases, args.val_file, args.out_dir, args.sample_limit)
    print(f"Wrote diagnostics to {args.out_dir}")
    print(
        "overall: "
        f"total={diagnostics['overall_metrics']['total']} "
        f"exact_match={diagnostics['overall_metrics']['exact_match']} "
        f"exact_match_rate={diagnostics['overall_metrics']['exact_match_rate']}"
    )
    if diagnostics["warnings"]:
        print("warnings:")
        for warning in diagnostics["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
