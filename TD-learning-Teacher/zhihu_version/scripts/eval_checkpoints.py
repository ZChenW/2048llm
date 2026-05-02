#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_KEYS = {
    "games",
    "wins",
    "failures",
    "win_rate",
    "elapsed_seconds",
    "games_per_sec",
    "avg_steps",
    "hist_json",
}


def parse_episode(path):
    match = re.search(r"ckpt_(\d+)\.bin$", path.name)
    if not match:
        raise ValueError(f"not a checkpoint file: {path}")
    return int(match.group(1))


def parse_summary(output):
    result = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in SUMMARY_KEYS:
            result[key] = value
    missing = SUMMARY_KEYS - set(result)
    if missing:
        raise RuntimeError(f"missing summary keys {sorted(missing)} in output:\n{output}")
    return {
        "games": int(result["games"]),
        "wins": int(result["wins"]),
        "failures": int(result["failures"]),
        "win_rate": float(result["win_rate"]),
        "elapsed_sec": float(result["elapsed_seconds"]),
        "games_per_sec": float(result["games_per_sec"]),
        "avg_steps": float(result["avg_steps"]),
        "hist_json": json.dumps(json.loads(result["hist_json"]), sort_keys=True),
    }


def run_eval(eval_bin, checkpoint, games, depth, seed, report_every):
    cmd = [
        str(eval_bin),
        "--weights",
        str(checkpoint),
        "--games",
        str(games),
        "--depth",
        str(depth),
        "--seed",
        str(seed),
        "--report-every",
        str(report_every),
    ]
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"eval failed with code {completed.returncode}:\n{completed.stdout}")
    return parse_summary(completed.stdout)


def write_csv(path, rows):
    fieldnames = [
        "checkpoint",
        "episode",
        "games",
        "depth",
        "wins",
        "failures",
        "win_rate",
        "elapsed_sec",
        "games_per_sec",
        "avg_steps",
        "hist_json",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate maxtile checkpoints.")
    parser.add_argument("--run-dir", default="../runs/maxtile_10m")
    parser.add_argument("--eval-bin", default="./check_2048_rate")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick-games", type=int, default=1000)
    parser.add_argument("--final-games", type=int, default=10000)
    parser.add_argument("--final-depths", default="1,2,3")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--quick-output", default="eval_depth1_1000.csv")
    parser.add_argument("--final-output", default="eval_top3.csv")
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--quick-report-every", type=int, default=None)
    parser.add_argument("--final-report-every", type=int, default=None)
    args = parser.parse_args()
    final_depths = [int(value) for value in args.final_depths.split(",") if value]
    quick_report_every = args.quick_report_every or args.report_every
    final_report_every = args.final_report_every or args.report_every

    run_dir = Path(args.run_dir)
    eval_bin = Path(args.eval_bin)
    if not eval_bin.is_absolute():
        eval_bin = (Path.cwd() / eval_bin).resolve()
    checkpoints = sorted(run_dir.glob("ckpt_*.bin"), key=parse_episode)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found in {run_dir}")

    quick_rows = []
    for checkpoint in checkpoints:
        episode = parse_episode(checkpoint)
        summary = run_eval(eval_bin, checkpoint, args.quick_games, 1, args.seed, quick_report_every)
        quick_rows.append(
            {
                "checkpoint": str(checkpoint),
                "episode": episode,
                "depth": 1,
                **summary,
            }
        )
    quick_csv = run_dir / args.quick_output
    write_csv(quick_csv, quick_rows)

    def quick_rank(row):
        hist = json.loads(row["hist_json"])
        return (
            -row["win_rate"],
            -hist.get("2048", 0),
            -hist.get("1024", 0),
            -hist.get("512", 0),
            -hist.get("256", 0),
            -row["avg_steps"],
            row["failures"],
            row["episode"],
        )

    top_rows = sorted(quick_rows, key=quick_rank)[: args.top_n]
    print(f"top {args.top_n} checkpoints:")
    for row in top_rows:
        print(f"  {row['checkpoint']} win_rate={row['win_rate']:.9f} failures={row['failures']}")

    final_rows = []
    for row in top_rows:
        checkpoint = Path(row["checkpoint"])
        episode = row["episode"]
        for depth in final_depths:
            summary = run_eval(eval_bin, checkpoint, args.final_games, depth, args.seed, final_report_every)
            final_rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "episode": episode,
                    "depth": depth,
                    **summary,
                }
            )
    final_csv = run_dir / args.final_output
    write_csv(final_csv, final_rows)
    print(f"wrote {quick_csv}")
    print(f"wrote {final_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
