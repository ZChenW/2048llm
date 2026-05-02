# zhihu_version Maxtile Pipeline Notes

## Code Changes

- Added CLI support to `zhihu_version/train_maxtile.cu`:
  - `--help`
  - `--episodes N`
  - `--lr FLOAT`
  - `--seed INT`
  - `--out-dir PATH`
  - `--save-every N`
  - `--resume PATH`
- Kept the original author algorithm intact:
  - tuple patterns unchanged
  - raw float32 weight format unchanged
  - one-step afterstate TD unchanged
  - high-win objective unchanged: truncate at 8192 and use `terminal_reward = max_tile_value`
  - no backward episode TD
- Added checkpoint outputs:
  - `weights_latest.bin`
  - `ckpt_<episodes>.bin`
  - matching `.json` metadata files
- Added CLI support to `zhihu_version/check_2048_rate.cpp`:
  - legacy positional usage still works: `./check_2048_rate weights_latest.bin`
  - new flags: `--weights`, `--games`, `--depth`, `--seed`, `--report-every`
  - old default depth remains 3
- Added `zhihu_version/scripts/eval_checkpoints.py` for checkpoint sweep.
- Added `zhihu_version/scripts/test_pipeline_cli.py` as a lightweight CLI/build regression test.
- Fixed `train_maxtile.cu` path creation after review: `ensure_dir()` no longer mutates the accumulated path while trimming trailing slashes, so nested paths such as `../runs/maxtile_50m` are created correctly instead of malformed names like `..runs`.

## Verification

Build commands run inside `conda activate cs485`:

```bash
g++ -O3 -std=c++17 check_2048_rate.cpp -o check_2048_rate
nvcc -O3 -std=c++17 train_maxtile.cu -o train_maxtile
./train_maxtile --help
./check_2048_rate --help
python3 scripts/test_pipeline_cli.py
```

The `g++` build completed with a linker `.sframe` warning but exit code 0. The CUDA build completed successfully.

Path-regression verification after the `ensure_dir()` fix:

```bash
./train_maxtile --episodes 0 --lr 0.1 --seed 1 --save-every 10000 --out-dir /tmp/td2048_ensure_dir_regression/a/b
```

This produced `/tmp/td2048_ensure_dir_regression/a/b/weights_latest.bin` and `ckpt_0.bin`, both `536870912` bytes, without malformed sibling directories.

## Smoke Test

Command:

```bash
mkdir -p ../runs/maxtile_smoke
./train_maxtile --episodes 10000 --lr 0.1 --seed 1 --save-every 10000 --out-dir ../runs/maxtile_smoke
```

Smoke artifacts:

- `../runs/maxtile_smoke/weights_latest.bin`: `536870912` bytes
- `../runs/maxtile_smoke/ckpt_10000.bin`: `536870912` bytes
- `../runs/maxtile_smoke/weights_latest.json`: present
- `../runs/maxtile_smoke/ckpt_10000.json`: present

Smoke eval:

```bash
./check_2048_rate --weights ../runs/maxtile_smoke/weights_latest.bin --games 100 --depth 1 --seed 1 --report-every 100
```

Result: `0/100` wins at 2048. This is only a pipeline smoke test; the model is intentionally far too early.

## 10M Training And Sweep

Training command:

```bash
mkdir -p ../runs/maxtile_10m
./train_maxtile --episodes 10000000 --lr 0.1 --seed 1 --save-every 1000000 --out-dir ../runs/maxtile_10m
```

Training completed successfully. It produced checkpoints from `ckpt_1000000.bin` through `ckpt_10000000.bin`; each `.bin` file is `536870912` bytes.

Sweep command:

```bash
python3 scripts/eval_checkpoints.py --run-dir ../runs/maxtile_10m --eval-bin ./check_2048_rate
```

Sweep outputs:

- `../runs/maxtile_10m/eval_depth1_1000.csv`
- `../runs/maxtile_10m/eval_top3.csv`

Quick depth-1 sweep over all 10 checkpoints found no 2048 wins:

| Checkpoint | Games | Depth | Wins | Win rate |
| ---------- | ----: | ----: | ---: | -------: |
| 1M         |  1000 |     1 |    0 |   0.0000 |
| 2M         |  1000 |     1 |    0 |   0.0000 |
| 3M         |  1000 |     1 |    0 |   0.0000 |
| 4M         |  1000 |     1 |    0 |   0.0000 |
| 5M         |  1000 |     1 |    0 |   0.0000 |
| 6M         |  1000 |     1 |    0 |   0.0000 |
| 7M         |  1000 |     1 |    0 |   0.0000 |
| 8M         |  1000 |     1 |    0 |   0.0000 |
| 9M         |  1000 |     1 |    0 |   0.0000 |
| 10M        |  1000 |     1 |    0 |   0.0000 |

Because all depth-1 quick results tied, the top-3 script selected 1M, 2M, and 3M by tie-break.

Top-3 final eval:

| Checkpoint | Games | Depth | Wins | Failures | Win rate | Avg steps | Eval time |
| ---------- | ----: | ----: | ---: | -------: | -------: | --------: | --------: |
| 1M         | 10000 |     1 |    0 |    10000 |   0.0000 |    101.58 |     0.09s |
| 1M         | 10000 |     2 |    0 |    10000 |   0.0000 |    223.62 |     3.80s |
| 1M         | 10000 |     3 |    3 |     9997 |   0.0003 |    400.39 |   102.63s |
| 2M         | 10000 |     1 |    0 |    10000 |   0.0000 |    101.63 |     0.20s |
| 2M         | 10000 |     2 |    0 |    10000 |   0.0000 |    223.72 |    17.94s |
| 2M         | 10000 |     3 |    4 |     9996 |   0.0004 |    400.28 |   501.29s |
| 3M         | 10000 |     1 |    0 |    10000 |   0.0000 |    101.67 |     0.21s |
| 3M         | 10000 |     2 |    0 |    10000 |   0.0000 |    223.07 |    10.57s |
| 3M         | 10000 |     3 |    4 |     9996 |   0.0004 |    400.91 |   444.03s |

Best checkpoint in this run: `../runs/maxtile_10m/ckpt_2000000.bin` or `ckpt_3000000.bin` by 3-ply win count, both at `4/10000`. This is not a useful trained model yet.

### 10M Simple Re-eval

Evaluation command template:

```bash
./check_2048_rate \
  --weights CKPT \
  --games 1000 \
  --depth 1 \
  --seed 1 \
  --report-every 1000
```

CSV output:

- `/home/chakew/Projects/llm_rl_2048/runs/maxtile_10m/eval_depth1_1000_rerun.csv`

Depth-1 1000-game rerun results:

| Checkpoint | Games | Depth | Wins | Failures | Win rate | Avg steps | Eval time | Games/sec |
| ---------- | ----: | ----: | ---: | -------: | -------: | --------: | --------: | --------: |
| 1M         |  1000 |     1 |    0 |     1000 |   0.0000 |    102.07 |    0.016s |  62384.58 |
| 2M         |  1000 |     1 |    0 |     1000 |   0.0000 |    101.81 |    0.020s |  50648.72 |
| 3M         |  1000 |     1 |    0 |     1000 |   0.0000 |    100.75 |    0.018s |  56251.58 |
| 4M         |  1000 |     1 |    0 |     1000 |   0.0000 |    102.83 |    0.029s |  34716.14 |
| 5M         |  1000 |     1 |    0 |     1000 |   0.0000 |    102.33 |    0.030s |  33635.47 |
| 6M         |  1000 |     1 |    0 |     1000 |   0.0000 |    102.57 |    0.016s |  64137.60 |
| 7M         |  1000 |     1 |    0 |     1000 |   0.0000 |    100.70 |    0.016s |  64067.94 |
| 8M         |  1000 |     1 |    0 |     1000 |   0.0000 |    100.93 |    0.018s |  54676.10 |
| 9M         |  1000 |     1 |    0 |     1000 |   0.0000 |    102.21 |    0.017s |  59870.84 |
| 10M        |  1000 |     1 |    0 |     1000 |   0.0000 |    102.10 |    0.015s |  67132.86 |

Best checkpoint by wins: `/home/chakew/Projects/llm_rl_2048/runs/maxtile_10m/ckpt_4000000.bin` with `0/1000` wins, win rate `0.0000`.

Conclusion: 10M 阶段仍未出现 depth-1 2048 学习信号，这与之前 sweep 一致。

## Recommendation

Do continue to a 50M diagnostic run, because 10M is much earlier than the author's reported high-win training horizon and shows no depth-1 learning signal yet. Do not commit to 100M/160M until the 50M sweep shows a clear shift in the depth-1 or depth-2 histogram toward 1024/2048.

Rough training-time estimate from checkpoint timestamps: 1M to 10M took about 8 seconds of wall time excluding startup/final overhead. That suggests:

- 50M: roughly 45-60 seconds
- 100M: roughly 90-120 seconds
- 160M: roughly 2.5-3.5 minutes

Evaluation is the actual bottleneck. In this run, 10k depth-3 eval took about 103s to 501s per checkpoint. For future sweeps, use 1k depth-1 for broad filtering, then run 10k depth-2/depth-3 only after depth-1 has nonzero 2048 wins or a much stronger 1024/2048 histogram.

## 50M Diagnostic Run

Training command:

```bash
mkdir -p ../runs/maxtile_50m
./train_maxtile \
  --episodes 50000000 \
  --lr 0.1 \
  --seed 1 \
  --save-every 5000000 \
  --out-dir ../runs/maxtile_50m
```

Training completed before this note update. Checkpoints were written every 5M episodes from `ckpt_5000000.bin` through `ckpt_50000000.bin`, plus `weights_latest.bin`. Every `.bin` file is `536870912` bytes.

Approximate training time from checkpoint mtimes:

- `ckpt_5000000.bin`: 2026-04-30 23:17:02 -0400
- `ckpt_50000000.bin`: 2026-04-30 23:50:05 -0400
- 5M to 50M interval: about 33 minutes; full 50M run was roughly mid-30 minutes including startup.

Depth-1 1000-game sweep:

| Checkpoint | Wins | Failures | Win rate | Avg steps |
| ---------- | ---: | -------: | -------: | --------: |
| 5M         |  109 |      891 |    0.109 |    827.17 |
| 10M        |  300 |      700 |    0.300 |    934.69 |
| 15M        |  519 |      481 |    0.519 |   1021.65 |
| 20M        |  784 |      216 |    0.784 |   1092.91 |
| 25M        |  886 |      114 |    0.886 |   1042.61 |
| 30M        |  943 |       57 |    0.943 |   1005.65 |
| 35M        |  980 |       20 |    0.980 |    989.76 |
| 40M        |  977 |       23 |    0.977 |    976.41 |
| 45M        |  980 |       20 |    0.980 |    971.17 |
| 50M        |  982 |       18 |    0.982 |    975.51 |

Top-3 10000-game eval:

| Checkpoint | Depth | Wins | Failures | Win rate | Avg steps | Eval time |
| ---------- | ----: | ---: | -------: | -------: | --------: | --------: |
| 50M        |     1 | 9832 |      168 |   0.9832 |    973.63 |     1.68s |
| 50M        |     2 | 9924 |       76 |   0.9924 |    967.22 |    85.48s |
| 35M        |     1 | 9760 |      240 |   0.9760 |    983.40 |     3.77s |
| 35M        |     2 | 9883 |      117 |   0.9883 |    973.99 |   205.71s |
| 45M        |     1 | 9765 |      235 |   0.9765 |    971.70 |     3.68s |
| 45M        |     2 | 9898 |      102 |   0.9898 |    967.25 |   199.85s |

CSV outputs:

- `../runs/maxtile_50m/eval_depth1_1000.csv`
- `../runs/maxtile_50m/eval_top3.csv`

Learning signal:

- Clear positive learning signal appeared by 50M.
- Depth-1 improved from `0/1000` at 10M to `982/1000` at 50M.
- Depth-2 improved to `9924/10000` on the 50M checkpoint.
- The histogram shifted strongly from mostly sub-2048 outcomes to mostly exact 2048 early-stop wins.

Best checkpoint:

- `../runs/maxtile_50m/ckpt_50000000.bin`

Depth-3 note:

- Because depth-1/depth-2 showed clear improvement, a 10k depth-3 run was attempted for the 50M checkpoint.
- It exceeded 15 minutes without completing (`check_2048_rate` accumulated about 4h20m CPU time across worker threads), so it was stopped to keep this diagnostic bounded.
- Future depth-3 work should use a smaller sample first, or optimize/cache the evaluator before running 10k depth-3 sweeps.

Recommendation:

- Continue to 100M from the best 50M checkpoint or run a fresh 100M trajectory with the same settings.
- Since 50M already gives depth-1 `98.32%` and depth-2 `99.24%` on 10k games, the training logic is likely working.
- Do not spend time on train-loop debugging before the 100M diagnostic unless a later run regresses.
- For 100M, keep 5M or 10M checkpoints and run depth-1 quick sweep first; only run depth-2 on the top few. Avoid 10k depth-3 until the evaluator is optimized or the sample size is reduced.

## Full Training Run

Additional code/automation change:

- `zhihu_version/scripts/eval_checkpoints.py` was parameterized with `--top-n`, `--quick-output`, `--final-output`, `--quick-report-every`, and `--final-report-every`.
- This only changes sweep orchestration. It does not change `train_maxtile.cu`, tuple patterns, reward, TD update logic, raw weight format, or `train_score.cu`.

### 50M Summary

Best 50M checkpoint:

- `../runs/maxtile_50m/ckpt_50000000.bin`

10k eval:

| Depth | Wins | Failures | Win rate | Avg steps | Eval time |
| ----: | ---: | -------: | -------: | --------: | --------: |
|     1 | 9832 |      168 |   0.9832 |    973.63 |     1.68s |
|     2 | 9924 |       76 |   0.9924 |    967.22 |    85.48s |

Depth-3 10k was attempted for the 50M checkpoint but stopped after exceeding 15 minutes. Depth-3 was deferred until later checkpoints were stronger.

### 100M From 50M

Resume command:

```bash
mkdir -p ../runs/maxtile_100m_from50m
./train_maxtile \
  --resume ../runs/maxtile_50m/ckpt_50000000.bin \
  --episodes 50000000 \
  --lr 0.1 \
  --seed 2 \
  --save-every 5000000 \
  --out-dir ../runs/maxtile_100m_from50m
```

`--episodes` is invocation-local in the current CLI. Therefore `ckpt_50000000.bin` in this directory corresponds to about 100M cumulative episodes. Checkpoints were written every 5M invocation episodes; all `.bin` files are `536870912` bytes.

Approximate timing from checkpoint mtimes:

- `ckpt_5000000.bin`: 2026-05-01 00:30 -0400
- `ckpt_50000000.bin`: 2026-05-01 01:27 -0400
- 5M-to-50M interval: about 57 minutes.

Depth-1 1000-game quick sweep:

| Invocation ckpt | Cumulative est. | Wins | Failures | Win rate | Avg steps |
| --------------: | --------------: | ---: | -------: | -------: | --------: |
|              5M |             55M |  986 |       14 |    0.986 |    969.55 |
|             10M |             60M |  985 |       15 |    0.985 |    967.28 |
|             15M |             65M |  996 |        4 |    0.996 |    964.82 |
|             20M |             70M |  995 |        5 |    0.995 |    958.91 |
|             25M |             75M |  997 |        3 |    0.997 |    961.50 |
|             30M |             80M |  994 |        6 |    0.994 |    958.13 |
|             35M |             85M |  990 |       10 |    0.990 |    957.67 |
|             40M |             90M |  994 |        6 |    0.994 |    956.18 |
|             45M |             95M |  998 |        2 |    0.998 |    954.04 |
|             50M |            100M |  999 |        1 |    0.999 |    954.32 |

Top-5 10k depth-1/depth-2 eval:

| Invocation ckpt | Cumulative est. | Depth | Wins | Failures | Win rate | Avg steps | Eval time |
| --------------: | --------------: | ----: | ---: | -------: | -------: | --------: | --------: |
|             50M |            100M |     1 | 9962 |       38 |   0.9962 |    954.27 |     1.26s |
|             50M |            100M |     2 | 9993 |        7 |   0.9993 |    950.41 |    44.82s |
|             45M |             95M |     1 | 9958 |       42 |   0.9958 |    955.44 |     1.23s |
|             45M |             95M |     2 | 9994 |        6 |   0.9994 |    950.97 |    51.50s |
|             25M |             75M |     1 | 9966 |       34 |   0.9966 |    959.44 |     1.24s |
|             25M |             75M |     2 | 9991 |        9 |   0.9991 |    951.07 |    47.77s |
|             15M |             65M |     1 | 9923 |       77 |   0.9923 |    963.88 |     1.09s |
|             15M |             65M |     2 | 9995 |        5 |   0.9995 |    960.16 |    46.18s |
|             20M |             70M |     1 | 9947 |       53 |   0.9947 |    961.52 |     1.06s |
|             20M |             70M |     2 | 9991 |        9 |   0.9991 |    958.45 |    43.17s |

Top-2 10k depth-3 eval:

| Invocation ckpt | Cumulative est. | Wins | Failures | Win rate | Avg steps | Eval time |
| --------------: | --------------: | ---: | -------: | -------: | --------: | --------: |
|             15M |             65M | 9994 |        6 |   0.9994 |    977.06 |  1124.31s |
|             45M |             95M | 9997 |        3 |   0.9997 |    949.05 |  1286.60s |

Decision:

- 100M clearly improved over 50M.
- For continuation, `../runs/maxtile_100m_from50m/ckpt_45000000.bin` was selected as the resume point because it had the best 10k depth-3 result and lower avg steps, even though `ckpt_15000000.bin` had one fewer depth-2 failure.
- Since this selected checkpoint is about 95M cumulative episodes, the next run used `--episodes 65000000` to reach about 160M cumulative.

CSV outputs:

- `../runs/maxtile_100m_from50m/eval_depth1_1000.csv`
- `../runs/maxtile_100m_from50m/eval_top5_10k_d1_d2.csv`
- `../runs/maxtile_100m_from50m/eval_top2_10k_d3.csv`

### 160M From 100M-Stage Best

Resume command:

```bash
mkdir -p ../runs/maxtile_160m_from100m
./train_maxtile \
  --resume ../runs/maxtile_100m_from50m/ckpt_45000000.bin \
  --episodes 65000000 \
  --lr 0.1 \
  --seed 3 \
  --save-every 5000000 \
  --out-dir ../runs/maxtile_160m_from100m
```

Because the resume checkpoint was about 95M cumulative, `ckpt_65000000.bin` in this directory corresponds to about 160M cumulative. All `.bin` files are `536870912` bytes.

Approximate timing from checkpoint mtimes:

- `ckpt_5000000.bin`: 2026-05-01 02:21 -0400
- `ckpt_65000000.bin`: 2026-05-01 03:40 -0400
- 5M-to-65M interval: about 79 minutes.

Depth-1 1000-game quick sweep:

| Invocation ckpt | Cumulative est. | Wins | Failures | Win rate | Avg steps |
| --------------: | --------------: | ---: | -------: | -------: | --------: |
|              5M |            100M |  993 |        7 |    0.993 |    953.78 |
|             10M |            105M |  996 |        4 |    0.996 |    954.63 |
|             15M |            110M |  997 |        3 |    0.997 |    953.55 |
|             20M |            115M |  997 |        3 |    0.997 |    954.08 |
|             25M |            120M |  997 |        3 |    0.997 |    958.34 |
|             30M |            125M |  999 |        1 |    0.999 |    954.91 |
|             35M |            130M |  999 |        1 |    0.999 |    954.35 |
|             40M |            135M |  993 |        7 |    0.993 |    954.56 |
|             45M |            140M | 1000 |        0 |    1.000 |    953.37 |
|             50M |            145M |  996 |        4 |    0.996 |    952.51 |
|             55M |            150M |  996 |        4 |    0.996 |    955.38 |
|             60M |            155M |  999 |        1 |    0.999 |    957.83 |
|             65M |            160M |  997 |        3 |    0.997 |    952.18 |

Top-5 10k depth-1/depth-2 eval:

| Invocation ckpt | Cumulative est. | Depth | Wins | Failures | Win rate | Avg steps | Eval time |
| --------------: | --------------: | ----: | ---: | -------: | -------: | --------: | --------: |
|             45M |            140M |     1 | 9972 |       28 |   0.9972 |    953.92 |     1.11s |
|             45M |            140M |     2 | 9994 |        6 |   0.9994 |    950.17 |    47.88s |
|             60M |            155M |     1 | 9971 |       29 |   0.9971 |    953.98 |     1.12s |
|             60M |            155M |     2 | 9995 |        5 |   0.9995 |    950.11 |    46.93s |
|             30M |            125M |     1 | 9971 |       29 |   0.9971 |    954.61 |     1.15s |
|             30M |            125M |     2 | 9995 |        5 |   0.9995 |    950.65 |    49.91s |
|             35M |            130M |     1 | 9969 |       31 |   0.9969 |    954.44 |     1.22s |
|             35M |            130M |     2 | 9997 |        3 |   0.9997 |    949.71 |    49.32s |
|             15M |            110M |     1 | 9964 |       36 |   0.9964 |    954.12 |     1.13s |
|             15M |            110M |     2 | 9998 |        2 |   0.9998 |    950.36 |    44.70s |

Top-2 10k depth-3 eval:

| Invocation ckpt | Cumulative est. | Wins | Failures | Win rate | Avg steps | Eval time |
| --------------: | --------------: | ---: | -------: | -------: | --------: | --------: |
|             15M |            110M | 9999 |        1 |   0.9999 |    948.32 |   864.01s |
|             35M |            130M | 9998 |        2 |   0.9998 |    948.33 |  1078.81s |

CSV outputs:

- `../runs/maxtile_160m_from100m/eval_depth1_1000.csv`
- `../runs/maxtile_160m_from100m/eval_top5_10k_d1_d2.csv`
- `../runs/maxtile_160m_from100m/eval_top2_10k_d3.csv`

### Global Best

Global summary CSV:

- `../runs/global_eval_summary.csv`

Top candidates by the requested priority order:

| Rank | Stage         | Invocation ckpt | Cumulative est. | D1 10k | D2 10k |  D3 10k | Checkpoint                                        |
| ---: | ------------- | --------------: | --------------: | -----: | -----: | ------: | ------------------------------------------------- |
|    1 | 160m_from100m |             15M |            110M | 0.9964 | 0.9998 |  0.9999 | `../runs/maxtile_160m_from100m/ckpt_15000000.bin` |
|    2 | 160m_from100m |             35M |            130M | 0.9969 | 0.9997 |  0.9998 | `../runs/maxtile_160m_from100m/ckpt_35000000.bin` |
|    3 | 100m_from50m  |             15M |             65M | 0.9923 | 0.9995 |  0.9994 | `../runs/maxtile_100m_from50m/ckpt_15000000.bin`  |
|    4 | 160m_from100m |             60M |            155M | 0.9971 | 0.9995 | not run | `../runs/maxtile_160m_from100m/ckpt_60000000.bin` |
|    5 | 160m_from100m |             30M |            125M | 0.9971 | 0.9995 | not run | `../runs/maxtile_160m_from100m/ckpt_30000000.bin` |

Selected global best:

- `../runs/maxtile_160m_from100m/ckpt_15000000.bin`

This is not the final 160M checkpoint. It is an earlier checkpoint in the 160M continuation run, about 110M cumulative episodes. This reinforces the earlier rule: do not assume the final checkpoint is best.

### 100k Verification

100k depth-2 command:

```bash
./check_2048_rate --weights ../runs/maxtile_160m_from100m/ckpt_15000000.bin --games 100000 --depth 2 --seed 123 --report-every 10000
```

Output:

- Saved to `../runs/best_global_100k_d2.txt`
- Wins: `99957/100000`
- Failures: `43`
- Win rate: `0.999570000`
- Avg steps: `950.174960`
- Eval time: `555.85s`
- Games/sec: `179.90`

100k depth-3 was triggered because 100k depth-2 exceeded 99.9%.

100k depth-3 command:

```bash
./check_2048_rate --weights ../runs/maxtile_160m_from100m/ckpt_15000000.bin --games 100000 --depth 3 --seed 123 --report-every 10000
```

Output:

- Saved to `../runs/best_global_100k_d3.txt`
- Wins: `99984/100000`
- Failures: `16`
- Win rate: `0.999840000`
- Avg steps: `948.507180`
- Eval time: `12465.93s`
- Games/sec: `8.02`

### 1M Recommendation

The 100k depth-3 result meets both 1M-entry conditions:

- failures <= 20: `16`
- win_rate >= 99.98%: `99.984%`

Recommendation: it is reasonable to run a 1M depth-3 validation next if a final high-confidence claim is needed. Do not run it casually: linear extrapolation from this 100k depth-3 run is about `34.6` hours for 1M depth-3 on the current evaluator. A 1M depth-2 run would be much cheaper, roughly `1.5` hours by the same extrapolation.

Before doing repeated large depth-3 evaluations, consider optimizing `check_2048_rate.cpp` progress flushing and expectimax/transposition-table performance. The training pipeline itself is working; the current bottleneck is evaluation.

### Next Steps

- Keep `../runs/maxtile_160m_from100m/ckpt_15000000.bin` as the current best high-win checkpoint.
- For a stronger final claim, run 1M depth-3 once, not as a sweep.
- If trying to improve further before 1M, resume from the global best with a lower LR such as `0.03` or `0.01` for 20M-50M, but evaluate checkpoints carefully because later checkpoints can regress.
- Multiple seeds are also worth testing, but only after freezing the current best and preserving all CSV summaries.

## Teacher Dataset Export

Current TD layout note:

- The old `gpt_version` directory has been moved to `TD-learning-Teacher/zhihu_version`.
- The user-provided old best-weight path under `/home/chakew/Projects/llm_rl_2048/runs/...` no longer exists after the reorganization.
- The equivalent migrated checkpoint used for freezing was:
  `/home/chakew/Projects/llm_rl_2048/TD-learning-Teacher/runs/maxtile_160m_from100m/ckpt_15000000.bin`

Frozen teacher artifact:

- `/home/chakew/Projects/llm_rl_2048/artifacts/best_td_teacher/best_highwin_td.bin`
- Size: `536870912` bytes
- Source: `best_highwin_td`

Exporter implementation:

- Added `TD-learning-Teacher/zhihu_version/export_teacher.cpp`.
- Reuses the Zhihu evaluator's bitboard, move LUT, tuple patterns, value function, and k-ply expectimax-style chance-node logic.
- Scoring semantics:
  - depth 1 uses `merge_reward + V(afterstate)`.
  - depth 2 uses the existing `check_2048_rate.cpp` expectimax-style score on the afterstate, matching its commented-out merge-reward behavior.
- Supports:
  `--weights`, `--samples`, `--depth`, `--seed`, `--out`, `--max-games`,
  `--min-max-tile`, `--hard-state-ratio`, and `--report-every`.
- JSONL rows include:
  `board`, `valid_moves`, `action_scores`, `action_ranking`, `teacher_action`,
  `max_tile`, `empty_cells`, `legal_move_count`, `top1_score`, `top2_score`,
  `score_margin`, `search_depth`, and `source`.
- Each export also writes a sidecar summary at `<out>.summary.json`.
- Added smoke test:
  `TD-learning-Teacher/zhihu_version/scripts/test_export_teacher.py`.

Build and smoke verification:

```bash
cd /home/chakew/Projects/llm_rl_2048/TD-learning-Teacher/zhihu_version
python scripts/test_export_teacher.py
g++ -O3 -std=c++17 export_teacher.cpp -o export_teacher
./export_teacher --help
```

The C++ link step emitted the same `.sframe` warning seen previously, but exited successfully.

1-ply export command:

```bash
./export_teacher \
  --weights /home/chakew/Projects/llm_rl_2048/artifacts/best_td_teacher/best_highwin_td.bin \
  --samples 100000 \
  --depth 1 \
  --seed 101 \
  --out /home/chakew/Projects/llm_rl_2048/data/teacher_1ply_100k.jsonl \
  --max-games 2000 \
  --min-max-tile 128 \
  --hard-state-ratio 0.35 \
  --report-every 10000
```

1-ply output:

- JSONL: `/home/chakew/Projects/llm_rl_2048/data/teacher_1ply_100k.jsonl`
- Summary: `/home/chakew/Projects/llm_rl_2048/data/teacher_1ply_100k.jsonl.summary.json`
- Count: `100000`
- Games scanned: `31`
- Hard/ambiguous count: `96393` (`0.96393`)
- Max-tile histogram: `128:2097`, `256:4916`, `512:5356`, `1024:14602`, `2048:38218`, `4096:34811`
- Empty-cells histogram: `0:1479`, `1:6054`, `2:14543`, `3:22723`, `4:22980`, `5:17341`, `6:9969`, `7:3770`, `8:959`, `9:151`, `10:29`, `11:2`
- Legal-move-count histogram: `1:41`, `2:7331`, `3:32178`, `4:60450`
- Action distribution: `up:23378`, `down:26385`, `left:26428`, `right:23809`
- Score-margin distribution: `0:64`, `(0,0.001):28`, `[0.001,0.01):113`, `[0.01,0.1):1087`, `[0.1,1):9678`, `[1,10):38726`, `[10,100):35279`, `[100,1000):9186`, `>=1000:5839`

2-ply export command:

```bash
./export_teacher \
  --weights /home/chakew/Projects/llm_rl_2048/artifacts/best_td_teacher/best_highwin_td.bin \
  --samples 50000 \
  --depth 2 \
  --seed 202 \
  --out /home/chakew/Projects/llm_rl_2048/data/teacher_2ply_50k.jsonl \
  --max-games 2000 \
  --min-max-tile 128 \
  --hard-state-ratio 0.35 \
  --report-every 5000
```

2-ply output:

- JSONL: `/home/chakew/Projects/llm_rl_2048/data/teacher_2ply_50k.jsonl`
- Summary: `/home/chakew/Projects/llm_rl_2048/data/teacher_2ply_50k.jsonl.summary.json`
- Count: `50000`
- Games scanned: `16`
- Hard/ambiguous count: `40530` (`0.8106`)
- Max-tile histogram: `128:946`, `256:2613`, `512:1804`, `1024:6037`, `2048:18790`, `4096:19810`
- Empty-cells histogram: `0:912`, `1:3688`, `2:9268`, `3:14063`, `4:11825`, `5:5805`, `6:3132`, `7:1054`, `8:225`, `9:26`, `10:1`, `11:1`
- Legal-move-count histogram: `1:2`, `2:2647`, `3:15508`, `4:31843`
- Action distribution: `up:12688`, `down:11036`, `left:12803`, `right:13473`
- Score-margin distribution: `0:20`, `(0,0.001):32`, `[0.001,0.01):251`, `[0.01,0.1):2367`, `[0.1,1):15790`, `[1,10):21923`, `[10,100):6111`, `[100,1000):2517`, `>=1000:989`

Notes:

- No TD training was run.
- No 1M eval was run.
- No LLM SFT/RL was started.
- `train_maxtile.cu` was not modified.

## SFT Data Preparation

Added scripts:

- `/home/chakew/Projects/llm_rl_2048/scripts/validate_teacher_dataset.py`
  validates teacher JSONL schema, board invariants, legal moves, full
  `action_scores`, full `action_ranking`, top score fields, and prints dataset
  histograms.
- `/home/chakew/Projects/llm_rl_2048/scripts/convert_teacher_to_sft.py`
  converts teacher JSONL into chat-style SFT JSONL for `best_action` or
  `action_ranking`, with `raw_move` or `answer_tag` assistant outputs.
- `/home/chakew/Projects/llm_rl_2048/scripts/test_sft_data_pipeline.py`
  builds a temporary teacher JSONL, validates it, converts all four task/format
  combinations, and checks output counts and assistant formats.

During strict validation, the previously exported teacher JSONL files exposed a
schema bug: rows with illegal moves used compact `action_scores` and
`action_ranking` fields instead of all four actions. No TD training or model
export was rerun. The existing JSONL files were schema-normalized in place by
preserving legal action scores, adding `null` for illegal actions, and appending
illegal actions to `action_ranking`.

Validation command:

```bash
python3 scripts/validate_teacher_dataset.py \
  --input data/teacher_1ply_100k.jsonl \
  --input data/teacher_2ply_50k.jsonl \
  --strict
```

Validation result:

- Total rows: `150000`
- Valid rows: `150000`
- Invalid rows: `0`
- Action distribution: `down:37421`, `left:39231`, `right:37282`, `up:36066`
- Max-tile histogram: `128:3043`, `256:7529`, `512:7160`, `1024:20639`,
  `2048:57008`, `4096:54621`
- Empty-cells histogram: `0:2391`, `1:9742`, `2:23811`, `3:36786`,
  `4:34805`, `5:23146`, `6:13101`, `7:4824`, `8:1184`, `9:177`, `10:30`,
  `11:3`
- Legal-move-count histogram: `1:43`, `2:9978`, `3:47686`, `4:92293`
- Search-depth histogram: `1:100000`, `2:50000`

Conversion commands:

```bash
python3 scripts/convert_teacher_to_sft.py \
  --input data/teacher_1ply_100k.jsonl \
  --input data/teacher_2ply_50k.jsonl \
  --out-dir data/sft \
  --task best_action \
  --format raw_move \
  --val-ratio 0.05 \
  --seed 42

python3 scripts/convert_teacher_to_sft.py \
  --input data/teacher_1ply_100k.jsonl \
  --input data/teacher_2ply_50k.jsonl \
  --out-dir data/sft \
  --task best_action \
  --format answer_tag \
  --val-ratio 0.05 \
  --seed 42

python3 scripts/convert_teacher_to_sft.py \
  --input data/teacher_1ply_100k.jsonl \
  --input data/teacher_2ply_50k.jsonl \
  --out-dir data/sft \
  --task action_ranking \
  --format raw_move \
  --val-ratio 0.05 \
  --seed 42

python3 scripts/convert_teacher_to_sft.py \
  --input data/teacher_1ply_100k.jsonl \
  --input data/teacher_2ply_50k.jsonl \
  --out-dir data/sft \
  --task action_ranking \
  --format answer_tag \
  --val-ratio 0.05 \
  --seed 42
```

Generated SFT files:

- `/home/chakew/Projects/llm_rl_2048/data/sft/best_action_raw_move_train.jsonl`:
  `142500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/best_action_raw_move_val.jsonl`:
  `7500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/best_action_answer_tag_train.jsonl`:
  `142500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/best_action_answer_tag_val.jsonl`:
  `7500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/action_ranking_raw_move_train.jsonl`:
  `142500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/action_ranking_raw_move_val.jsonl`:
  `7500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/action_ranking_answer_tag_train.jsonl`:
  `142500` rows
- `/home/chakew/Projects/llm_rl_2048/data/sft/action_ranking_answer_tag_val.jsonl`:
  `7500` rows

Each conversion also wrote a summary JSON:

- `/home/chakew/Projects/llm_rl_2048/data/sft/best_action_raw_move_summary.json`
- `/home/chakew/Projects/llm_rl_2048/data/sft/best_action_answer_tag_summary.json`
- `/home/chakew/Projects/llm_rl_2048/data/sft/action_ranking_raw_move_summary.json`
- `/home/chakew/Projects/llm_rl_2048/data/sft/action_ranking_answer_tag_summary.json`

All four SFT datasets use the same source distribution:

- Used rows: `150000`
- Train rows: `142500`
- Val rows: `7500`
- Action distribution: `down:37421`, `left:39231`, `right:37282`, `up:36066`
- Max-tile histogram: `128:3043`, `256:7529`, `512:7160`, `1024:20639`,
  `2048:57008`, `4096:54621`
- Empty-cells histogram: `0:2391`, `1:9742`, `2:23811`, `3:36786`,
  `4:34805`, `5:23146`, `6:13101`, `7:4824`, `8:1184`, `9:177`, `10:30`,
  `11:3`
- Search-depth histogram: `1:100000`, `2:50000`

Verification:

```bash
python3 scripts/test_sft_data_pipeline.py
wc -l data/sft/best_action_raw_move_train.jsonl \
  data/sft/best_action_raw_move_val.jsonl \
  data/sft/best_action_answer_tag_train.jsonl \
  data/sft/best_action_answer_tag_val.jsonl \
  data/sft/action_ranking_raw_move_train.jsonl \
  data/sft/action_ranking_raw_move_val.jsonl \
  data/sft/action_ranking_answer_tag_train.jsonl \
  data/sft/action_ranking_answer_tag_val.jsonl
```

Recommended first SFT baseline:

- Primary: `best_action + raw_move`
- Secondary: `best_action + answer_tag`
- Do not use `<think>` for this phase.

No LLM training, RL training, TD training, 1M eval, or `train_maxtile.cu`
changes were performed in this phase.
