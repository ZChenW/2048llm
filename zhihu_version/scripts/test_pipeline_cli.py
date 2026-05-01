#!/usr/bin/env python3
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, *, timeout=30):
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def assert_contains(text, needle):
    if needle not in text:
        raise AssertionError(f"expected {needle!r} in output:\n{text}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        eval_bin = tmp_path / "check_2048_rate"
        train_bin = tmp_path / "train_maxtile"

        commands = [
            ["g++", "-O3", "-std=c++17", "check_2048_rate.cpp", "-o", str(eval_bin)],
            ["nvcc", "-O3", "-std=c++17", "train_maxtile.cu", "-o", str(train_bin)],
        ]
        for command in commands:
            result = run(command, timeout=120)
            if result.returncode != 0:
                raise AssertionError(f"build failed: {' '.join(command)}\n{result.stdout}")

        result = run([str(train_bin), "--help"], timeout=5)
        if result.returncode != 0:
            raise AssertionError(result.stdout)
        assert_contains(result.stdout, "--episodes")
        assert_contains(result.stdout, "--out-dir")
        assert_contains(result.stdout, "--resume")

        result = run([str(eval_bin), "--help"], timeout=5)
        if result.returncode != 0:
            raise AssertionError(result.stdout)
        assert_contains(result.stdout, "--weights")
        assert_contains(result.stdout, "--games")
        assert_contains(result.stdout, "--depth")

        source = (ROOT / "train_maxtile.cu").read_text()
        match = re.search(r"bool ensure_dir\(const std::string& path\)\{(?P<body>.*?)\n\}", source, re.S)
        if not match:
            raise AssertionError("ensure_dir function not found")
        body = match.group("body")
        if "current.pop_back()" in body:
            raise AssertionError("ensure_dir must not mutate the accumulated path while trimming slashes")
        assert_contains(body, "std::string dir = current")

    return 0


if __name__ == "__main__":
    sys.exit(main())
