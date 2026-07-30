"""Reproducible, leakage-safe Depth-2 Teacher Policy corpus exports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ACTIONS = ("up", "down", "left", "right")
SPLITS = ("train", "validation", "test")
STRATA = ("natural", "hard", "late")


class CorpusError(ValueError):
    """Raised when a Teacher Policy corpus contract is violated."""


@dataclass(frozen=True)
class CorpusConfig:
    export_name: str
    seed: int
    checkpoint_path: Path
    checkpoint_path_display: str
    checkpoint_sha256: str
    checkpoint_metadata_path: Path
    checkpoint_metadata_path_display: str
    checkpoint_metadata_sha256: str
    exporter_path: Path | None
    exporter_path_display: str | None
    exporter_source_path: Path | None
    exporter_source_path_display: str | None
    search_depth: int
    source: str
    split_targets: dict[str, dict[str, int]]
    teacher_core_count: int
    late_min_max_tile: int
    max_trajectories: int
    input_sha256: str

    @classmethod
    def load(cls, path: Path) -> "CorpusConfig":
        try:
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
        except OSError as error:
            raise CorpusError(f"cannot read corpus configuration: {error}") from error
        except json.JSONDecodeError as error:
            raise CorpusError(
                f"corpus configuration is not valid JSON: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise CorpusError("corpus configuration must be a JSON object")
        required = {
            "schema_version",
            "export_name",
            "seed",
            "checkpoint",
            "teacher_policy",
            "exporter",
            "corpus",
        }
        _require_keys(raw, required, "corpus configuration")
        if raw["schema_version"] != 1:
            raise CorpusError("corpus schema_version must be 1")
        export_name = _nonempty_string(raw["export_name"], "export_name")
        seed = _integer(raw["seed"], "seed", minimum=0)

        checkpoint = _object(raw["checkpoint"], "checkpoint")
        _require_keys(
            checkpoint,
            {"path", "sha256", "metadata_path", "metadata_sha256"},
            "checkpoint",
        )
        checkpoint_display = _nonempty_string(checkpoint["path"], "checkpoint.path")
        checkpoint_sha256 = _sha256(checkpoint["sha256"], "checkpoint.sha256")
        checkpoint_metadata_display = _nonempty_string(
            checkpoint["metadata_path"], "checkpoint.metadata_path"
        )
        checkpoint_metadata_sha256 = _sha256(
            checkpoint["metadata_sha256"], "checkpoint.metadata_sha256"
        )

        teacher_policy = _object(raw["teacher_policy"], "teacher_policy")
        _require_keys(
            teacher_policy, {"search_depth", "source"}, "teacher_policy"
        )
        search_depth = _integer(
            teacher_policy["search_depth"],
            "teacher_policy.search_depth",
            minimum=1,
        )
        if search_depth != 2:
            raise CorpusError("teacher_policy.search_depth must be 2")
        source = _nonempty_string(teacher_policy["source"], "teacher_policy.source")
        if source != "retained_100m_teacher_policy":
            raise CorpusError(
                "teacher_policy.source must be 'retained_100m_teacher_policy'"
            )

        exporter = _object(raw["exporter"], "exporter")
        if set(exporter) not in ({"path"}, {"source_path"}):
            raise CorpusError(
                "exporter must contain exactly one of path or source_path"
            )
        exporter_display = (
            _nonempty_string(exporter["path"], "exporter.path")
            if "path" in exporter
            else None
        )
        exporter_source_display = (
            _nonempty_string(exporter["source_path"], "exporter.source_path")
            if "source_path" in exporter
            else None
        )

        corpus = _object(raw["corpus"], "corpus")
        _require_keys(
            corpus,
            {
                "splits",
                "teacher_core_count",
                "late_min_max_tile",
                "max_trajectories",
            },
            "corpus",
        )
        splits = _object(corpus["splits"], "corpus.splits")
        if set(splits) != set(SPLITS):
            raise CorpusError(
                "corpus.splits must contain exactly train, validation, and test"
            )
        split_targets: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            target = _object(splits[split], f"corpus.splits.{split}")
            if set(target) != set(STRATA):
                raise CorpusError(
                    f"corpus.splits.{split} must contain exactly "
                    "natural, hard, and late"
                )
            split_targets[split] = {
                stratum: _integer(
                    target[stratum],
                    f"corpus.splits.{split}.{stratum}",
                    minimum=0,
                )
                for stratum in STRATA
            }
            if sum(split_targets[split].values()) <= 0:
                raise CorpusError(f"corpus.splits.{split} must not be empty")

        train_total = sum(split_targets["train"].values())
        expected_train = {
            "natural": train_total * 50 // 100,
            "hard": train_total * 30 // 100,
            "late": train_total * 20 // 100,
        }
        if (
            train_total % 10 != 0
            or split_targets["train"] != expected_train
        ):
            raise CorpusError(
                "train targets must be exactly 50% natural, 30% hard, and 20% late"
            )
        teacher_core_count = _integer(
            corpus["teacher_core_count"], "corpus.teacher_core_count", minimum=1
        )
        if teacher_core_count > train_total:
            raise CorpusError("teacher_core_count must not exceed the train count")
        if teacher_core_count % 10 != 0:
            raise CorpusError(
                "teacher_core_count must preserve the 50%/30%/20% train strata"
            )
        late_min_max_tile = _integer(
            corpus["late_min_max_tile"],
            "corpus.late_min_max_tile",
            minimum=1,
        )
        if late_min_max_tile != 512:
            raise CorpusError("corpus.late_min_max_tile must be 512")
        max_trajectories = _integer(
            corpus["max_trajectories"],
            "corpus.max_trajectories",
            minimum=1,
        )

        base = path.resolve().parent
        return cls(
            export_name=export_name,
            seed=seed,
            checkpoint_path=_resolve_path(base, checkpoint_display),
            checkpoint_path_display=checkpoint_display,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_metadata_path=_resolve_path(
                base, checkpoint_metadata_display
            ),
            checkpoint_metadata_path_display=checkpoint_metadata_display,
            checkpoint_metadata_sha256=checkpoint_metadata_sha256,
            exporter_path=(
                _resolve_path(base, exporter_display)
                if exporter_display is not None
                else None
            ),
            exporter_path_display=exporter_display,
            exporter_source_path=(
                _resolve_path(base, exporter_source_display)
                if exporter_source_display is not None
                else None
            ),
            exporter_source_path_display=exporter_source_display,
            search_depth=search_depth,
            source=source,
            split_targets=split_targets,
            teacher_core_count=teacher_core_count,
            late_min_max_tile=late_min_max_tile,
            max_trajectories=max_trajectories,
            input_sha256=sha256(raw_bytes).hexdigest(),
        )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusError(f"{name} must be an object")
    return value


def _require_keys(value: dict[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise CorpusError(f"{name} is missing required fields: {', '.join(missing)}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CorpusError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CorpusError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CorpusError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def _slide(line: list[int]) -> list[int]:
    values = [value for value in line if value]
    result: list[int] = []
    index = 0
    while index < len(values):
        if index + 1 < len(values) and values[index] == values[index + 1]:
            result.append(values[index] * 2)
            index += 2
        else:
            result.append(values[index])
            index += 1
    return result + [0] * (4 - len(result))


def _move(board: list[list[int]], action: str) -> list[list[int]]:
    if action == "left":
        return [_slide(row) for row in board]
    if action == "right":
        return [list(reversed(_slide(list(reversed(row))))) for row in board]
    columns = [[board[row][column] for row in range(4)] for column in range(4)]
    if action == "up":
        moved = [_slide(column) for column in columns]
    else:
        moved = [
            list(reversed(_slide(list(reversed(column))))) for column in columns
        ]
    return [[moved[column][row] for column in range(4)] for row in range(4)]


def _canonical_orbit_id(board: list[list[int]]) -> str:
    def rotate(value: list[list[int]]) -> list[list[int]]:
        return [
            [value[3 - column][row] for column in range(4)]
            for row in range(4)
        ]

    reflected = [list(reversed(row)) for row in board]
    variants: list[list[list[int]]] = []
    for start in (board, reflected):
        value = start
        for _ in range(4):
            variants.append(value)
            value = rotate(value)

    def packed(value: list[list[int]]) -> int:
        result = 0
        for index, tile in enumerate(
            cell for board_row in value for cell in board_row
        ):
            exponent = 0 if tile == 0 else tile.bit_length() - 1
            result |= exponent << (index * 4)
        return result

    return f"{min(packed(value) for value in variants):016x}"


def _validate_board(value: Any, context: str) -> list[list[int]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise CorpusError(f"{context}.board must be a 4x4 matrix")
    board: list[list[int]] = value
    for row in board:
        for tile in row:
            if (
                not isinstance(tile, int)
                or isinstance(tile, bool)
                or tile < 0
                or (tile != 0 and tile & (tile - 1) != 0)
            ):
                raise CorpusError(
                    f"{context}.board tiles must be zero or positive powers of two"
                )
    return board


def _number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise CorpusError(f"{name} must be a finite number")
    return float(value)


def _validate_record(
    row: Any,
    *,
    context: str,
    split: str,
    config: CorpusConfig,
) -> tuple[str, str, str, float, bool]:
    if not isinstance(row, dict):
        raise CorpusError(f"{context} must be a JSON object")
    required = {
        "schema_version",
        "record_id",
        "board",
        "valid_moves",
        "action_scores",
        "action_ranking",
        "teacher_action",
        "max_tile",
        "empty_cells",
        "legal_move_count",
        "top1_score",
        "top2_score",
        "score_margin",
        "search_depth",
        "source",
        "split",
        "stratum",
        "teacher_core",
        "lineage",
    }
    _require_keys(row, required, context)
    if row["schema_version"] != 1:
        raise CorpusError(f"{context}.schema_version must be 1")
    record_id = _nonempty_string(row["record_id"], f"{context}.record_id")
    board = _validate_board(row["board"], context)
    flat_board = [tile for board_row in board for tile in board_row]
    if row["max_tile"] != max(flat_board):
        raise CorpusError(f"{context}.max_tile does not match the board")
    if row["empty_cells"] != sum(tile == 0 for tile in flat_board):
        raise CorpusError(f"{context}.empty_cells does not match the board")

    actual_moves = [action for action in ACTIONS if _move(board, action) != board]
    if row["valid_moves"] != actual_moves:
        raise CorpusError(f"{context}.valid_moves does not match board legality")
    if row["legal_move_count"] != len(actual_moves):
        raise CorpusError(f"{context}.legal_move_count does not match valid_moves")

    scores = row["action_scores"]
    if not isinstance(scores, dict) or set(scores) != set(ACTIONS):
        raise CorpusError(
            f"{context}.action_scores must contain up, down, left, right in order"
        )
    numeric_scores: dict[str, float] = {}
    for action in ACTIONS:
        if action in actual_moves:
            numeric_scores[action] = _number(
                scores[action], f"{context}.action_scores.{action}"
            )
        elif scores[action] is not None:
            raise CorpusError(
                f"{context}.action_scores.{action} must be null for an illegal move"
            )
    ranking = row["action_ranking"]
    if (
        not isinstance(ranking, list)
        or len(ranking) != 4
        or set(ranking) != set(ACTIONS)
    ):
        raise CorpusError(f"{context}.action_ranking must contain every action once")
    legal_ranking = ranking[: len(actual_moves)]
    if set(legal_ranking) != set(actual_moves):
        raise CorpusError(f"{context}.action_ranking must put legal moves first")
    if any(
        numeric_scores[first] < numeric_scores[second]
        for first, second in zip(legal_ranking, legal_ranking[1:])
    ):
        raise CorpusError(
            f"{context}.action_ranking must order legal moves by descending score"
        )
    if row["teacher_action"] != legal_ranking[0]:
        raise CorpusError(f"{context}.teacher_action must be the top-ranked move")

    ordered_scores = sorted(numeric_scores.values(), reverse=True)
    top1 = _number(row["top1_score"], f"{context}.top1_score")
    top2 = _number(row["top2_score"], f"{context}.top2_score")
    margin = _number(row["score_margin"], f"{context}.score_margin")
    expected_top2 = ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]
    if not math.isclose(top1, ordered_scores[0], rel_tol=1e-6, abs_tol=1e-9):
        raise CorpusError(f"{context}.top1_score does not match action_scores")
    if not math.isclose(top2, expected_top2, rel_tol=1e-6, abs_tol=1e-9):
        raise CorpusError(f"{context}.top2_score does not match action_scores")
    if not math.isclose(margin, top1 - top2, rel_tol=1e-6, abs_tol=1e-9):
        raise CorpusError(f"{context}.score_margin must equal top1_score - top2_score")
    if margin < 0:
        raise CorpusError(f"{context}.score_margin must be non-negative")

    if row["search_depth"] != config.search_depth:
        raise CorpusError(f"{context}.search_depth must be 2")
    if row["source"] != config.source:
        raise CorpusError(f"{context}.source does not match the corpus configuration")
    if row["split"] != split:
        raise CorpusError(f"{context}.split does not match its artifact")
    stratum = row["stratum"]
    if stratum not in STRATA:
        raise CorpusError(f"{context}.stratum is unsupported")
    if stratum == "late" and row["max_tile"] < config.late_min_max_tile:
        raise CorpusError(f"{context} late state must have max_tile >= 512")
    if stratum == "hard" and not (
        row["empty_cells"] <= 4
        or row["legal_move_count"] <= 2
        or margin <= 0.0025
    ):
        raise CorpusError(f"{context} hard state does not meet the hard-state rule")
    if not isinstance(row["teacher_core"], bool):
        raise CorpusError(f"{context}.teacher_core must be a boolean")
    if row["teacher_core"] and split != "train":
        raise CorpusError(f"{context} held-out rows cannot be in the Teacher core")

    lineage = _object(row["lineage"], f"{context}.lineage")
    _require_keys(
        lineage,
        {"trajectory_id", "trajectory_step", "orbit_id", "symmetry_id"},
        f"{context}.lineage",
    )
    trajectory_id = _nonempty_string(
        lineage["trajectory_id"], f"{context}.lineage.trajectory_id"
    )
    orbit_id = _nonempty_string(lineage["orbit_id"], f"{context}.lineage.orbit_id")
    expected_orbit_id = _canonical_orbit_id(board)
    if orbit_id != expected_orbit_id:
        raise CorpusError(
            f"{context}.lineage.orbit_id must be canonical D4 id "
            f"{expected_orbit_id}"
        )
    _integer(
        lineage["trajectory_step"], f"{context}.lineage.trajectory_step", minimum=0
    )
    symmetry_id = _integer(
        lineage["symmetry_id"], f"{context}.lineage.symmetry_id", minimum=0
    )
    if symmetry_id != 0:
        raise CorpusError(
            f"{context}.lineage.symmetry_id must be 0; augmentation is online-only"
        )
    return record_id, trajectory_id, orbit_id, margin, row["teacher_core"]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def export_teacher_corpus(
    config_path: Path, output_directory: Path
) -> dict[str, Any]:
    config = CorpusConfig.load(config_path)
    if output_directory.exists():
        raise CorpusError("output directory already exists")
    actual_checkpoint_sha256 = _file_sha256(config.checkpoint_path)
    if actual_checkpoint_sha256 != config.checkpoint_sha256:
        raise CorpusError(
            "Teacher Policy checkpoint checksum does not match the configuration"
        )
    actual_metadata_sha256 = _file_sha256(config.checkpoint_metadata_path)
    if actual_metadata_sha256 != config.checkpoint_metadata_sha256:
        raise CorpusError(
            "Teacher Policy checkpoint metadata checksum does not match "
            "the configuration"
        )
    try:
        checkpoint_metadata = json.loads(
            config.checkpoint_metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read Teacher Policy checkpoint metadata: {error}")
    expected_metadata = {
        "episodes": 100_000_000,
        "seed": 100,
        "mode": "highwin8192",
        "tuple_count": 8,
        "tuple_len": 6,
        "alphabet_size": 16,
    }
    if not isinstance(checkpoint_metadata, dict) or any(
        checkpoint_metadata.get(key) != value
        for key, value in expected_metadata.items()
    ):
        raise CorpusError(
            "checkpoint metadata does not identify the retained 100M "
            "Teacher Policy"
        )
    if (
        not isinstance(checkpoint_metadata.get("weight_file_size"), int)
        or isinstance(checkpoint_metadata["weight_file_size"], bool)
        or checkpoint_metadata["weight_file_size"] <= 0
    ):
        raise CorpusError("checkpoint metadata weight_file_size must be positive")
    try:
        checkpoint_size = config.checkpoint_path.stat().st_size
    except OSError as error:
        raise CorpusError(f"cannot stat Teacher Policy checkpoint: {error}") from error
    if checkpoint_size != checkpoint_metadata["weight_file_size"]:
        raise CorpusError("checkpoint size does not match its metadata")
    if (
        config.exporter_source_path is not None
        and not config.exporter_source_path.is_file()
    ):
        raise CorpusError(
            f"exporter source does not exist: {config.exporter_source_path}"
        )
    if config.exporter_path is not None and not config.exporter_path.is_file():
        raise CorpusError(f"exporter does not exist: {config.exporter_path}")
    output_directory.mkdir(parents=True)
    exporter_metadata: dict[str, Any]
    if config.exporter_source_path is not None:
        built_exporter = output_directory / "export_teacher"
        try:
            build = subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    str(config.exporter_source_path),
                    "-o",
                    str(built_exporter),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise CorpusError(
                f"cannot start the C++ compiler: {error}"
            ) from error
        if build.returncode != 0:
            detail = build.stderr.strip() or build.stdout.strip()
            raise CorpusError(f"cannot build Teacher Policy exporter: {detail}")
        command = [str(built_exporter)]
        exporter_metadata = {
            "source_path": config.exporter_source_path_display,
            "source_sha256": _file_sha256(config.exporter_source_path),
            "binary_sha256": _file_sha256(built_exporter),
            "compiler": "g++ -O3 -std=c++17",
        }
    else:
        assert config.exporter_path is not None
        command = (
            [sys.executable, str(config.exporter_path)]
            if config.exporter_path.suffix == ".py"
            else [str(config.exporter_path)]
        )
        exporter_metadata = {
            "path": config.exporter_path_display,
            "sha256": _file_sha256(config.exporter_path),
        }
    command.extend(
        [
            "--weights",
            str(config.checkpoint_path),
            "--depth",
            str(config.search_depth),
            "--seed",
            str(config.seed),
            "--out-dir",
            str(output_directory),
            "--max-trajectories",
            str(config.max_trajectories),
            "--late-min-max-tile",
            str(config.late_min_max_tile),
            "--teacher-core-count",
            str(config.teacher_core_count),
        ]
    )
    for split in SPLITS:
        target = config.split_targets[split]
        command.extend(
            [
                "--split-target",
                f"{split}:{target['natural']}:{target['hard']}:{target['late']}",
            ]
        )
    try:
        completed = subprocess.run(
            command, capture_output=True, check=False, text=True
        )
    except OSError as error:
        raise CorpusError(f"cannot start Teacher Policy exporter: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CorpusError(
            f"Teacher Policy exporter exited with {completed.returncode}: {detail}"
        )

    counts: dict[str, int] = {}
    strata: dict[str, dict[str, int]] = {}
    artifacts: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    trajectory_splits: dict[str, str] = {}
    orbit_splits: dict[str, str] = {}
    train_positive_margins: list[float] = []
    core_digest = sha256()
    core_member_ids_digest = sha256()
    core_count = 0
    core_strata: Counter[str] = Counter()

    for split in SPLITS:
        artifact_path = output_directory / f"{split}.jsonl"
        split_strata: Counter[str] = Counter()
        split_count = 0
        try:
            lines = artifact_path.read_bytes().splitlines(keepends=True)
        except OSError as error:
            raise CorpusError(f"cannot read {split} artifact: {error}") from error
        for line_number, raw_line in enumerate(lines, 1):
            context = f"{split}.jsonl:{line_number}"
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CorpusError(f"{context} is not valid JSON: {error}") from error
            record_id, trajectory_id, orbit_id, margin, teacher_core = (
                _validate_record(
                    row, context=context, split=split, config=config
                )
            )
            if record_id in record_ids:
                raise CorpusError(f"{context}.record_id is duplicated")
            record_ids.add(record_id)
            for identifier, owners, label in (
                (trajectory_id, trajectory_splits, "trajectory"),
                (orbit_id, orbit_splits, "symmetry orbit"),
            ):
                owner = owners.setdefault(identifier, split)
                if owner != split:
                    raise CorpusError(
                        f"{label} {identifier!r} crosses {owner} and {split}"
                    )
            split_count += 1
            split_strata[row["stratum"]] += 1
            if split == "train" and margin > 0:
                train_positive_margins.append(margin)
            if teacher_core:
                core_count += 1
                core_digest.update(raw_line)
                core_member_ids_digest.update(record_id.encode())
                core_member_ids_digest.update(b"\n")
                core_strata[row["stratum"]] += 1

        expected = config.split_targets[split]
        observed = {stratum: split_strata[stratum] for stratum in STRATA}
        if observed != expected:
            raise CorpusError(
                f"{split} stratum counts do not match the configuration: "
                f"expected {expected}, found {observed}"
            )
        counts[split] = split_count
        strata[split] = observed
        artifacts.append(
            {
                "name": split,
                "path": f"{split}.jsonl",
                "records": split_count,
                "sha256": _file_sha256(artifact_path),
            }
        )

    if core_count != config.teacher_core_count:
        raise CorpusError(
            "Teacher core count does not match the configuration: "
            f"expected {config.teacher_core_count}, found {core_count}"
        )
    expected_core_strata = {
        "natural": core_count * 50 // 100,
        "hard": core_count * 30 // 100,
        "late": core_count * 20 // 100,
    }
    observed_core_strata = {
        stratum: core_strata[stratum] for stratum in STRATA
    }
    if observed_core_strata != expected_core_strata:
        raise CorpusError(
            "Teacher core must preserve the 50%/30%/20% train strata: "
            f"expected {expected_core_strata}, found {observed_core_strata}"
        )
    if not train_positive_margins:
        raise CorpusError("train split has no positive margin for tau calibration")
    tau = statistics.median(train_positive_margins)
    manifest = {
        "schema_version": 1,
        "export": {
            "name": config.export_name,
            "seed": config.seed,
            "configuration_sha256": config.input_sha256,
        },
        "teacher_policy": {
            "source": config.source,
            "search_depth": config.search_depth,
            "checkpoint": {
                "path": config.checkpoint_path_display,
                "sha256": actual_checkpoint_sha256,
                "metadata_path": config.checkpoint_metadata_path_display,
                "metadata_sha256": actual_metadata_sha256,
                "episodes": checkpoint_metadata["episodes"],
                "seed": checkpoint_metadata["seed"],
                "mode": checkpoint_metadata["mode"],
            },
        },
        "corpus": {
            "counts": counts,
            "strata": strata,
            "late_min_max_tile": config.late_min_max_tile,
            "split_unit": "trajectory_and_symmetry_orbit",
            "split_assignment": "splitmix64_seeded_complete_trajectory",
            "symmetry_augmentation": "none_identity_lineage_only",
        },
        "teacher_core": {
            "count": core_count,
            "scope": "train",
            "strata": observed_core_strata,
            "sha256": core_digest.hexdigest(),
            "member_ids_sha256": core_member_ids_digest.hexdigest(),
        },
        "calibration": {
            "method": "median_positive_margin",
            "scope": "train",
            "positive_margin_count": len(train_positive_margins),
            "tau": tau,
        },
        "exporter": exporter_metadata,
        "artifacts": artifacts,
    }
    _write_json(output_directory / "manifest.json", manifest)
    return {
        "counts": counts,
        "manifest": str(output_directory / "manifest.json"),
        "status": "completed",
        "tau": tau,
    }
