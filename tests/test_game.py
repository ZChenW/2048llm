from __future__ import annotations

import json
import unittest

from llm2048.game import Game2048


class Game2048SnapshotTests(unittest.TestCase):
    def test_in_memory_snapshot_is_json_native_and_resumes_identical_future(
        self,
    ) -> None:
        uninterrupted = Game2048(7)
        snapshot = uninterrupted.snapshot()

        self.assertEqual(snapshot, json.loads(json.dumps(snapshot)))
        resumed = Game2048.from_snapshot(snapshot)

        for action in ("LEFT", "UP", "LEFT", "UP"):
            self.assertEqual(
                resumed.move(action),
                uninterrupted.move(action),
            )
            self.assertEqual(resumed.snapshot(), uninterrupted.snapshot())

        self.assertEqual(uninterrupted.score, 12)
        self.assertEqual(
            uninterrupted.board,
            [
                [8, 2, 4, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
