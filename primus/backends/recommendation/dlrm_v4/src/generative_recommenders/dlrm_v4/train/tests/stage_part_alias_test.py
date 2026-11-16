#!/usr/bin/env python3
# pyre-strict
"""v2verify regression: a stale <ref>.part that is a hard link or a symlink to the
source file must not be written through (dd's O_TRUNC truncated the source file and,
for a hard link, the size check and the end state then certified the 0-byte copy)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_ref_only_hardening_test import _StagerCase  # noqa: E402


class PartAliasTest(_StagerCase):
    def test_part_link_to_a_source_file_is_not_written_through(self) -> None:
        for kind in ("hard", "sym"):
            with self.subTest(kind=kind):
                src = self.src / "processed_5b" / "item_popularity.npy"
                before = src.read_bytes()
                dest = self.root / f"dest_{kind}"
                part = dest / "processed_5b" / "item_popularity.npy.part"
                part.parent.mkdir(parents=True)
                (os.link if kind == "hard" else os.symlink)(src, part)
                rc, log = self._stage(dest=dest)
                self.assertEqual(src.read_bytes(), before, log)
                self.assertEqual(rc, 0, log)
                self.assertIn("[stage] data dir is reference-only: 21 regular files", log)
                self.assertEqual((dest / "processed_5b" / "item_popularity.npy").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
