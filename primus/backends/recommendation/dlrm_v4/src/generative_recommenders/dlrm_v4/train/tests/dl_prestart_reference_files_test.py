"""CPU tests for the ship15.1b reference-data-only rules (user rule 2026-10-12).

- dl_prestart.reference_data_files() matches stage_data_local.sh REF_FILES and
  the b5e8e2a manifest set (+ item_popularity + shared_metadata mappings).
- page_cache_snapshot opens only reference .npy files (no glob of the data root).
- yambda_5b.gin keeps DL_WINDOW_INDEX=1 (mlcommons/training PR 907 window
  index, allowed by the user 2026-10-12); its files are not opened before
  RUN_START.
- ops/flydsl/verify_integration.py uses a synthetic, deterministic length pool
  and the real_lens_*.npy files are gone.
"""

import ast
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generative_recommenders.dlrm_v4.train import dl_prestart

HERE = Path(__file__).resolve()
REC = HERE.parents[4]  # .../recommendation
GR = REC / "generative_recommenders"
STAGER = REC / "scripts" / "stage_data_local.sh"
GIN = GR / "dlrm_v4" / "train" / "gin" / "yambda_5b.gin"
VERIFY = GR / "ops" / "flydsl" / "verify_integration.py"
YAMBDA = GR / "dlrm_v4" / "datasets" / "yambda.py"

# b5e8e2a:recommendation/sha256sums_yambda_5b_cache.txt entries.
B5E8E2A_CACHE = {
    "_READY", "anchor_ts_L4086.npy", "flat_event_types.npy", "flat_is_like.npy",
    "flat_is_listen_plus.npy", "flat_is_organic.npy", "flat_is_skip.npy",
    "flat_item_ids.npy", "flat_played_ratio.npy", "flat_timestamps.npy",
    "flat_uid.npy", "positions_L4086.npy", "store_meta.json", "unique_uids.npy",
    "user_end.npy", "user_start.npy",
}
# mlcommons/training PR 907 offline time-window index (user-allowed).
PR907_CACHE = {"ts_indices_L4086_W86400.npy", "ts_offsets_L4086_W86400.npy"}


def _stager_ref_files(hist: int) -> list:
    text = STAGER.read_text()
    m = re.search(r"^REF_FILES=\(\n(.*?)^\)", text, re.S | re.M)
    assert m, "REF_FILES array not found in stage_data_local.sh"
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(
            line.strip('"')
            .replace("$CACHE", f"processed_5b/hstu_cache_L{hist}")
            .replace("${HIST}", str(hist))
        )
    return out


class ReferenceFilesTest(unittest.TestCase):
    def test_matches_manifest(self) -> None:
        files = dl_prestart.reference_data_files(4086)
        cache = {f.split("/", 2)[2] for f in files if f.startswith("processed_5b/hstu_cache_L4086/")}
        self.assertEqual(cache, B5E8E2A_CACHE | PR907_CACHE)
        others = {f for f in files if not f.startswith("processed_5b/hstu_cache_L")}
        self.assertEqual(others, {
            "processed_5b/item_popularity.npy",
            "shared_metadata/artist_item_mapping.parquet",
            "shared_metadata/album_item_mapping.parquet",
        })
        names = [os.path.basename(f) for f in files]
        for bad in ("ts_index_manifest", "window_index",
                    "embeddings", "session_index", "test_events", "split_meta", "_lock"):
            self.assertFalse(any(n.startswith(bad) for n in names), bad)
        self.assertFalse(any(re.search(r"_m\d+\.npy$", n) for n in names))

    def test_stager_agrees(self) -> None:
        for hist in (4086, 2039):
            self.assertEqual(sorted(_stager_ref_files(hist)),
                             sorted(dl_prestart.reference_data_files(hist)))

    def test_prestart_subset_excludes_anchor_ts(self) -> None:
        pre = dl_prestart.reference_data_files(4086, prestart_only=True)
        self.assertNotIn("processed_5b/hstu_cache_L4086/anchor_ts_L4086.npy", pre)
        self.assertFalse(any("ts_indices" in f or "ts_offsets" in f for f in pre))
        self.assertEqual(len([f for f in pre if f.endswith(".npy")]), 14)


class SnapshotOpensOnlyReferenceTest(unittest.TestCase):
    def test_only_reference_npy_opened(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root, "processed_5b", "hstu_cache_L4086")
            cache.mkdir(parents=True)
            Path(root, "shared_metadata").mkdir()
            for rel in dl_prestart.reference_data_files(4086):
                np.save(Path(root, rel).with_suffix(".tmp.npy"), np.arange(4096, dtype=np.int64))
                os.replace(Path(root, rel).with_suffix(".tmp.npy"), Path(root, rel))
            for extra in ("positions_L4086_m0.npy",
                          "anchor_ts_L4086_m0.npy", "window_index_L4086_w86400_buckets.npy"):
                np.save(cache / extra, np.arange(4096, dtype=np.int64))
            opened = []
            real_open = os.open

            def spy(path, *a, **k):
                opened.append(os.path.relpath(path, root))
                return real_open(path, *a, **k)

            os.open = spy  # type: ignore[assignment]
            try:
                line = dl_prestart.page_cache_snapshot(root, 0, history_length=4086)
            finally:
                os.open = real_open  # type: ignore[assignment]
            self.assertIsNotNone(line)
            self.assertIn("over 14 .npy files", line)
            want = {f for f in dl_prestart.reference_data_files(4086, prestart_only=True)
                    if f.endswith(".npy")}
            self.assertEqual(set(opened), want)


class GinDefaultTest(unittest.TestCase):
    def test_window_index_default_on(self) -> None:
        text = GIN.read_text()
        self.assertRegex(text, r'swi/env_int\.key\s*=\s*"DL_WINDOW_INDEX"\nswi/env_int\.default\s*=\s*1\n')
        self.assertIn("get_dataset.use_offline_ts_indices = @swi/env_int()", text)


class VerifyIntegrationSyntheticTest(unittest.TestCase):
    def test_no_real_lens(self) -> None:
        self.assertFalse((VERIFY.parent / "data" / "real_lens_dense.npy").exists())
        self.assertFalse((VERIFY.parent / "data" / "real_lens_prod.npy").exists())
        src = VERIFY.read_text()
        self.assertNotIn("np.load(", src)

    def test_synthetic_pool_deterministic(self) -> None:
        # verify_integration imports torch/flydsl at module level (GPU box);
        # exec just its constants and synthetic_lens on CPU.
        tree = ast.parse(VERIFY.read_text())
        keep = [n for n in tree.body if (
            isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("MAX_SEQ_LEN", "MAX_LEN", "MIN_LEN", "LENS_SEED")
                for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "synthetic_lens")]
        ns = {"np": np}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(VERIFY), "exec"), ns)
        a, b = ns["synthetic_lens"](64), ns["synthetic_lens"](64)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.dtype, np.int64)
        self.assertGreaterEqual(int(a.min()), ns["MIN_LEN"])
        self.assertLessEqual(int(a.max()), ns["MAX_LEN"])
        self.assertEqual(ns["MAX_LEN"], 4095)


if __name__ == "__main__":
    unittest.main()
