#!/usr/bin/env python3
# pyre-strict
"""yambda_5b.gin item (17b): DLRM_HSTU_FP8_STAGE_REF_ONLY (reference-data-only
staging in examples/mlperf/dlrm_v4/data/stage_data_local.sh; knob rule 2026-10-13: every feature is
a gin binding, env only the kill switch, gin default = shipped behaviour).

  GIN   exactly one binding, "= True", right after the item (17) block; the
        item block says it must stay True for an MLPerf submission; no
        apply_env_bootstrap binding is repeated.
  AST   apply_env_bootstrap declares the parameter once (Optional[bool] = None)
        and binds it with one env-wins _bind_bool call.
  BOOT  _env_bootstrap.apply_env_bootstrap in a child interpreter (real gin if
        installed, else a pass-through stub): binding True -> "1"; a pre-set
        "0" wins; binding absent -> unset.
  STAGE the real stager on a synthetic source tree (bytes only, no dataset):
        ON (gin default) stages exactly REF_FILES, verifies them, prunes a
        stale non-reference file, and a missing reference file is FATAL; OFF
        (env 0, or a gin file bound to False) stages the c68395d set
        (whole cache dir + shared_metadata + present processed_5b split files)
        and prunes nothing; an unreadable gin file, a double or non-bool
        binding, and an env value other than 1/0 are FATAL.
CPU only, stdlib only (no torch, no numpy)."""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

KNOB = "DLRM_HSTU_FP8_STAGE_REF_ONLY"
HERE = Path(__file__).resolve()
REC = HERE.parents[4]  # .../recommendation
TRAIN = HERE.parents[1]
GIN = TRAIN / "gin" / "yambda_5b.gin"
BOOT = TRAIN / "_env_bootstrap.py"
# Primus: the stager is examples/mlperf/dlrm_v4/data/ (REC = .../dlrm_v4/src).
STAGER = REC.parents[4] / "examples" / "mlperf" / "dlrm_v4" / "data" / "stage_data_local.sh"
HIST = 4086
CACHE = f"processed_5b/hstu_cache_L{HIST}"
# Plain, scoped and module-qualified forms: gin resolves each to the same
# parameter, so the stager counts them all (dlstart iter49 REVIEW m1).
BIND_RE = re.compile(r"^\s*(?:[\w./]*[./])?apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)


def _gin_lines() -> List[str]:
    return GIN.read_text().splitlines()


def _stager_ref_files(hist: int = HIST) -> List[str]:
    m = re.search(r"^REF_FILES=\(\n(.*?)^\)", STAGER.read_text(), re.S | re.M)
    assert m, "REF_FILES array not found in stage_data_local.sh"
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.strip('"').replace("$CACHE", f"processed_5b/hstu_cache_L{hist}")
                       .replace("${HIST}", str(hist)))
    return out


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_true_and_no_repeated_binding(self) -> None:
        lines = _gin_lines()
        self.assertEqual([m.group(1) for m in map(BIND_RE.match, lines) if m], ["True"])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln)
                                      for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        # The stager cannot follow include/import (an included file could
        # rebind the knob), so it FATALs on them; none may appear here.
        stmt = re.compile(r"^\s*(?:include|import|from)(?:[^\w./]|$)")
        self.assertEqual([ln for ln in lines if stmt.match(ln)], [])

    def test_item_block_follows_item_17_and_documents_the_rule(self) -> None:
        lines = _gin_lines()
        i_bind = next(i for i, ln in enumerate(lines) if BIND_RE.match(ln))
        i_head = max(i for i in range(i_bind) if lines[i].startswith("# (17b) "))
        i_17 = max(i for i in range(i_head) if lines[i].startswith("# (17) MLPerf cache flush"))
        self.assertTrue(all(ln.startswith("#") for ln in lines[i_17:i_bind]))
        block = lines[i_head:i_bind]
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = " ".join(ln.lstrip("#").strip() for ln in block)
        for marker in ("MUST STAY True FOR AN MLPERF SUBMISSION", "the c68395d stager",
                       "REF_FILES", "dl_prestart.reference_data_files()",
                       "NUMERICS: none", "ARMED", "reference-only staging ON",
                       "DLRM_HSTU_FP8_STAGE_REF_ONLY=0", "FATAL"):
            self.assertIn(marker, text)


class BootstrapAstTest(unittest.TestCase):
    def test_parameter_and_bind(self) -> None:
        tree = ast.parse(BOOT.read_text())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        args = fn.args.args
        names = [a.arg for a in args]
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(len(names), len(set(names)))
        i = names.index(KNOB)
        default = fn.args.defaults[i - (len(args) - len(fn.args.defaults))]
        self.assertIsInstance(default, ast.Constant)
        self.assertIsNone(default.value)
        self.assertEqual(ast.unparse(args[i].annotation), "Optional[bool]")
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and any(isinstance(a, ast.Constant) and a.value == KNOB for a in c.args)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), f"_bind_bool('{KNOB}', {KNOB})")


_BOOT_CHILD = textwrap.dedent(
    """
    import importlib.util, json, os, sys, types
    boot, gin_file, mode = sys.argv[1:4]
    try:
        import gin
        real = True
    except ImportError:
        gin = types.ModuleType("gin")
        gin.configurable = lambda f=None, **k: f if f is not None else (lambda g: g)
        sys.modules["gin"] = gin
        real = False
    spec = importlib.util.spec_from_file_location("_eb", boot)
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    if real and mode == "gin":
        gin.parse_config_file(gin_file, skip_unknown=True)
        eb.apply_env_bootstrap()
    elif mode == "absent":
        eb.apply_env_bootstrap()
    else:
        eb.apply_env_bootstrap(DLRM_HSTU_FP8_STAGE_REF_ONLY=True)
    print(json.dumps({"v": os.environ.get("DLRM_HSTU_FP8_STAGE_REF_ONLY")}))
    """
)


class BootstrapRunTest(unittest.TestCase):
    def _run(self, mode: str, preset: Optional[str]) -> Optional[str]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("DLRM_HSTU_FP8_", "TRITON_", "HSTU_ATTN_", "GR_PIN_"))}
        if preset is not None:
            env[KNOB] = preset
        out = subprocess.run([sys.executable, "-c", _BOOT_CHILD, str(BOOT), str(GIN), mode],
                             env=env, capture_output=True, text=True, cwd=str(REC))
        self.assertEqual(out.returncode, 0, out.stderr[-4000:])
        import json
        return json.loads(out.stdout.strip().splitlines()[-1])["v"]

    def test_binding_true_sets_1(self) -> None:
        self.assertEqual(self._run("gin", None), "1")

    def test_preset_0_wins(self) -> None:
        self.assertEqual(self._run("gin", "0"), "0")

    def test_absent_leaves_unset(self) -> None:
        self.assertIsNone(self._run("absent", None))


def _scratch_parent() -> Optional[str]:
    """A directory on a filesystem with > 21 GiB free: the stager wants 20 GiB
    of headroom beyond the missing bytes before it copies anything."""
    for d in (os.environ.get("STAGE_TEST_DIR"), "/dev/shm", "/var/tmp", "/tmp"):
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize > 21 * (1 << 30):
                return d
    return None


# Non-reference files an older stager / offline tooling left at the source.
_EXTRA_SRC = (
    f"{CACHE}/window_index_L{HIST}_w86400_buckets.npy",
    f"{CACHE}/positions_L{HIST}_m0.npy",
    f"{CACHE}/anchor_ts_L{HIST}_m0.npy",
    f"{CACHE}/ts_index_manifest_L{HIST}_W86400.json",
    "shared_metadata/embeddings.parquet",
    "processed_5b/split_meta.json",
    "processed_5b/session_index.parquet",
    "processed_5b/test_events.parquet",
    "processed_5b/train_sessions.parquet",  # never staged in either mode
)
_STALE_DEST = f"{CACHE}/window_index_L{HIST}_w86400_offsets.npy"


def _ship14_set(src: Path) -> Set[str]:
    """The c68395d copy list: every file of the cache dir and shared_metadata,
    plus the four processed_5b files when present."""
    out = {f"{CACHE}/{p.name}" for p in (src / CACHE).iterdir() if p.is_file()}
    out |= {f"shared_metadata/{p.name}" for p in (src / "shared_metadata").iterdir()
            if p.is_file()}
    for f in ("item_popularity.npy", "split_meta.json", "session_index.parquet",
              "test_events.parquet"):
        if (src / "processed_5b" / f).is_file():
            out.add(f"processed_5b/{f}")
    return out


@unittest.skipIf(shutil.which("bash") is None, "bash not available")
class StagerRunTest(unittest.TestCase):
    def setUp(self) -> None:
        parent = _scratch_parent()
        if parent is None:
            self.skipTest("no scratch filesystem with > 21 GiB free")
        self.root = Path(tempfile.mkdtemp(prefix="stage_ref_test_", dir=parent))
        self.src = self.root / "src"
        self.dest = self.root / "dest"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # dest_in_use(): no running container may mount the test dest.
        (self.bin / "docker").write_text("#!/bin/sh\nexit 0\n")
        (self.bin / "docker").chmod(0o755)
        for i, rel in enumerate(_stager_ref_files() + list(_EXTRA_SRC)):
            p = self.src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes([i % 251]) * (100 + 37 * i))
        stale = self.dest / _STALE_DEST
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _stage(self, env_val: Optional[str] = None, gin: Optional[str] = None
               ) -> Tuple[int, str]:
        env = {k: v for k, v in os.environ.items()
               if k not in (KNOB, "STAGE_GIN", "LOCAL_DATA_MOUNT", "HISTORY_LENGTH",
                            "MIN_HISTORY", "STAGE_PRUNE_NONREF", "STAGE_CHECK_TS_INDEX")}
        env.update(DLRM_DATA_SRC=str(self.src), STAGE_CLEAN="0", STAGE_JOBS="2",
                   PATH=f"{self.bin}:{env.get('PATH', '/usr/bin:/bin')}")
        if env_val is not None:
            env[KNOB] = env_val
        if gin is not None:
            env["STAGE_GIN"] = gin
        out = subprocess.run(["bash", str(STAGER), str(self.dest), str(HIST)], env=env,
                             capture_output=True, text=True)
        return out.returncode, out.stdout + out.stderr

    def _dest_set(self) -> Set[str]:
        return {str(p.relative_to(self.dest)) for p in self.dest.rglob("*") if p.is_file()}

    def _gin_file(self, body: str) -> str:
        p = self.root / "test.gin"
        p.write_text(body)
        return str(p)

    def _assert_bytes_equal(self, rels: Set[str]) -> None:
        for rel in rels:
            self.assertEqual((self.src / rel).read_bytes(), (self.dest / rel).read_bytes(), rel)

    def test_on_by_gin_default(self) -> None:
        rc, log = self._stage()
        self.assertEqual(rc, 0, log)
        self.assertIn(f"[stage] reference-only staging ON (gin {GIN} = True)", log)
        self.assertEqual(self._dest_set(), set(_stager_ref_files()))
        self._assert_bytes_equal(set(_stager_ref_files()))
        self.assertIn("[stage] OK", log)

    def test_on_missing_reference_file_is_fatal(self) -> None:
        (self.src / CACHE / "flat_uid.npy").unlink()
        rc, log = self._stage()
        self.assertNotEqual(rc, 0)
        self.assertIn("[stage] FATAL: reference file", log)
        self.assertIn("flat_uid.npy not found", log)
        # OFF (ship14) never required it.
        rc, log = self._stage(env_val="0")
        self.assertEqual(rc, 0, log)

    def test_off_env_restores_ship14_set(self) -> None:
        rc, log = self._stage(env_val="0")
        self.assertEqual(rc, 0, log)
        self.assertIn(f"reference-only staging OFF (env {KNOB}=0 over gin {GIN} = True)", log)
        want = _ship14_set(self.src)
        self.assertNotIn("processed_5b/train_sessions.parquet", want)
        self.assertEqual(self._dest_set(), want | {_STALE_DEST})  # nothing pruned
        self._assert_bytes_equal(want)
        n = len(want)
        self.assertIn(f"[stage] {n} files tracked, {n} to copy (0 already staged)", log)

    def test_off_by_gin_false(self) -> None:
        gin = self._gin_file(f"apply_env_bootstrap.{KNOB} = False\n")
        rc, log = self._stage(gin=gin)
        self.assertEqual(rc, 0, log)
        self.assertIn("reference-only staging OFF", log)
        self.assertEqual(self._dest_set(), _ship14_set(self.src) | {_STALE_DEST})
        # env 1 wins over a False binding.
        rc, log = self._stage(env_val="1", gin=gin)
        self.assertEqual(rc, 0, log)
        self.assertIn("reference-only staging ON", log)
        self.assertEqual(self._dest_set(), set(_stager_ref_files()))

    def test_bad_gin_or_env_is_fatal(self) -> None:
        cases: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            "missing gin": (None, str(self.root / "nope.gin")),
            "no binding": (None, self._gin_file("# nothing\n")),
            "two bindings": (None, self._gin_file(
                f"apply_env_bootstrap.{KNOB} = True\napply_env_bootstrap.{KNOB} = True\n")),
            "non-bool": (None, self._gin_file(f"apply_env_bootstrap.{KNOB} = 1\n")),
            "bad env": ("yes", None),
        }
        for name, (env_val, gin) in cases.items():
            with self.subTest(name):
                rc, log = self._stage(env_val=env_val, gin=gin)
                self.assertNotEqual(rc, 0, log)
                self.assertIn("[stage] FATAL", log)
                self.assertEqual(self._dest_set(), {_STALE_DEST})


if __name__ == "__main__":
    unittest.main()
