#!/usr/bin/env python3
# pyre-strict
"""Hardening of examples/mlperf/dlrm_v4/data/stage_data_local.sh, yambda_5b.gin item (17b)
DLRM_HSTU_FP8_STAGE_REF_ONLY (dlstart iter49 REVIEW M2/m1/m6/m7, VERIFY V1-V3).

  END STATE  REF_ONLY=1 is FATAL unless the staging root holds exactly
             REF_FILES as regular files, at any depth: files left by
             STAGE_PRUNE_NONREF=0 or while a container mounts the root, an
             unremovable file, deeper or outside files, symlinks, *.part
             files (never pruned) and find errors (unreadable directory,
             counted as NOT verified). REF_ONLY=0 reports the same, rc 0.
  GIN        include/import statements, a second binding in scoped or
             module-qualified form and a binding split before its "=" are
             FATAL before the staging root is touched; a lone qualified
             binding counts as the binding (gin resolves it the same way).
  MIN_HIST   an effective MIN_HISTORY (env when non-empty, else the gin
             default 4086) other than the history length is FATAL with
             REF_ONLY=1 before the staging root is touched; a note with 0.
             It is compared as int() reads it (" 4086", "+4086", "4_086").
  AWK        the stager gives the same result under every awk installed
             (gawk, mawk, busybox awk).
  STATIC     the gin default min_history the stager assumes, the real gin's
             parse inputs, and launch_slurm.sh passing $REPO's gin.
  ALIAS      (v2, VERIFY M1a) a staging root that is, lies inside or contains
             the source (symlink, "//", "..", symlinked data directory) is
             FATAL before anything is touched, --purge included.
  NUL        (v2, VERIFY M1b/m1) names holding a newline are pruned one by
             one (never rm -r), cannot forge the end-state certificate or a
             "[stage]" log line, and never reach shell arithmetic.
  CR         (v2, VERIFY m2/m3) a carriage return is a line break, as in gin;
             an include followed by a backslash continuation is FATAL under
             every awk.
CPU only, stdlib only; synthetic bytes in a scratch directory, never data."""

import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

KNOB = "DLRM_HSTU_FP8_STAGE_REF_ONLY"
HERE = Path(__file__).resolve()
REC = HERE.parents[4]  # .../recommendation
TRAIN = HERE.parents[1]
GIN = TRAIN / "gin" / "yambda_5b.gin"
# Primus: the stager is examples/mlperf/dlrm_v4/data/ (REC = .../dlrm_v4/src).
STAGER = REC.parents[4] / "examples" / "mlperf" / "dlrm_v4" / "data" / "stage_data_local.sh"
LAUNCHER = REC / "scripts" / "launch_slurm.sh"
HIST = 4086
CACHE = f"processed_5b/hstu_cache_L{HIST}"
TRUE = f"apply_env_bootstrap.{KNOB} = True\n"
FALSE = f"apply_env_bootstrap.{KNOB} = False\n"
# Every environment variable the stager reads; each run sets its own.
SCRUB = (KNOB, "STAGE_GIN", "LOCAL_DATA_MOUNT", "HISTORY_LENGTH", "MIN_HISTORY",
         "STAGE_PRUNE_NONREF", "STAGE_CHECK_TS_INDEX", "STAGE_CHECK_TS_INDEX_ARGS",
         "DLRM_DATA_SRC", "STAGE_CLEAN", "STAGE_JOBS", "TMPDIR", "FAKE_DOCKER_MOUNT",
         "TEST_ROOT")
_EXTRA_SRC = (
    f"{CACHE}/window_index_L{HIST}_w86400_buckets.npy",
    f"{CACHE}/positions_L{HIST}_m0.npy",
    "shared_metadata/embeddings.parquet",
    "processed_5b/split_meta.json",
    "processed_5b/train_sessions.parquet",
)
# A test double for docker: no container, or one that bind-mounts
# $FAKE_DOCKER_MOUNT (the stager's dest_in_use() asks `ps -q` and `inspect`).
_FAKE_DOCKER = """#!/bin/sh
[ -n "${FAKE_DOCKER_MOUNT:-}" ] || exit 0
case "$1" in ps) echo c0ffee ;; inspect) echo "$FAKE_DOCKER_MOUNT " ;; esac
exit 0
"""
# Test doubles for the --purge tests: rm refuses (and logs) every path outside
# the test root, `id -un` names a user with no /tmp/<user>, squeue reports every
# job alive. So even a broken guard cannot remove anything outside the root.
_TRIPWIRE_RM = """#!/bin/bash
for a in "$@"; do
  case "$a" in -*) continue ;; esac
  case "$(readlink -m -- "$a")" in "$TEST_ROOT"/*) ;; *)
    echo "$a" >> "$TEST_ROOT/rm_refused.log"; exit 1 ;; esac
done
exec /bin/rm "$@"
"""
_FAKE_ID = """#!/bin/sh
[ "$1" = -un ] && { echo stage_test_no_such_user; exit 0; }
exec /usr/bin/id "$@"
"""
_FAKE_SQUEUE = "#!/bin/sh\necho 1\n"
# readlink that resolves nothing: stands in for an alias that path resolution
# cannot see (a bind mount), so only the device:inode check is left.
_NO_RESOLVE_READLINK = """#!/bin/sh
for a; do last=$a; done
printf '%s\\n' "$last"
"""


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


def _scratch_parent() -> Optional[str]:
    """A directory on a filesystem with > 21 GiB free: the stager wants 20 GiB
    of headroom beyond the missing bytes before it copies anything."""
    for d in (os.environ.get("STAGE_TEST_DIR"), "/dev/shm", "/var/tmp", "/tmp"):
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize > 21 * (1 << 30):
                return d
    return None


def _awks() -> Dict[str, List[str]]:
    """awk implementations installed here, as argv prefixes."""
    out: Dict[str, List[str]] = {}
    for name in ("gawk", "mawk"):
        p = shutil.which(name)
        if p:
            out[name] = [p]
    bb = shutil.which("busybox")
    if bb and subprocess.run([bb, "awk", "BEGIN{exit 0}"], capture_output=True).returncode == 0:
        out["busybox"] = [bb, "awk"]
    return out


class _StagerCase(unittest.TestCase):
    root: Path
    src: Path
    dest: Path

    def setUp(self) -> None:
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        parent = _scratch_parent()
        if parent is None:
            self.skipTest("no scratch filesystem with > 21 GiB free")
        self.root = Path(tempfile.mkdtemp(prefix="stage_hardening_test_", dir=parent))
        self.src = self.root / "src"
        self.dest = self.root / "dest"
        self.tmp = self.root / "tmp"
        self.tmp.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(_FAKE_DOCKER)
        (self.bin / "docker").chmod(0o755)
        for name, argv in _awks().items():
            d = self.root / f"awk-{name}"
            d.mkdir()
            (d / "awk").write_text("#!/bin/sh\nexec %s \"$@\"\n" % " ".join(argv))
            (d / "awk").chmod(0o755)
        for i, rel in enumerate(_stager_ref_files() + list(_EXTRA_SRC)):
            p = self.src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes([i % 251]) * (100 + 37 * i))
        self.gin_n = 0

    def tearDown(self) -> None:
        for d, dirs, _ in os.walk(self.root):  # undo the tests' chmods, top-down
            for x in dirs:
                p = os.path.join(d, x)
                if not os.path.islink(p):
                    os.chmod(p, 0o755)
        shutil.rmtree(self.root, ignore_errors=True)

    def _gin(self, body: str) -> str:
        self.gin_n += 1
        p = self.root / f"test{self.gin_n}.gin"
        p.write_bytes(body.encode())
        return str(p)

    def _run(self, args: List[str], env: Optional[Dict[str, str]] = None,
             gin: Optional[str] = None, awk: Optional[str] = None,
             shims: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        e = {k: v for k, v in os.environ.items() if k not in SCRUB}
        path = f"{self.bin}:{e.get('PATH', '/usr/bin:/bin')}"
        if awk is not None:
            path = f"{self.root / ('awk-' + awk)}:{path}"
        if shims:
            d = Path(tempfile.mkdtemp(prefix="shims_", dir=self.root))
            for name, body in shims.items():
                (d / name).write_text(body)
                (d / name).chmod(0o755)
            path = f"{d}:{path}"
        e.update(DLRM_DATA_SRC=str(self.src), STAGE_CLEAN="0", STAGE_JOBS="2",
                 TMPDIR=str(self.tmp), PATH=path, TEST_ROOT=str(self.root))
        if gin is not None:
            e["STAGE_GIN"] = gin
        e.update(env or {})
        out = subprocess.run(["bash", str(STAGER)] + args, env=e, capture_output=True,
                             encoding="utf-8", errors="replace")
        return out.returncode, out.stdout + out.stderr

    def _stage(self, env: Optional[Dict[str, str]] = None, gin: Optional[str] = None,
               awk: Optional[str] = None, dest: Optional[Path] = None,
               hist: int = HIST) -> Tuple[int, str]:
        return self._run([str(dest or self.dest), str(hist)], env=env, gin=gin, awk=awk)

    @staticmethod
    def _snapshot(root: Path) -> Dict[str, Tuple[str, int]]:
        """Every entry under root (type, size of regular files), symlinks not followed."""
        out: Dict[str, Tuple[str, int]] = {}
        for d, dirs, files in os.walk(root):
            for name in dirs + files:
                st = os.lstat(os.path.join(d, name))
                kind = ("l" if stat.S_ISLNK(st.st_mode) else
                        "d" if stat.S_ISDIR(st.st_mode) else "f")
                out[os.path.relpath(os.path.join(d, name), root)] = (
                    kind, st.st_size if kind == "f" else 0)
        return out

    def _tree(self, root: Optional[Path] = None) -> Set[str]:
        """Every non-directory entry under root (symlinks included), relative."""
        root = root or self.dest
        out = set()
        for d, dirs, files in os.walk(root):
            for f in files:
                out.add(os.path.relpath(os.path.join(d, f), root))
            for f in dirs:
                if os.path.islink(os.path.join(d, f)):
                    out.add(os.path.relpath(os.path.join(d, f), root))
        return out

    def _staged_clean(self) -> None:
        rc, log = self._stage()
        self.assertEqual(rc, 0, log)
        self.assertEqual(self._tree(), set(_stager_ref_files()))

    def _put(self, rel: str, data: bytes = b"leftover") -> Path:
        p = self.dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def assertFatalEndState(self, rc: int, log: str, *extras: str) -> None:
        self.assertNotEqual(rc, 0, log)
        self.assertIn("[stage] data dir NOT reference-only", log)
        self.assertIn(f"[stage] FATAL: {KNOB}=1 needs exactly the reference set", log)
        self.assertNotIn("[stage] data dir is reference-only", log)
        self.assertNotIn("[stage] OK", log)
        for x in extras:
            self.assertIn(x, log)

    def assertReportOnly(self, rc: int, log: str) -> None:
        self.assertEqual(rc, 0, log)
        self.assertIn("[stage] end state (report only, reference-only staging OFF)", log)
        self.assertNotIn("FATAL", log)
        self.assertIn("[stage] OK", log)


class EndStateTest(_StagerCase):
    def test_on_clean_is_verified_and_idempotent(self) -> None:
        rc, log = self._stage()
        self.assertEqual(rc, 0, log)
        self.assertIn(f"[stage] reference-only staging ON (gin {GIN} = True)", log)
        self.assertIn(f"[stage] effective min_history {HIST} = history_length {HIST}", log)
        self.assertIn(f"[stage] data dir is reference-only: 21 regular files under {self.dest}, "
                      "exactly REF_FILES", log)
        self.assertEqual(self._tree(), set(_stager_ref_files()))
        rc, log = self._stage()
        self.assertEqual(rc, 0, log)
        self.assertIn("[stage] 21 files tracked, 0 to copy (21 already staged)", log)
        self.assertIn("[stage] data dir is reference-only: 21 regular files", log)

    def test_off_keeps_c68395d_message_and_only_reports(self) -> None:
        self._put(f"{CACHE}/sub/deep.npy")
        rc, log = self._stage(env={KNOB: "0"})
        self.assertReportOnly(rc, log)
        self.assertIn(f"[stage] reference-only staging OFF (env {KNOB}=0 over gin {GIN} = True): "
                      "c68395d full-cache staging, NOT for an MLPerf submission", log)
        self.assertNotIn("effective min_history", log)
        self.assertIn(f"{CACHE}/sub/deep.npy", self._tree())

    def test_missing_reference_file_is_fatal_before_copy(self) -> None:
        (self.src / CACHE / "flat_uid.npy").unlink()
        rc, log = self._stage()
        self.assertNotEqual(rc, 0, log)
        self.assertIn("[stage] FATAL: reference file", log)
        self.assertEqual(self._tree(), set())  # (only the empty data dirs exist)

    def test_extra_files_pruned_by_default(self) -> None:
        self._staged_clean()
        for rel in (f"{CACHE}/window_index_x.npy", "processed_5b/split_meta.json",
                    "shared_metadata/embeddings.parquet"):
            self._put(rel)
        rc, log = self._stage()
        self.assertEqual(rc, 0, log)
        self.assertEqual(log.count("— non-reference file"), 3, log)
        self.assertIn("[stage] data dir is reference-only: 21 regular files", log)
        self.assertEqual(self._tree(), set(_stager_ref_files()))

    def test_prune_off_leftover_is_fatal_on_reported_off(self) -> None:
        self._staged_clean()
        self._put("processed_5b/split_meta.json")
        rc, log = self._stage(env={"STAGE_PRUNE_NONREF": "0"})
        self.assertFatalEndState(rc, log, "(STAGE_PRUNE_NONREF=0;",
                                 "[stage]   extra: processed_5b/split_meta.json")
        self.assertIn("processed_5b/split_meta.json", self._tree())
        rc, log = self._stage(env={"STAGE_PRUNE_NONREF": "0", KNOB: "0"})
        self.assertReportOnly(rc, log)

    def test_mounted_dest_leftover_is_fatal(self) -> None:
        self._staged_clean()
        rc, log = self._stage(env={"FAKE_DOCKER_MOUNT": str(self.dest)})
        self.assertEqual(rc, 0, log)  # a mounted, clean root is fine
        self._put(f"{CACHE}/positions_L{HIST}_m0.npy")
        rc, log = self._stage(env={"FAKE_DOCKER_MOUNT": str(self.dest)})
        self.assertFatalEndState(rc, log, "a running container mounts it, so pruning is skipped",
                                 f"[stage]   extra: {CACHE}/positions_L{HIST}_m0.npy")
        self.assertIn(f"{CACHE}/positions_L{HIST}_m0.npy", self._tree())

    def test_part_file_is_never_pruned(self) -> None:
        self._staged_clean()
        inflight = self._put(f"{CACHE}/flat_uid.npy.part", b"in flight")
        other = self._put("shared_metadata/embeddings.parquet.part", b"in flight 2")
        rc, log = self._stage()
        self.assertFatalEndState(rc, log, f"[stage]   extra: {CACHE}/flat_uid.npy.part",
                                 "[stage]   extra: shared_metadata/embeddings.parquet.part")
        self.assertNotIn(".part (~", log)  # no "removed ….part" line
        self.assertEqual(inflight.read_bytes(), b"in flight")
        self.assertEqual(other.read_bytes(), b"in flight 2")
        rc, log = self._stage(env={KNOB: "0"})
        self.assertReportOnly(rc, log)
        self.assertEqual(inflight.read_bytes(), b"in flight")

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_unremovable_file_is_loud_and_fatal(self) -> None:
        # Stand-in for a root-owned leftover: our own file in a read-only
        # directory, so drop()'s rm fails ("could not remove"). OFF first, so
        # that no later run needs to write into that directory.
        rc, log = self._stage(env={KNOB: "0"})
        self.assertEqual(rc, 0, log)
        self._put("processed_5b/stale_from_root.npy")
        os.chmod(self.dest / "processed_5b", 0o555)
        rc, log = self._stage(env={KNOB: "0"})
        self.assertReportOnly(rc, log)
        rc, log = self._stage()
        self.assertFatalEndState(rc, log, "[stage]   could not remove",
                                 "[stage]   extra: processed_5b/stale_from_root.npy",
                                 "[stage]   extra: processed_5b/split_meta.json")
        self.assertTrue({"processed_5b/stale_from_root.npy", "processed_5b/split_meta.json"}
                        <= self._tree())

    def test_symlinks_are_not_reference_files(self) -> None:
        self._staged_clean()
        os.symlink("nowhere", self.dest / "processed_5b" / "dangling.npy")
        # A reference name that is a symlink whose own size (`stat -c %s`
        # does not follow links) equals the source size: the size check and
        # the verify step both pass it; only the end state sees it.
        pop = self.dest / "processed_5b" / "item_popularity.npy"
        size = (self.src / "processed_5b" / "item_popularity.npy").stat().st_size
        pop.unlink()
        os.symlink("x" * size, pop)
        rc, log = self._stage()
        self.assertIn("[stage] 21 files tracked, 0 to copy (21 already staged)", log)
        self.assertNotIn("MISMATCH", log)
        self.assertFatalEndState(rc, log, "[stage]   extra: processed_5b/dangling.npy (find type l)",
                                 "[stage]   extra: processed_5b/item_popularity.npy (find type l)",
                                 "[stage]   missing or not regular: processed_5b/item_popularity.npy")

    def test_deep_and_outside_files_are_fatal_on_reported_off(self) -> None:
        self._staged_clean()
        rels = (f"{CACHE}/sub/x.npy", "processed_5b/hstu_cache_L2039/flat_uid.npy",
                "other/readme.txt", "at_root.txt")
        for rel in rels:
            self._put(rel)
        rc, log = self._stage()
        self.assertFatalEndState(rc, log, *(f"[stage]   extra: {r}" for r in rels))
        self.assertTrue(set(rels) <= self._tree())
        rc, log = self._stage(env={KNOB: "0"})
        self.assertReportOnly(rc, log)

    def test_more_than_20_leftovers_are_counted(self) -> None:
        self._staged_clean()
        for i in range(30):
            self._put(f"other/f{i:02d}.npy")
        rc, log = self._stage()
        self.assertFatalEndState(rc, log, "51 entries under", "30 not in REF_FILES",
                                 "[stage]   extra: ... and 10 more")

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_find_error_is_not_verified(self) -> None:
        self._staged_clean()
        self._put(f"{CACHE}/locked/hidden.npy")
        os.chmod(self.dest / CACHE / "locked", 0)
        rc, log = self._stage()
        self.assertFatalEndState(rc, log, "find rc=1 with 1 error line(s)", "[stage]   find: ",
                                 "Permission denied")
        self.assertIn("0 not in REF_FILES", log)  # the hidden file is not listed ...
        os.chmod(self.dest / CACHE / "locked", 0o755)
        os.unlink(self.dest / CACHE / "locked" / "hidden.npy")
        os.chmod(self.dest / CACHE / "locked", 0)  # ... and an empty one still fails
        rc, log = self._stage()
        self.assertFatalEndState(rc, log, "find rc=1")
        rc, log = self._stage(env={KNOB: "0"})
        self.assertReportOnly(rc, log)
        self.assertIn("find rc=1 with 1 error line(s)", log)


class GinParseTest(_StagerCase):
    def _assert_fatal_untouched(self, body: str, want: str, env: Optional[Dict[str, str]] = None
                                ) -> None:
        rc, log = self._stage(gin=self._gin(body), env=env)
        self.assertNotEqual(rc, 0, log)
        self.assertIn("[stage] FATAL", log)
        self.assertIn(want, log)
        self.assertFalse(self.dest.exists(), log)

    def _assert_mode(self, body: str, mode: str) -> None:
        rc, log = self._stage(gin=self._gin(body))
        self.assertEqual(rc, 0, log)
        self.assertIn(f"[stage] reference-only staging {mode} (gin ", log)

    def test_include_and_import_are_fatal(self) -> None:
        for body in (TRUE + "include 'x.gin'\n", "include \"x.gin\"\n" + TRUE,
                     TRUE + "include'x.gin'\n", TRUE + "  include 'x.gin'\r\n",
                     "from __gin__ import dynamic_registration\n" + TRUE,
                     TRUE + "import generative_recommenders.dlrm_v4.train._env_bootstrap\n",
                     FALSE + "include 'binds_true.gin'\n"):
            with self.subTest(body=body):
                self._assert_fatal_untouched(body, "gin include/import statement")
        # The env value decides the mode but does not excuse the gin file.
        self._assert_fatal_untouched(TRUE + "include 'x.gin'\n", "include/import", {KNOB: "1"})

    def test_include_like_names_and_comments_are_not_statements(self) -> None:
        self._assert_mode(TRUE + "include_x.param = 1\nfrom_y.z = 2\nimportant.k = 3\n"
                          "# include 'x.gin'\n#import y\n", "ON")

    def test_second_binding_in_any_form_is_fatal(self) -> None:
        for extra in ("_env_bootstrap.apply_env_bootstrap.%s = False",
                      "train._env_bootstrap.apply_env_bootstrap.%s = False",
                      "generative_recommenders.dlrm_v4.train._env_bootstrap."
                      "apply_env_bootstrap.%s = False",
                      "stage/apply_env_bootstrap.%s = False",
                      "  _env_bootstrap.apply_env_bootstrap.%s=True"):
            with self.subTest(extra=extra):
                self._assert_fatal_untouched(TRUE + extra % KNOB + "\n", "(found: <2 bindings>)")
        self._assert_fatal_untouched(
            FALSE + "_env_bootstrap.apply_env_bootstrap.%s = True\n" % KNOB, "<2 bindings>")

    def test_lone_qualified_binding_counts(self) -> None:
        self._assert_mode("_env_bootstrap.apply_env_bootstrap.%s = True\n" % KNOB, "ON")
        self.dest = self.root / "dest_off"
        self._assert_mode("generative_recommenders.dlrm_v4.train._env_bootstrap."
                          "apply_env_bootstrap.%s = False\n" % KNOB, "OFF")

    def test_split_binding_and_other_mentions_are_fatal(self) -> None:
        self._assert_fatal_untouched(TRUE + f"apply_env_bootstrap.{KNOB} \\\n    = False\n",
                                     "<1 bindings + 1 other non-comment mention(s)>")
        self._assert_fatal_untouched(TRUE + f'DOC = "{KNOB}"\n', "other non-comment mention")
        # Comments and longer names are not mentions.
        self._assert_mode(f"# {KNOB} = False\n" + TRUE + f"x.y = 1  # {KNOB}\n"
                          f"apply_env_bootstrap.{KNOB}_STRICT = False\n", "ON")

    def test_previous_fatal_cases_still_fatal(self) -> None:
        for body, want in (("", "<0 bindings>"), ("# " + TRUE, "<0 bindings>"),
                           (TRUE + TRUE, "<2 bindings>"),
                           (TRUE.replace("True", "true"), "(found: true)"),
                           (TRUE.replace("True", "True  # keep"), "(found: True  # keep)"),
                           (TRUE.replace("True", "\\\n    True"), "(found: \\)")):
            with self.subTest(body=body):
                self._assert_fatal_untouched(body, want)
        self._assert_fatal_untouched(TRUE, "not 1 or 0", {KNOB: "true"})

    def test_carriage_return_is_a_line_break_as_in_gin(self) -> None:
        # VERIFY m2: gin opens the file with universal newlines, so a lone CR
        # ends a line there; it must not hide an include or a binding here.
        self._assert_fatal_untouched(TRUE + "# note\rinclude 'x.gin'\n",
                                     "gin include/import statement")
        self._assert_fatal_untouched(FALSE + "# note\r" + TRUE, "(found: <2 bindings>)")
        self._assert_fatal_untouched(TRUE + f"apply_env_bootstrap.{KNOB}\r= False\n",
                                     "<1 bindings + 1 other non-comment mention(s)>")
        self._assert_mode("# c\r" + TRUE.replace("\n", "\r"), "ON")  # a CR-only file
        self.dest = self.root / "dest_off"
        self._assert_mode("# c\r\n" + FALSE.replace("\n", "\r\n"), "OFF")  # CR LF

    def test_include_with_backslash_continuation_is_fatal(self) -> None:
        # VERIFY m3: busybox awk read "\/" in a bracket expression as a backslash.
        for body in (TRUE + "include\\\n'x.gin'\n", TRUE + "include\\\r\n'x.gin'\r\n",
                     "from\\\n__gin__ import dynamic_registration\n" + TRUE):
            with self.subTest(body=body):
                self._assert_fatal_untouched(body, "gin include/import statement")


class PathAliasTest(_StagerCase):
    """VERIFY M1a: a staging root that is, lies inside or contains the source
    is FATAL before anything is touched. A string compare let 734246a's prune
    delete the source's non-reference files through such a root."""

    def _assert_refused(self, argv: List[str], want: str = "[stage] FATAL: refusing to use",
                        shims: Optional[Dict[str, str]] = None) -> str:
        before = self._snapshot(self.src)
        rc, log = self._run(argv, shims=shims)
        self.assertNotEqual(rc, 0, log)
        self.assertIn("[stage] FATAL: refusing to use", log)
        self.assertIn(want, log)
        self.assertNotIn("[stage] host=", log)
        self.assertNotIn("purge:", log)
        self.assertEqual(self._snapshot(self.src), before)
        return log

    def test_symlink_to_the_source_is_fatal(self) -> None:
        link = self.root / "alias"
        os.symlink(self.src, link)
        for dest in (str(link), str(link) + "/"):
            with self.subTest(dest=dest):
                self._assert_refused([dest, str(HIST)], "lies inside or contains the source")

    def test_other_spellings_of_the_source_are_fatal(self) -> None:
        for dest in (f"{self.root}//src", f"{self.root}/./src", f"{self.root}/tmp/../src",
                     f"{self.src}/.", f"{self.src}/"):
            with self.subTest(dest=dest):
                self._assert_refused([dest, str(HIST)])

    def test_dest_inside_or_containing_the_source_is_fatal(self) -> None:
        link = self.root / "srclink"
        os.symlink(self.src, link)
        up = self.root / "uplink"
        os.symlink(self.root, up)
        for dest, want in ((self.src / "stage_here", "not scratch"),
                           (link / "stage_here", "lies inside or contains the source"),
                           (self.root, "lies inside or contains the source"),
                           (up, "lies inside or contains the source")):
            with self.subTest(dest=dest):
                self._assert_refused([str(dest), str(HIST)], want)
        self.assertFalse((self.src / "stage_here").exists())

    def test_purge_of_the_source_is_fatal(self) -> None:
        link = self.root / "alias"
        os.symlink(self.src, link)
        shims = {"rm": _TRIPWIRE_RM, "id": _FAKE_ID, "squeue": _FAKE_SQUEUE}
        for dest in (str(link), str(link) + "/", f"{self.root}//src"):
            with self.subTest(dest=dest):
                self._assert_refused(["--purge", dest], "the source", shims=shims)
        self.assertFalse((self.root / "rm_refused.log").exists())

    def test_symlinked_data_directory_is_fatal_in_either_mode(self) -> None:
        self.dest.mkdir()
        os.symlink(self.src / "processed_5b", self.dest / "processed_5b")
        before = self._snapshot(self.src)
        want = (f"[stage] FATAL: {self.dest}/processed_5b is a symlink or lies below one "
                f"(resolves to {os.path.realpath(self.src / 'processed_5b')})")
        for env in ({}, {KNOB: "0"}):
            with self.subTest(env=env):
                rc, log = self._stage(env=env)
                self.assertNotEqual(rc, 0, log)
                self.assertIn(want, log)
                self.assertEqual(self._snapshot(self.src), before)
        os.unlink(self.dest / "processed_5b")
        other = self.root / "elsewhere"
        other.mkdir()
        os.symlink(other, self.dest / "shared_metadata")
        rc, log = self._stage()
        self.assertNotEqual(rc, 0, log)
        self.assertIn(f"[stage] FATAL: {self.dest}/shared_metadata is a symlink or lies below "
                      f"one (resolves to {os.path.realpath(other)})", log)
        self.assertEqual(list(other.iterdir()), [])
        os.unlink(self.dest / "shared_metadata")
        os.symlink("shared_metadata", self.dest / "shared_metadata")  # a loop: resolves nowhere
        rc, log = self._stage()
        self.assertNotEqual(rc, 0, log)
        self.assertIn(f"[stage] FATAL: {self.dest}/shared_metadata is a symlink", log)

    def test_symlinked_root_elsewhere_and_trailing_slash_still_work(self) -> None:
        real = self.root / "real_dest"
        real.mkdir()
        link = self.root / "dest_link"
        os.symlink(real, link)
        rc, log = self._stage(dest=link)
        self.assertEqual(rc, 0, log)
        self.assertIn("[stage] data dir is reference-only: 21 regular files", log)
        self.assertEqual(self._tree(real), set(_stager_ref_files()))
        rc, log = self._run([str(self.dest) + "/", str(HIST)])
        self.assertEqual(rc, 0, log)
        self.assertIn("[stage] data dir is reference-only: 21 regular files", log)

    def test_device_inode_check_catches_what_path_resolution_misses(self) -> None:
        link = self.root / "alias"
        os.symlink(self.src, link)
        up = self.root / "uplink"  # the source's parent under another name
        os.symlink(self.root, up)
        for dest in (link, up):
            with self.subTest(dest=dest):
                self._assert_refused([str(dest), str(HIST)], "same directory by device:inode",
                                     shims={"readlink": _NO_RESOLVE_READLINK})


class NulSafetyTest(_StagerCase):
    """VERIFY M1b and m1: newline-bearing names in the prune and end state."""

    def test_newline_names_are_pruned_one_by_one(self) -> None:
        self._staged_clean()
        for name in ("x\nshared_metadata", "x\n", "q\na[$(echo INJECTED >&2)]", "a b.npy"):
            self._put(f"processed_5b/{name}")
        rc, log = self._stage()
        self.assertEqual(rc, 0, log)
        self.assertNotIn("INJECTED\n", log)
        self.assertEqual(self._tree(), set(_stager_ref_files()))
        self.assertEqual(log.count("— non-reference file"), 4, log)
        self.assertIn("/processed_5b/x\\nshared_metadata' (~0 GiB) — non-reference file", log)
        self.assertIn(f"[stage]   removed {self.dest}/processed_5b/a b.npy (~0 GiB)", log)
        self.assertIn("[stage] data dir is reference-only: 21 regular files", log)

    def test_newline_name_cannot_forge_the_end_state(self) -> None:
        # VERIFY E1: two reference paths are directories (hidden by "! -type d")
        # whose size equals the source's, and one non-reference file whose
        # name prints exactly the two missing "f <ref>" lines.
        self._staged_clean()
        for name in ("_READY", "store_meta.json"):
            p = self.dest / CACHE / name
            p.unlink()
            p.mkdir()
            (self.src / CACHE / name).write_bytes(b"\0" * p.stat().st_size)
        forged = self.dest / CACHE / "_READY\nf processed_5b" / f"hstu_cache_L{HIST}"
        forged.mkdir(parents=True)
        (forged / "store_meta.json").write_bytes(b"not a reference file")
        rc, log = self._stage()
        self.assertIn("[stage] 21 files tracked, 0 to copy (21 already staged)", log)
        self.assertFatalEndState(rc, log, "20 entries under", "1 not in REF_FILES",
                                 "2 of 21 reference files missing or not regular",
                                 f"[stage]   missing or not regular: {CACHE}/_READY",
                                 f"[stage]   missing or not regular: {CACHE}/store_meta.json")

    def test_names_cannot_forge_stage_lines(self) -> None:
        self._staged_clean()
        fake = "[stage] data dir is reference-only: 21 regular files under x, exactly REF_FILES"
        self._put(f"other/x\n{fake}")  # deep: never pruned, listed as an extra
        self._put(f"processed_5b/y\n{fake}")  # depth 1: listed while pruning is off
        rc, log = self._stage(env={"STAGE_PRUNE_NONREF": "0"})
        self.assertNotEqual(rc, 0, log)
        self.assertIn("[stage] data dir NOT reference-only", log)
        self.assertEqual([ln for ln in log.splitlines()
                          if ln.startswith("[stage] data dir is reference-only")], [])
        self.assertIn("[stage]   $'processed_5b/y\\n[stage] data dir", log)

    def test_prune_never_removes_a_directory(self) -> None:
        self._staged_clean()
        kept = (self._put("processed_5b/somedir/inner.npy"), self._put(f"{CACHE}/x\n/inner.npy"))
        rc, log = self._stage()
        self.assertNotEqual(rc, 0, log)
        self.assertIn("[stage] data dir NOT reference-only", log)
        self.assertNotIn("removed", log)
        for p in kept:
            self.assertTrue(p.exists(), p)
        self.assertTrue(set(_stager_ref_files()) <= self._tree())


class MinHistoryTest(_StagerCase):
    def test_effective_min_history_must_equal_history_length(self) -> None:
        for env, hist, want in (({"MIN_HISTORY": "0"}, HIST, "effective min_history 0 (MIN_HISTORY=0)"),
                                ({"MIN_HISTORY": "2039"}, HIST, "effective min_history 2039"),
                                ({"MIN_HISTORY": "-4086"}, HIST, "effective min_history -4086"),
                                ({"MIN_HISTORY": "+2_039"}, HIST, "effective min_history 2039 "
                                 "(MIN_HISTORY=+2_039, read as 2039)"),
                                ({}, 2039, "effective min_history 4086 (MIN_HISTORY=<unset>, "
                                           "gin default) != history_length 2039")):
            with self.subTest(env=env, hist=hist):
                rc, log = self._stage(env=env, hist=hist)
                self.assertNotEqual(rc, 0, log)
                self.assertIn("[stage] FATAL: " + want, log)
                self.assertFalse(self.dest.exists(), log)

    def test_unparsed_min_history_is_fatal(self) -> None:
        # int() raises on most of these; it reads the Arabic-Indic digits and
        # the 65-digit form as 4086, which this parse does not try (fail-closed).
        for raw in ("4086.0", "4__086", "_4086", "4086_", "+ 4086", " ", "0x0ff6",
                    "٤٠٨٦", "0" * 61 + "4086"):
            with self.subTest(raw=raw):
                rc, log = self._stage(env={"MIN_HISTORY": raw})
                self.assertNotEqual(rc, 0, log)
                self.assertIn("not an ASCII decimal integer", log)
                self.assertFalse(self.dest.exists(), log)

    def test_matching_or_empty_min_history_passes(self) -> None:
        # VERIFY m4 / E12: every spelling int() reads as 4086 (env_int: int(raw)).
        for raw in (str(HIST), "", " 4086", "4086 ", "+4086", "4_086", "04086", "4086\n", "\t4086 "):
            with self.subTest(raw=raw):
                if raw:
                    self.assertEqual(int(raw), HIST)
                rc, log = self._stage(env={"MIN_HISTORY": raw})
                self.assertEqual(rc, 0, log)
                self.assertIn(f"[stage] effective min_history {HIST} = history_length {HIST}", log)
                self.assertIn("[stage] data dir is reference-only: 21 regular files", log)

    def test_off_only_notes_it(self) -> None:
        rc, log = self._stage(env={KNOB: "0", "MIN_HISTORY": "0"})
        self.assertReportOnly(rc, log)
        self.assertIn(f"[stage] note: effective min_history 0 != history_length {HIST}", log)


class AwkAgreementTest(_StagerCase):
    CASES = (
        ("real gin", None),
        ("false", FALSE),
        ("qualified false added", TRUE + "_env_bootstrap.apply_env_bootstrap.%s = False\n" % KNOB),
        ("scoped false added", TRUE + "stage/apply_env_bootstrap.%s = False\n" % KNOB),
        ("lone qualified true", "_env_bootstrap.apply_env_bootstrap.%s = True\n" % KNOB),
        ("include", TRUE + "include 'x.gin'\n"),
        ("include no space", TRUE + "include'x.gin'\n"),
        ("split binding", TRUE + f"apply_env_bootstrap.{KNOB} \\\n    = False\n"),
        ("crlf + tabs", f"\tapply_env_bootstrap.{KNOB}\t=\tTrue\t\r\n"),
        ("trailing comment", TRUE.replace("True", "True  # keep")),
        ("mention in string", TRUE + f'DOC = "{KNOB}"\n'),
        ("not an include", TRUE + "include_x.param = 1\n"),
        # v2 (VERIFY m2/m3)
        ("include backslash continuation", TRUE + "include\\\n'x.gin'\n"),
        ("import backslash continuation", TRUE + "import\\\n  foo\n"),
        ("cr hides include", TRUE + "# note\rinclude 'x.gin'\n"),
        ("cr hides binding", FALSE + "# note\r" + TRUE),
        ("cr-only file", "# c\r" + TRUE.replace("\n", "\r")),
        ("backslash in qualified name", TRUE + f"a\\b.apply_env_bootstrap.{KNOB} = False\n"),
    )

    def test_all_awks_agree(self) -> None:
        awks = sorted(_awks())
        if len(awks) < 2:
            self.skipTest("fewer than two awk implementations installed")
        for name, body in self.CASES:
            gin = None if body is None else self._gin(body)
            results = {}
            for a in awks:
                dest = self.root / f"dest_{a}_{len(results)}_{abs(hash(name))}"
                rc, log = self._stage(gin=gin, awk=a, dest=dest)
                first = [ln for ln in log.splitlines()
                         if ln.startswith(("[stage] reference-only", "[stage] FATAL"))][:1]
                results[a] = (rc, first)
            with self.subTest(case=name):
                self.assertEqual(len({repr(v) for v in results.values()}), 1, results)


class StaticTest(unittest.TestCase):
    def test_gin_min_history_default_matches_stager(self) -> None:
        lines = GIN.read_text().splitlines()
        q = r"^\s*(?:[\w./]*[./])?"

        def bound(sel: str) -> List[str]:
            rx = re.compile(q + re.escape(sel) + r"\s*=\s*(.*?)\s*$")
            return [m.group(1) for m in map(rx.match, lines) if m]

        self.assertEqual(bound("get_dataset.min_history"), ["@mh/env_int()"])
        self.assertEqual(bound("mh/env_int.key"), ['"MIN_HISTORY"'])
        default = bound("mh/env_int.default")
        self.assertEqual(len(default), 1)
        m = re.search(r"^MH_EFF=\$\{MIN_HISTORY:-(\d+)\}$", STAGER.read_text(), re.M)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), default[0])
        self.assertEqual(int(default[0]), HIST)

    def test_real_gin_parse_inputs(self) -> None:
        lines = GIN.read_text().splitlines()
        stmt = re.compile(r"^\s*(?:include|import|from)(?:[^\w./]|$)")
        self.assertEqual([ln for ln in lines if stmt.match(ln)], [])
        mention = re.compile(r"%s(?:\W|$)" % KNOB)
        hits = [ln for ln in lines if mention.search(ln.split("#", 1)[0])]
        self.assertEqual(hits, [TRUE.strip()])

    def test_launcher_passes_repo_gin_and_logs_mode(self) -> None:
        if not LAUNCHER.exists():
            self.skipTest("no scripts/launch_slurm.sh in this tree (Primus launches with primus-cli)")
        text = LAUNCHER.read_text()
        self.assertIn('_stage_gin="$REPO/generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"',
                      text)
        self.assertIn('if ! STAGE_GIN="$_stage_gin" srun --ntasks-per-node=1 bash "$_stager"', text)
        self.assertIn("item (17b) applied by this launch", text)
        self.assertEqual(text.count("item (17b) NOT applied by this launch"), 2)

    def test_scripts_parse(self) -> None:
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        for script in (s for s in (STAGER, LAUNCHER) if s.exists()):
            out = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)

    def test_awk_parse_has_no_backslash_in_a_bracket(self) -> None:
        # VERIFY m3: busybox awk keeps a backslash inside [...] ("\/" there
        # matched "\"), so the parse writes its classes without one.
        prog = STAGER.read_text().split("_ref_gin=$(awk '", 1)[1].split("' \"$STAGE_GIN\")", 1)[0]
        self.assertNotIn("\\/", prog)
        classes = re.findall(r"\[\^?(?:\[:[a-z]+:\]|[^\[\]])*\]", prog)
        self.assertGreaterEqual(len(classes), 8, prog)
        self.assertEqual([c for c in classes if "\\" in c], [])

    def test_guard_runs_before_purge_and_prune_only_unlinks(self) -> None:
        text = STAGER.read_text()
        self.assertLess(text.index('src_alias=\nif within "$R_DEST" "$R_SRC"'),
                        text.index('if [ "$PURGE" = 1 ]; then'))
        unlink = text.split("unlink_nonref() {", 1)[1].split("\n}\n", 1)[0]
        self.assertIn('rm -f -- "$p"', unlink)
        self.assertNotIn("rm -r", unlink)
        prune = text.split("declare -A is_ref=()", 1)[1].split("\n# End state", 1)[0]
        self.assertNotIn("drop ", prune)
        self.assertIn("-printf '%P\\0'", prune)


if __name__ == "__main__":
    unittest.main()
