"""X2H (ops/hip/ec_dgrad_x2h.py): the build cache under the launcher's umask, on CPU (fake hipcc, no GPU).

scripts/launch_slurm.sh runs the workers under umask 000. The cache root must still come out 0700, so
the shared entry is trusted and a relaunch is a cache hit; a private build directory (an untrusted root,
a rebuild) is removed once its library is loaded. A pre-existing group/world-writable root, or one of
another uid, is neither trusted nor chmod-ed (a planted .so must not be loaded). ec_dgrad_x2h_cpu_test.py
holds the rest of the cache tests.
"""

import glob
import os
import shutil
import stat
import tempfile
import types
import unittest
from unittest import mock

from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h

_FAKE_HIPCC = """#!/usr/bin/env python3
import os, sys
a = sys.argv[1:]
if "--version" in a:
    print("fake hipcc 0.0"); sys.exit(0)
if os.environ.get("FAKE_HIPCC_FAIL"):
    sys.stderr.write("x.cu:1:1: error: fake failure\\n"); sys.exit(1)
open(a[a.index("-o") + 1], "wb").write(b"not a real library")
sys.stderr.write("x.cu:1:1: remark: Function Name: _ZN3x2h8x2h_passILi1ELi0EEEvNS_6ParamsE [-R]\\n")
sys.stderr.write("x.cu:1:1: remark:     VGPRs Spill: 0 [-R]\\n")
sys.stderr.write("x.cu:1:1: remark:     ScratchSize [bytes/lane]: 0 [-R]\\n")
"""


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


class X2HUmaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="x2h_umasktest_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.hipcc = os.path.join(self.tmp, "hipcc")
        with open(self.hipcc, "w") as fh:
            fh.write(_FAKE_HIPCC)
        os.chmod(self.hipcc, 0o755)
        self.root = os.path.join(self.tmp, "cache")  # not created: build() makes it
        self.scratch = os.path.join(self.tmp, "tmpdir")  # mkdtemp's default dir (private builds)
        os.mkdir(self.scratch)
        for p in (mock.patch.dict(os.environ, {x2h.HIPCC_ENV_VAR: self.hipcc, x2h.CACHE_ENV_VAR: self.root}),
                  mock.patch.object(tempfile, "tempdir", self.scratch)):
            p.start()
            self.addCleanup(p.stop)
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        x2h.reset_for_testing()
        self.addCleanup(x2h.reset_for_testing)
        self.loaded = []  # (path, existed at "dlopen", cached)

    def _privates(self):
        return glob.glob(os.path.join(self.scratch, x2h._PRIVATE_PREFIX + "*"))

    def _fake_lib(self, so, log, build_s, cached):
        self.loaded.append((str(so), os.path.isfile(so), cached))
        return types.SimpleNamespace(path=so, log=log, build_s=build_s, cached=cached, census=x2h.census(log))

    def _load(self):
        """One process's load(1) (reset first: a relaunch)."""
        x2h.reset_for_testing()
        with mock.patch.object(x2h, "_Lib", self._fake_lib):
            return x2h.load(1)

    def test_umask_000_shared_cache_is_trusted_and_hit(self):
        os.umask(0o000)
        lib1 = self._load()
        entry = x2h.cache_dir()
        self.assertEqual(_mode(self.root), 0o700)
        self.assertEqual(_mode(entry), 0o700)
        self.assertTrue(x2h._trusted(entry.parent) and x2h._trusted(entry))
        self.assertFalse(lib1.cached)
        self.assertEqual(lib1.path.parent, entry)  # built into the shared entry, not privately
        lib2 = self._load()  # the relaunch
        self.assertTrue(lib2.cached)
        self.assertEqual(lib2.path, lib1.path)
        self.assertIsNotNone(lib2.census.get(x2h.pass_kernel(1)))
        self.assertEqual(self._privates(), [])
        self.assertTrue(all(existed for _, existed, _ in self.loaded))
        # build() directly: also a hit, and nothing private is created
        self.assertTrue(x2h.build()[3])
        self.assertEqual(self._privates(), [])

    def test_world_writable_root_is_private_and_removed(self):
        for um in (0o000, 0o022):
            os.umask(um)
            shutil.rmtree(self.root, True)
            os.mkdir(self.root)
            os.chmod(self.root, 0o777)  # pre-existing (e.g. made by an older loader under umask 000)
            lib = self._load()
            self.assertFalse(lib.cached)
            self.assertTrue(lib.path.parent.name.startswith(x2h._PRIVATE_PREFIX), lib.path)
            self.assertTrue(self.loaded[-1][1])  # the .so existed when it was loaded ...
            self.assertEqual(self._privates(), [])  # ... and its private directory is gone afterwards
            self.assertIn("VGPRs Spill: 0", lib.log)  # the build log stays in memory (census / arming line)
            self.assertIsNotNone(lib.census.get(x2h.pass_kernel(1)))
            self.assertEqual(_mode(self.root), 0o777)  # never chmod-ed, never trusted
            self.assertFalse(os.path.exists(x2h.cache_dir() / x2h._LIB_NAME))  # nothing written there

    def test_foreign_uid_root_is_not_trusted(self):
        os.umask(0o000)
        os.mkdir(self.root)
        os.chmod(self.root, 0o700)
        with mock.patch.object(os, "getuid", return_value=os.getuid() + 4242):
            self.assertFalse(x2h._trusted(x2h.Path(self.root)))
            lib = self._load()
        self.assertFalse(lib.cached)
        self.assertTrue(lib.path.parent.name.startswith(x2h._PRIVATE_PREFIX), lib.path)
        self.assertEqual(self._privates(), [])
        self.assertEqual(_mode(self.root), 0o700)
        self.assertFalse(os.path.exists(x2h.cache_dir() / x2h._LIB_NAME))

    def test_umask_022_unchanged(self):
        lib1 = self._load()
        entry = x2h.cache_dir()
        self.assertFalse(lib1.cached)
        self.assertEqual(lib1.path.parent, entry)
        self.assertTrue(x2h._trusted(entry.parent) and x2h._trusted(entry))
        self.assertEqual(_mode(entry), 0o700)
        self.assertTrue(self._load().cached)
        self.assertEqual(self._privates(), [])
        # an existing root of ours at 0755 (what the old loader made under umask 022) is used as it is
        shutil.rmtree(self.root)
        os.mkdir(self.root)
        os.chmod(self.root, 0o755)
        self.assertFalse(self._load().cached)
        self.assertTrue(self._load().cached)
        self.assertEqual(_mode(self.root), 0o755)
        self.assertEqual(self._privates(), [])

    def test_rebuild_of_a_bad_entry_is_private_and_removed(self):
        os.umask(0o000)
        so = self._load().path
        (so.parent / "build.log").write_text("no census here\n")  # a cached entry without its census
        lib = self._load()
        self.assertFalse(lib.cached)
        self.assertTrue(lib.path.parent.name.startswith(x2h._PRIVATE_PREFIX))
        self.assertFalse(so.exists())  # fresh=True dropped the shared entry
        self.assertEqual(self._privates(), [])
        self.assertTrue(self.loaded[-1][1])
        self.assertFalse(self._load().cached)  # the next launch rebuilds the shared entry ...
        self.assertTrue(self._load().cached)  # ... and the one after hits it
        self.assertEqual(self._privates(), [])

    def test_failed_private_build_leaves_nothing(self):
        os.umask(0o000)
        os.mkdir(self.root)
        os.chmod(self.root, 0o777)
        with mock.patch.dict(os.environ, {"FAKE_HIPCC_FAIL": "1"}):
            with self.assertRaisesRegex(RuntimeError, "hipcc rc=1"):
                self._load()
        self.assertEqual(self._privates(), [])


if __name__ == "__main__":
    unittest.main()
