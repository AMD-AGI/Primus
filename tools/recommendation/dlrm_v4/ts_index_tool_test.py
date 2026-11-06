"""Unit tests for scripts/ts_index_tool.py (CPU, numpy only, no torch)."""

import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import ts_index_tool as tool  # noqa: E402

H, W, N = 8, 1000, 50_000
CHUNK = ["--chunk-size", "7001"]


def quiet(argv):
    out = io.StringIO()
    lines = []
    with redirect_stderr(out):
        args = tool.build_parser().parse_args(argv)
        try:
            rc = args.func(args, log=lines.append)
        except tool.ToolError as e:
            lines.append(str(e))
            rc = e.code
    return rc, "\n".join(lines)


@unittest.skipIf(np is None, "numpy required")
class Base(unittest.TestCase):
    anchor = True

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tsit-"))
        self.cache = self.tmp / "cache"
        self.cache.mkdir()
        rng = np.random.default_rng(7)
        # windows 0..19 with a hole at window 5; non-identity positions into a 2x array
        ts = rng.integers(0, 20 * W, size=N, dtype=np.int64)
        ts[(ts // W) == 5] += W
        ts[0], ts[1] = 0, 20 * W - 1
        positions = np.sort(rng.choice(2 * N, size=N, replace=False)).astype(np.int64)
        flat = rng.integers(0, 20 * W, size=2 * N, dtype=np.int64)
        flat[positions] = ts
        np.save(self.cache / f"positions_L{H}.npy", positions)
        np.save(self.cache / "flat_timestamps.npy", flat)
        if self.anchor:
            np.save(self.cache / f"anchor_ts_L{H}.npy", ts)
        self.ts = ts
        self.nm = tool.names(f"L{H}", W)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def idx_args(self, cache=None):
        return ["--cache-dir", str(cache or self.cache), "--history-length", str(H), "--window-seconds", str(W)]

    def prebuild(self, *extra):
        return quiet(["prebuild", *self.idx_args(), *CHUNK, *extra])

    def verify(self, *extra, cache=None):
        return quiet(["verify", *self.idx_args(cache), *CHUNK, *extra])

    def path(self, key, cache=None):
        return (cache or self.cache) / self.nm[key]

    def sidecar(self, cache=None):
        return json.loads(self.path("sidecar", cache).read_text())

    def mutate_indices(self, fn):
        a = np.load(self.path("indices"), mmap_mode="r+")
        fn(a)
        a.flush()
        del a


class TestBuildVerify(Base):
    def test_build_verify_pass_and_sidecar(self):
        rc, out = self.prebuild("--verify")
        self.assertEqual(rc, 0, out)
        side = self.sidecar()
        self.assertEqual(side["verify"]["status"], "passed")
        self.assertEqual(side["anchor_count"], N)
        self.assertEqual(side["num_windows"], 20)
        self.assertEqual(side["window_seconds"], W)
        self.assertEqual(side["dtype"], "<i8")
        ip = self.path("indices")
        self.assertEqual(side["files"]["indices"]["size"], ip.stat().st_size)
        self.assertEqual(side["files"]["indices"]["hash"], hashlib.sha256(ip.read_bytes()).hexdigest())
        self.assertEqual(side["builder"]["sha256"], hashlib.sha256(tool.BUILDER_PATH.read_bytes()).hexdigest())
        self.assertIn(f"anchor_ts_L{H}.npy", side["source"])
        self.assertIn(f"positions_L{H}.npy", side["source"])
        self.assertEqual(list(self.cache.glob(".ts-*")), [])
        # independent reference: stable bucket of each anchor, ascending
        ref = np.argsort(self.ts // W, kind="stable")
        self.assertTrue(np.array_equal(np.load(ip), ref))

    def test_tool_names_match_builder(self):
        b = tool.load_builder()
        for mh in (None, H, 3, 0):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                tag = tool.tag_for(H, mh)
                nm = tool.names(tag, W)
                (d / nm["indices"]).touch()
                (d / nm["offsets"]).touch()
                ip, op = b.build_offline_ts_indices(d, H, mh, window_seconds=W)
                self.assertEqual((ip.name, op.name), (nm["indices"], nm["offsets"]))

    def test_refuse_overwrite_and_force(self):
        self.assertEqual(self.prebuild()[0], 0)
        ino = self.path("indices").stat().st_ino
        rc, out = self.prebuild()
        self.assertEqual(rc, tool.EXIT_REFUSED, out)
        self.assertEqual(self.path("indices").stat().st_ino, ino)
        rc, out = self.prebuild("--force", "--verify")
        self.assertEqual(rc, 0, out)
        self.assertNotEqual(self.path("indices").stat().st_ino, ino)

    def test_missing_sidecar_and_adopt(self):
        tool.load_builder().build_offline_ts_indices(self.cache, H, None, window_seconds=W)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("missing", out)
        self.assertFalse(self.path("sidecar").exists())
        rc, out = self.verify("--adopt")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.sidecar()["verify"]["status"], "passed")
        self.assertTrue(self.sidecar()["adopted"])
        self.assertEqual(self.verify()[0], 0)

    def test_truncated(self):
        self.prebuild()
        with open(self.path("indices"), "r+b") as f:
            f.truncate(self.path("indices").stat().st_size - 8)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("truncated", out)
        self.assertEqual(self.sidecar()["verify"]["status"], "failed")

    def test_different_length_source(self):
        self.prebuild()
        pos = np.load(self.path("positions"))
        np.save(self.path("positions"), np.concatenate([pos, [2 * N - 1]]))
        np.save(self.path("anchor_ts"), np.concatenate([self.ts, [3]]))
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("entries", out)

    def test_stale_same_length_source(self):
        self.prebuild("--verify")
        ts = self.ts.copy()
        ts[100:200] = (ts[100:200] + 7 * W) % (20 * W)  # same length, other windows
        np.save(self.path("anchor_ts"), ts)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertRegex(out, "window count|wrong window")

    def test_stale_source_same_windows_caught_by_fingerprint(self):
        self.prebuild("--verify")
        ts = self.ts.copy()
        ts[:] = ts // W * W + (ts % W + 1) % W  # every timestamp moves, no window does
        np.save(self.path("anchor_ts"), ts)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("sampled content differs", out)
        # an operator who has checked the content in full may rebind
        rc, out = self.verify("--rebind-source")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.verify()[0], 0)

    def test_out_of_range(self):
        self.prebuild("--verify")
        self.mutate_indices(lambda a: a.__setitem__(1234, N + 5))
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("out of [0", out)

    def test_entry_equal_to_count(self):
        self.prebuild()
        self.mutate_indices(lambda a: a.__setitem__(N - 1, N))
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("out of [0", out)

    def test_negative_entry(self):
        self.prebuild()
        self.mutate_indices(lambda a: a.__setitem__(0, -1))
        self.assertEqual(self.verify()[0], 1)

    def test_swap_within_window(self):
        self.prebuild("--verify")
        offs = np.load(self.path("offsets"))
        lo = int(offs[3])

        def swap(a):
            a[lo + 10], a[lo + 11] = int(a[lo + 11]), int(a[lo + 10])
        self.mutate_indices(swap)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("order violations", out)

    def test_swap_across_chunk_boundary(self):
        self.prebuild("--verify")
        offs = np.load(self.path("offsets"))
        # chunk size 7001: slots 7000/7001 straddle a chunk boundary; ensure same window
        k = 7000
        w = int(np.searchsorted(offs, k, side="right")) - 1
        self.assertLess(k + 1, offs[w + 1])

        def swap(a):
            a[k], a[k + 1] = int(a[k + 1]), int(a[k])
        self.mutate_indices(swap)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("order violations", out)

    def test_swap_across_windows(self):
        self.prebuild()
        offs = np.load(self.path("offsets"))
        i, j = int(offs[2]), int(offs[9])

        def swap(a):
            a[i], a[j] = int(a[j]), int(a[i])
        self.mutate_indices(swap)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("wrong window", out)

    def test_duplicate_entry(self):
        self.prebuild()
        offs = np.load(self.path("offsets"))
        lo = int(offs[4])
        self.mutate_indices(lambda a: a.__setitem__(lo + 1, int(a[lo])))
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("order violations", out)

    def test_offsets_corrupt(self):
        self.prebuild()
        offs = np.load(self.path("offsets"))
        offs[3] += 1
        np.save(self.path("offsets"), offs)
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("histogram", out)

    def test_hash_mismatch_same_content_size(self):
        # bytes changed in a way the structural checks cannot see: header padding
        self.prebuild("--verify")
        p = self.path("indices")
        raw = bytearray(p.read_bytes())
        hdr_end = raw.index(b"\n")
        self.assertEqual(raw[hdr_end - 1:hdr_end], b" ")
        raw[hdr_end - 1:hdr_end] = b"\t"
        p.write_bytes(bytes(raw))
        rc, out = self.verify()
        self.assertEqual(rc, 1)
        self.assertIn("differs from sidecar", out)


class TestFlatTimestamps(Base):
    anchor = False

    def test_build_verify_from_flat(self):
        rc, out = self.prebuild("--verify")
        self.assertEqual(rc, 0, out)
        self.assertIn("flat_timestamps.npy", self.sidecar()["source"])
        self.assertEqual(self.sidecar()["verify"]["timestamps_source"], "flat_timestamps")
        self.mutate_indices(lambda a: a.__setitem__(5, int(a[6])))
        self.assertEqual(self.verify()[0], 1)


class TestTempAndSignals(Base):
    def test_cleanup_temp_and_busy_lock(self):
        d = self.cache / ".ts-indices-ab_d1234"
        d.mkdir()
        (d / self.nm["indices"]).write_bytes(b"x" * 4096)
        f = self.cache / ".ts-index-tool-zz.tmp"
        f.write_text("{}")
        keep = self.cache / ".ts-indices-not-a-match"
        keep.mkdir()
        rc, out = quiet(["status", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 1)
        self.assertIn("leftover temp", out)
        with open(self.cache / "_positions_lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
            self.assertEqual(rc, tool.EXIT_BUSY, out)
            self.assertTrue(d.exists() and f.exists())
        rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache), "--dry-run"])
        self.assertTrue(d.exists())
        rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 0, out)
        self.assertFalse(d.exists() or f.exists())
        self.assertTrue(keep.exists())

    def test_cleanup_temp_non_owner_is_usage_error_and_deletes_nothing(self):
        d = self.cache / ".ts-indices-ab_d1234"
        d.mkdir()
        (d / "part.npy").write_bytes(b"x" * 4096)
        (self.cache / "_positions_lock").touch()
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory permissions")
        try:
            d.chmod(0o555)  # stand-in for a root-owned temp dir left by an in-init build
            rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
            self.assertEqual(rc, tool.EXIT_USAGE, out)
            self.assertIn("cannot remove", out)
            self.assertTrue((d / "part.npy").exists())
            d.chmod(0o755)
            self.cache.chmod(0o555)  # non-writable cache dir
            rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
            self.assertEqual(rc, tool.EXIT_USAGE, out)
            self.assertTrue((d / "part.npy").exists())
            # CLI path: clean message, exit 2, no traceback
            p = subprocess.run([sys.executable, str(HERE / "ts_index_tool.py"), "cleanup-temp",
                                "--cache-dir", str(self.cache)], capture_output=True, text=True)
            self.assertEqual(p.returncode, tool.EXIT_USAGE, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.cache.chmod(0o755)
            lk = self.cache / "_positions_lock"
            lk.chmod(0o444)  # root-owned 0644 lock left by a root trainer, seen by a non-owner
            rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
            self.assertEqual(rc, tool.EXIT_USAGE, out)
            self.assertTrue((d / "part.npy").exists())
        finally:
            self.cache.chmod(0o755)
            d.chmod(0o755)
            (self.cache / "_positions_lock").chmod(0o644)
        rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 0, out)
        self.assertFalse(d.exists())

    def test_load_builder_writes_no_bytecode(self):
        src = self.tmp / "bsrc"
        src.mkdir()
        shutil.copyfile(tool.BUILDER_PATH, src / "offline_ts_indices.py")
        code = ("import sys; sys.path.insert(0, %r); import ts_index_tool as t; "
                "from pathlib import Path; t.BUILDER_PATH = Path(%r); m = t.load_builder(); "
                "assert hasattr(m, 'build_offline_ts_indices')" % (str(HERE), str(src / "offline_ts_indices.py")))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((src / "__pycache__").exists(), list(src.rglob("*")))

    def _slow_prebuild(self, flag):
        code = f"""
import sys, time, numpy
sys.path.insert(0, {str(HERE)!r})
import ts_index_tool as t
orig = t.load_builder
class NP:
    def __getattr__(self, k): return getattr(numpy, k)
    def argsort(self, *a, **k):
        open({str(flag)!r}, 'w').close(); time.sleep(0.3); return numpy.argsort(*a, **k)
def lb():
    m = orig(); m.np = NP(); return m
t.load_builder = lb
sys.exit(t.main(['prebuild', '--cache-dir', {str(self.cache)!r}, '--history-length', '{H}',
                 '--window-seconds', '{W}', '--chunk-size', '2000']))
"""
        return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def _wait(self, flag, proc):
        t0 = time.time()
        while not flag.exists():
            if proc.poll() is not None:
                self.fail(proc.communicate()[1])
            self.assertLess(time.time() - t0, 60)
            time.sleep(0.02)
        self.assertEqual(len(list(self.cache.glob(".ts-indices-*"))), 1)

    def test_sigterm_cleans_temp(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            flag = self.tmp / f"flag{int(sig)}"
            p = self._slow_prebuild(flag)
            self._wait(flag, p)
            p.send_signal(sig)
            err = p.communicate(timeout=60)[1]
            self.assertEqual(p.returncode, 128 + int(sig), err)
            self.assertEqual(list(self.cache.glob(".ts-*")), [])
            self.assertFalse(self.path("indices").exists() or self.path("offsets").exists())
            self.assertFalse(self.path("sidecar").exists())

    def test_sigkill_leak_then_cleanup(self):
        flag = self.tmp / "flagk"
        p = self._slow_prebuild(flag)
        self._wait(flag, p)
        p.kill()
        p.communicate(timeout=60)
        self.assertEqual(len(list(self.cache.glob(".ts-indices-*"))), 1)
        rc, out = quiet(["cleanup-temp", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 0, out)
        self.assertEqual(list(self.cache.glob(".ts-indices-*")), [])
        # prebuild also clears leftovers before building
        p = self._slow_prebuild(self.tmp / "flagk2")
        self._wait(self.tmp / "flagk2", p)
        p.kill()
        p.communicate(timeout=60)
        rc, out = self.prebuild("--verify")
        self.assertEqual(rc, 0, out)
        self.assertEqual(list(self.cache.glob(".ts-*")), [])


class TestCheckStaged(Base):
    def stage(self):
        dest = self.tmp / "staged"
        dest.mkdir()
        for e in self.cache.iterdir():
            if e.is_file():
                shutil.copyfile(e, dest / e.name)  # like dd: new mtimes
        return dest

    def check(self, dest, *extra, no_numpy=True):
        argv = ["check-staged", str(dest), "--history-length", str(H), "--window-seconds", str(W), *extra]
        if not no_numpy:
            return quiet(argv)
        code = ("import sys; sys.modules['numpy'] = None; "
                f"sys.path.insert(0, {str(HERE)!r}); import ts_index_tool as t; sys.exit(t.main({argv!r}))")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def test_ok_without_numpy(self):
        self.prebuild("--verify")
        dest = self.stage()
        rc, out = self.check(dest)
        self.assertEqual(rc, 0, out)
        rc, out = self.check(dest, "--hash")
        self.assertEqual(rc, 0, out)

    def test_truncated_and_same_size_corruption(self):
        self.prebuild("--verify")
        dest = self.stage()
        p = dest / self.nm["indices"]
        raw = bytearray(p.read_bytes())
        raw[-8] ^= 1
        p.write_bytes(bytes(raw))
        self.assertEqual(self.check(dest)[0], 0)          # size-only mode cannot see it
        rc, out = self.check(dest, "--hash")
        self.assertEqual(rc, 1)
        self.assertIn("hash differs", out)
        with open(p, "r+b") as f:
            f.truncate(len(raw) - 16)
        rc, out = self.check(dest)
        self.assertEqual(rc, 1)

    def test_missing_sidecar_unverified_and_external_sidecar(self):
        self.prebuild()
        dest = self.stage()
        rc, out = self.check(dest)
        self.assertEqual(rc, 1)
        self.assertIn("not 'passed'", out)
        self.assertEqual(self.check(dest, "--allow-unverified")[0], 0)
        (dest / self.nm["sidecar"]).unlink()
        rc, out = self.check(dest)
        self.assertEqual(rc, 1)
        self.assertIn("missing", out)
        self.verify()
        rc, out = self.check(dest, "--sidecar", str(self.path("sidecar")))
        self.assertEqual(rc, 0, out)

    def test_offsets_and_source_change(self):
        self.prebuild("--verify")
        dest = self.stage()
        a = dest / self.nm["anchor_ts"]
        raw = bytearray(a.read_bytes())
        raw[200] ^= 0xFF  # inside the fingerprinted header block
        a.write_bytes(bytes(raw))
        rc, out = self.check(dest)
        self.assertEqual(rc, 1)
        self.assertIn("sampled content differs", out)
        self.assertEqual(self.check(dest, "--no-source")[0], 0)
        o = dest / self.nm["offsets"]
        offs = np.load(o)
        offs[2] += 1
        np.save(o, offs)
        rc, out = self.check(dest, "--no-source")
        self.assertEqual(rc, 1)

    def test_sidecar_dir_uses_authoritative_copy(self):
        self.prebuild("--verify")
        dest = self.stage()
        # stale staged sidecar that vouches for a corrupted same-size staged copy
        o = dest / self.nm["offsets"]
        offs = np.load(o)
        offs[3] += 1
        np.save(o, offs)
        side = json.loads((dest / self.nm["sidecar"]).read_text())
        side["files"]["offsets"]["hash"] = hashlib.sha256(o.read_bytes()).hexdigest()
        (dest / self.nm["sidecar"]).write_text(json.dumps(side))
        rc, out = self.check(dest, "--no-source")
        self.assertEqual(rc, 0, out)  # the staged sidecar vouches for its own stale copy
        rc, out = self.check(dest, "--sidecar-dir", str(self.cache))
        self.assertEqual(rc, 1, out)
        self.assertIn("hash differs", out)
        self.assertIn("stale copy", out)
        np.save(o, np.load(self.path("offsets")))
        rc, out = self.check(dest, "--sidecar-dir", str(self.cache))
        self.assertEqual(rc, 0, out)
        rc, out = self.check(dest, "--sidecar-dir", str(self.cache), "--sidecar", str(self.path("sidecar")))
        self.assertEqual(rc, tool.EXIT_USAGE, out)

    def test_sidecar_dir_respects_min_history_tag(self):
        self.prebuild("--verify")
        dest = self.stage()
        rc, out = self.check(dest, "--sidecar-dir", str(self.cache), "--min-history", "0")
        self.assertEqual(rc, 1, out)
        self.assertIn(f"ts_index_manifest_L{H}_m0_W{W}.json", out)
        rc, out = self.check(dest, "--sidecar-dir", str(self.cache), "--min-history", str(H))
        self.assertEqual(rc, 0, out)  # min_history == history_length -> plain L tag

    def test_sidecar_names_cannot_escape_dest(self):
        self.prebuild("--verify")
        dest = self.stage()
        outside = self.tmp / "outside.npy"
        shutil.copyfile(dest / self.nm["offsets"], outside)
        sp = dest / self.nm["sidecar"]
        good = sp.read_text()
        side = json.loads(good)
        side["files"]["offsets"]["name"] = "../outside.npy"
        sp.write_text(json.dumps(side))
        (dest / self.nm["offsets"]).unlink()
        rc, out = self.check(dest)
        self.assertEqual(rc, 1, out)
        self.assertIn("is not", out)
        shutil.copyfile(outside, dest / self.nm["offsets"])
        side = json.loads(good)
        side["source"]["../outside.npy"] = side["source"].pop(self.nm["anchor_ts"])
        sp.write_text(json.dumps(side))
        rc, out = self.check(dest)
        self.assertEqual(rc, 1, out)
        self.assertIn("source entry", out)
        side = json.loads(good)
        side["files"]["offsets"]["hash_algo"] = "nope"
        sp.write_text(json.dumps(side))
        rc, out = self.check(dest)
        self.assertEqual(rc, 1, out)
        self.assertIn("hash_algo", out)
        self.assertNotIn("Traceback", out)
        sp.write_text(good)
        self.assertEqual(self.check(dest)[0], 0)
        # verify and status apply the same rule on the cache itself
        csp = self.path("sidecar")
        cgood = csp.read_text()
        side = json.loads(cgood)
        side["files"]["indices"]["name"] = "../x.npy"
        csp.write_text(json.dumps(side))
        rc, out = self.verify("--no-write")
        self.assertEqual(rc, 1, out)
        self.assertIn("files.indices.name", out)
        rc, out = quiet(["status", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 1, out)
        csp.write_text(cgood)
        self.assertEqual(self.verify("--no-write")[0], 0)

    def test_missing_index_file(self):
        self.prebuild("--verify")
        dest = self.stage()
        (dest / self.nm["indices"]).unlink()
        rc, out = self.check(dest)
        self.assertEqual(rc, 1)
        self.assertIn("missing", out)


class TestStatus(Base):
    def test_status(self):
        rc, out = quiet(["status", "--cache-dir", str(self.cache)])
        self.assertIn("no PR 907 index files", out)
        self.prebuild()
        rc, out = quiet(["status", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 1)  # built but not verified
        self.verify()
        rc, out = quiet(["status", "--cache-dir", str(self.cache)])
        self.assertEqual(rc, 0, out)
        self.assertIn("complete", out)
        self.path("offsets").unlink()
        rc, out = quiet(["status", "--cache-dir", str(self.cache)])
        self.assertIn("INCOMPLETE", out)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
