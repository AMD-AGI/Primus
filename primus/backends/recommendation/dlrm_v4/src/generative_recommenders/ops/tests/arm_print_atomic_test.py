# pyre-ignore-all-errors
"""CPU tests of ops.utils.arm_print_all_ranks, the one-write arming line (r16 iteration 44).

The trainer's launcher pipes `python ... 2>&1` through
`sed -e '/Latency not found for/d' -e '/Returning latency value of/d'`. Eight ranks also
write hipBLASLt's untuned-shape warning into that pipe as unbuffered std::cerr tokens. A
line printed while a token is pending shares its physical line and is deleted with it.
These tests pin the one-write form ("\\n<msg> [rank N]\\n" in a single os.write), its
fallbacks and its kill switch, and replay the launcher pipe with real processes.
"""
import contextlib
import io
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from generative_recommenders.ops import utils as U

SED = ["sed", "-e", "/Latency not found for/d", "-e", "/Returning latency value of/d"]


class _Out(io.StringIO):
    """A captured stdout that claims fd 1 (so the one-write path is taken)."""

    def fileno(self):
        return 1


class ArmPrintAtomicTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop("DLRM_HSTU_FP8_ARM_PRINT_ATOMIC", None)
        self._rank_cache = U._ARM_PRINT_RANK
        U._ARM_PRINT_RANK = None

    def tearDown(self) -> None:
        self._env.stop()
        U._ARM_PRINT_RANK = self._rank_cache

    def _print_with(self, msg, initialized, rank=None):
        out = []

        def fake_write(fd, data):
            out.append(bytes(data))
            return len(data)

        with mock.patch.object(U.torch.distributed, "is_available", return_value=True), \
                mock.patch.object(U.torch.distributed, "is_initialized", return_value=initialized), \
                mock.patch.object(U.torch.distributed, "get_rank", return_value=rank), \
                mock.patch.object(U.os, "write", side_effect=fake_write), contextlib.redirect_stdout(_Out()):
            U.arm_print_all_ranks(msg)
        return b"".join(out)

    def test_exit_line_reuses_the_last_process_group_rank(self) -> None:
        # r16 iteration 55: exit hooks print after the process group is destroyed; RANK is not exported, so these
        # lines read "[rank 0]" on every rank. The rank seen at the last in-group print is reused.
        os.environ.pop("RANK", None)
        self.assertEqual(self._print_with("armed OK: X ACTIVE", True, 5), b"\narmed OK: X ACTIVE [rank 5]\n")
        self.assertEqual(self._print_with("X counters: final=1", False), b"\nX counters: final=1 [rank 5]\n")

    def test_live_process_group_rank_wins_over_the_cache(self) -> None:
        U._ARM_PRINT_RANK = "5"
        self.assertEqual(self._print_with("m", True, 2), b"\nm [rank 2]\n")
        self.assertEqual(U._ARM_PRINT_RANK, "2")

    def test_no_cache_keeps_the_old_fallback(self) -> None:
        os.environ["RANK"] = "3"
        self.assertEqual(self._print_with("m", False), b"\nm [rank 3]\n")
        os.environ.pop("RANK")
        self.assertEqual(self._print_with("m", False), b"\nm [rank 0]\n")
        self.assertIsNone(U._ARM_PRINT_RANK)

    def test_one_write_with_leading_newline(self) -> None:
        calls = []

        def fake_write(fd, data):
            calls.append((fd, bytes(data)))
            return len(data)

        os.environ["RANK"] = "3"
        with mock.patch.object(U.os, "write", side_effect=fake_write), contextlib.redirect_stdout(_Out()):
            U.arm_print_all_ranks("armed OK: X ACTIVE")
        self.assertEqual(calls, [(1, b"\narmed OK: X ACTIVE [rank 3]\n")])

    def test_short_write_is_finished_not_repeated(self) -> None:
        calls = []

        def fake_write(fd, data):
            data = bytes(data)
            n = 5 if not calls else len(data)
            calls.append(data[:n])
            return n

        os.environ["RANK"] = "0"
        with mock.patch.object(U.os, "write", side_effect=fake_write), contextlib.redirect_stdout(_Out()):
            U.arm_print_all_ranks("armed OK: Y")
        self.assertEqual(b"".join(calls), b"\narmed OK: Y [rank 0]\n")
        self.assertEqual(len(calls), 2)

    def test_kill_switch_uses_print(self) -> None:
        out = _Out()
        os.environ.update({"RANK": "1", "DLRM_HSTU_FP8_ARM_PRINT_ATOMIC": "0"})
        with mock.patch.object(U.os, "write", side_effect=AssertionError("os.write must not be used")), \
                contextlib.redirect_stdout(out):
            U.arm_print_all_ranks("armed OK: Z")
        self.assertEqual(out.getvalue(), "armed OK: Z [rank 1]\n")

    def test_fallback_without_fileno_matches_old_output(self) -> None:
        # contextlib.redirect_stdout(io.StringIO()) is how the existing tests capture these lines.
        buf = io.StringIO()
        os.environ["RANK"] = "2"
        with contextlib.redirect_stdout(buf):
            U.arm_print_all_ranks("armed OK: W")
        self.assertEqual(buf.getvalue(), "armed OK: W [rank 2]\n")

    def test_failed_write_falls_back_to_print(self) -> None:
        out = _Out()
        os.environ["RANK"] = "4"
        with mock.patch.object(U.os, "write", side_effect=OSError(9, "EBADF")), contextlib.redirect_stdout(out):
            U.arm_print_all_ranks("armed OK: V")
        self.assertEqual(out.getvalue(), "armed OK: V [rank 4]\n")

    @unittest.skipUnless(shutil.which("sed") and shutil.which("sh"), "needs sed and sh")
    def test_launcher_pipe_keeps_every_line(self) -> None:
        """Real processes: 4 writers emit torn hipBLASLt-style tokens on fd 2 and arming lines on stdout, merged
        2>&1 into the launcher's exact sed filter. Every arming line must arrive, on a physical line of its own."""
        n_ranks, n_lines = 4, 150
        child = textwrap.dedent(
            """
            import os, random, sys
            from generative_recommenders.ops.utils import arm_print_all_ranks
            rank, n = int(os.environ["RANK"]), int(sys.argv[1])
            rnd = random.Random(rank)
            toks = ["Warning: Latency not found for MI_M=", "16", ", MI_N=", "16", ", MI_K=", "128",
                    ", mi_input_type=", "BFloat8Float8_fnuz", ". Returning latency value of ", "32",
                    " (really slow).", "\\n"]
            for i in range(n):
                for _ in range(rnd.randint(0, 3)):
                    for t in toks:
                        os.write(2, t.encode())
                arm_print_all_ranks("armed OK: TESTLINE seq=%d %s" % (i, "x" * 200))
            """
        )
        # .../<root>/generative_recommenders/ops/utils.py -> <root>, the dir that holds the package
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(U.__file__))))
        procs = " & ".join(
            f"RANK={r} {shlex.join([sys.executable, '-B', '-c', child, str(n_lines)])}"
            for r in range(n_ranks)
        )
        sed = shlex.join(SED)
        env = dict(os.environ, PYTHONPATH=root, PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run(["sh", "-c", f"{{ {procs} & wait; }} 2>&1 | {sed}"], env=env, capture_output=True,
                             timeout=600, check=True)
        out = res.stdout.decode()
        self.assertNotIn("Traceback", out, out[-2000:])
        lines = [ln for ln in out.split("\n") if "armed OK: TESTLINE" in ln]
        self.assertEqual(len(lines), n_ranks * n_lines)
        for ln in lines:  # each on a physical line of its own: starts with the message, ends with the rank tag
            self.assertTrue(ln.startswith("armed OK: TESTLINE seq="), ln[:120])
            self.assertRegex(ln, r" \[rank [0-9]+\]$")


if __name__ == "__main__":
    unittest.main()
