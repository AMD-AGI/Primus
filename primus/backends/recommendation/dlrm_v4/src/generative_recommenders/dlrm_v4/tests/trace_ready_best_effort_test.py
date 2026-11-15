"""The profiler's on_trace_ready callback is best-effort.

``_on_trace_ready_fn`` builds the callback that torch's ``profile.step()`` runs
at the end of every trace window, and ``Profiler.step()`` calls that
unguarded. So any exception the callback lets out ends the training job.
Building the op summary table (``key_averages().table()``) can raise inside
torch: ``UnicodeDecodeError`` on ``with_stack=True`` captures whose window holds
a FlyDSL JIT first step. Creating the trace directory can fail too. Neither may
crash the run, and the trace export and the file labels must be unaffected.
These cases use a stand-in profile object: no GPU and no real profiler.
"""

import logging
import os
import tempfile
import unittest

from generative_recommenders.dlrm_v4 import utils


class _Table:
    def __init__(self, text: str) -> None:
        self._text = text

    def table(self, sort_by=None) -> str:  # noqa: ARG002 - signature of torch's EventList.table
        return self._text


class _FakeProfile:
    """Minimal torch.profiler.profile stand-in for the callback."""

    def __init__(self, table_exc=None, step_num: int = 7) -> None:
        self.step_num = step_num
        self.exported = []
        self._table_exc = table_exc

    def key_averages(self, group_by_input_shape: bool = False) -> _Table:  # noqa: ARG002
        if self._table_exc is not None:
            raise self._table_exc
        return _Table("op-table")

    def export_chrome_trace(self, path: str) -> None:
        self.exported.append(path)
        with open(path, "w") as f:
            f.write('{"traceEvents": []}')


def _decode_error() -> UnicodeDecodeError:
    return UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


class TraceReadyBestEffortTest(unittest.TestCase):
    def test_table_failure_does_not_raise_and_still_exports(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            fn = utils._on_trace_ready_fn(rank=3, trace_dir=d)
            p = _FakeProfile(table_exc=_decode_error())
            with self.assertLogs(utils.logger, level=logging.WARNING) as cm:
                fn(p)  # must not raise
            self.assertEqual(p.exported, [os.path.join(d, "trace_step7_rank3.json")])
            self.assertTrue(os.path.isfile(p.exported[0]))
            self.assertTrue(
                any("Trace op table failed" in m and "UnicodeDecodeError" in m for m in cm.output),
                cm.output,
            )

    def test_trace_dir_failure_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "not_a_dir")
            with open(blocker, "w"):
                pass
            fn = utils._on_trace_ready_fn(rank=0, trace_dir=os.path.join(blocker, "traces"))
            p = _FakeProfile()
            with self.assertLogs(utils.logger, level=logging.WARNING) as cm:
                fn(p)  # os.makedirs under a regular file raises; must not propagate
            self.assertEqual(p.exported, [])
            self.assertTrue(any("Trace export/trim failed" in m for m in cm.output), cm.output)

    def test_labels_advance_across_table_failures(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            fn = utils._on_trace_ready_fn(rank=0, trace_dir=d, trace_steps=[52, 60])
            p = _FakeProfile(table_exc=RuntimeError("boom"))
            fn(p)
            fn(p)
            self.assertEqual(
                [os.path.basename(x) for x in p.exported],
                ["trace_step52_rank0.json", "trace_step60_rank0.json"],
            )

    def test_success_path_logs_table_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "new_dir")
            fn = utils._on_trace_ready_fn(rank=1, trace_dir=sub)
            p = _FakeProfile(step_num=12)
            with self.assertLogs(utils.logger, level=logging.WARNING) as cm:
                fn(p)
            self.assertEqual(p.exported, [os.path.join(sub, "trace_step12_rank1.json")])
            self.assertTrue(any("op-table" in m for m in cm.output), cm.output)
            self.assertFalse(any("Trace op table failed" in m for m in cm.output), cm.output)


if __name__ == "__main__":
    unittest.main()
