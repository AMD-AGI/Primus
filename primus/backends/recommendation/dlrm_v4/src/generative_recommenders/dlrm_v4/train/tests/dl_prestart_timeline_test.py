#!/usr/bin/env python3
# pyre-strict
"""[dlstart] startup timeline, item (109) DL_PRESTART_RANK_TIMELINE.

OFF (the gin default; env unset, "0" or anything but "1"): report() is
byte-identical to 544e691: rank 0's line is unchanged and no other rank logs
anything new (only the compliance-violation ERROR any rank already logged).
ON ("1"): rank 0's line is still unchanged; every other rank logs the same
fields as '[dlstart] rank-timeline rank=<r> ...' in a 'k:v ; ...' syntax that
shares none of rank 0's anchors (a torn rank-0 line in the launcher capture can
get another rank's fragment glued on; readers must never parse such a fragment
as rank 0's values).

Standard library only: dl_prestart.py is loaded by file path, so the test runs
without torch (login host or container)."""

import importlib.util
import logging
import os
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "dl_prestart.py")
_KNOB = "DL_PRESTART_RANK_TIMELINE"

# Fixed startup marks (wall seconds) and the rank-0 lines the shipped (544e691)
# report() printed for them, byte for byte.
_MARKS = {
    "prestart_begin": 100.0,
    "prestart_workers_ready": 102.5,
    "loop_entry": 146.9,
    "dataset_built": 146.9,
    "first_progress_start": 147.0,
    "pull1": 147.05,
    "fx_begin": 147.06,
    "idist0_begin": 147.08,
    "idist0_done": 147.2,
    "pull2": 147.3,
    "fwd_begin": 147.31,
    "fwd_end": 147.5,
    "pull3": 147.5,
    "enq3_done": 147.501,
    "idist1_begin": 147.502,
    "bwd_begin": 147.503,
    "bwd_end": 147.8,
    "opt_begin": 147.8,
    "opt_end": 147.81,
    "first_progress_done": 147.82,
}
_RUN_START = 146.9
_MARKS_TXT = (
    "prestart_begin=-46.900 prestart_workers_ready=-44.400 loop_entry=+0.000 "
    "dataset_built=+0.000 first_progress_start=+0.100 pull1=+0.150 fx_begin=+0.160 "
    "idist0_begin=+0.180 idist0_done=+0.300 pull2=+0.400 fwd_begin=+0.410 fwd_end=+0.600 "
    "pull3=+0.600 enq3_done=+0.601 idist1_begin=+0.602 bwd_begin=+0.603 bwd_end=+0.900 "
    "opt_begin=+0.900 opt_end=+0.910 first_progress_done=+0.920"
)
_SPLIT_TXT = (
    "data_wait=0.050 fill_pulls=0.450 after_pulls=0.320 first_progress=0.820 fx=0.020 "
    "idist0=0.120 fwd=0.190 enq3=0.001 idist1=0.001 bwd=0.297 opt=0.010 deq=0.010"
)
_RANK0_LINE = (
    "[dlstart] timeline s rel. run_start (prestart=on): " + _MARKS_TXT
    + " | split " + _SPLIT_TXT + " | compliance OK"
)
_RANK0_VIOL = (
    "[dlstart] timeline s rel. run_start (prestart=on): " + _MARKS_TXT
    + " train_first_read=-1.000 | split " + _SPLIT_TXT + " | COMPLIANCE VIOLATION"
)
_RANK0_LOOP = (
    "[dlstart] timeline s rel. loop_entry (prestart=off): " + _MARKS_TXT
    + " | split " + _SPLIT_TXT
)
_RANK5_LINE = (
    "[dlstart] rank-timeline rank=5 s rel. run_start (prestart:on): "
    + _MARKS_TXT.replace("=", ":") + " ; split " + _SPLIT_TXT.replace("=", ":")
    + " ; verdict:OK"
)


def _load():  # pyre-ignore[3]
    spec = importlib.util.spec_from_file_location("dl_prestart_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)  # pyre-ignore[6]
    spec.loader.exec_module(mod)  # pyre-ignore[16]
    return mod


class _NoLogs:
    """assertNoLogs for python < 3.10."""

    def __init__(self, tc: unittest.TestCase, logger: logging.Logger) -> None:
        self.tc, self.logger, self.records = tc, logger, []

    def __enter__(self) -> "_NoLogs":
        records = self.records

        class _H(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        self.h = _H(level=logging.DEBUG)
        self.old_level = self.logger.level
        self.logger.addHandler(self.h)
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc: object) -> None:
        self.logger.removeHandler(self.h)
        self.logger.setLevel(self.old_level)
        self.tc.assertEqual([r.getMessage() for r in self.records], [])


class _Base(unittest.TestCase):
    # None = the knob is absent from the environment (gin binding not applied)
    KNOB_VALUE = None  # type: object

    def setUp(self) -> None:
        # load first, set the env after: report() must read it at report time
        self.dlp = _load()
        self.dlp._reset_for_tests()
        self.dlp._MARKS.update(_MARKS)
        self.dlp._REF = ("run_start", _RUN_START)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(_KNOB, None)
        if self.KNOB_VALUE is not None:
            os.environ[_KNOB] = str(self.KNOB_VALUE)

    def _report(self, rank: int, prestarted: bool = True):  # pyre-ignore[3]
        with self.assertLogs(self.dlp.logger, level="INFO") as cm:
            ret = self.dlp.report(rank, prestarted=prestarted)
        return ret, [(r.levelno, r.getMessage()) for r in cm.records]


class TimelineAllRanksTest(_Base):
    """ON (DL_PRESTART_RANK_TIMELINE=1): the dlstart row's expectations."""

    KNOB_VALUE = "1"

    def test_knob_name(self) -> None:
        self.assertEqual(self.dlp.RANK_TIMELINE_ENV, _KNOB)
        self.assertTrue(self.dlp.rank_timeline_enabled())

    def test_rank0_line_unchanged(self) -> None:
        ret, msgs = self._report(0)
        self.assertEqual(ret, _RANK0_LINE)
        self.assertEqual(msgs, [(logging.INFO, _RANK0_LINE)])

    def test_rank0_violation_unchanged(self) -> None:
        self.dlp.new_read_slot("train").value = _RUN_START - 1.0
        ret, msgs = self._report(0)
        self.assertEqual(ret, _RANK0_VIOL)
        self.assertEqual(
            msgs,
            [
                (logging.ERROR, "[dlstart] COMPLIANCE VIOLATION rank=0: sample read before RUN_START: train@-1.000s"),
                (logging.INFO, _RANK0_VIOL),
            ],
        )

    def test_rank0_loop_entry_fallback_unchanged(self) -> None:
        self.dlp._REF = None
        ret, msgs = self._report(0, prestarted=False)
        self.assertEqual(ret, _RANK0_LOOP)
        self.assertEqual(msgs, [(logging.INFO, _RANK0_LOOP)])

    def test_other_rank_logs_rank_timeline(self) -> None:
        ret, msgs = self._report(5)
        self.assertIsNone(ret)
        self.assertEqual(msgs, [(logging.INFO, _RANK5_LINE)])

    def test_other_rank_shares_no_rank0_anchor(self) -> None:
        # the property the readers rely on (torn launcher lines): after
        # 'rank=<r>' there is no ' | ', no '=', no rank-0 tag and no verdict token
        for viol in (False, True):
            for ref in (("run_start", _RUN_START), None):
                self.setUp()
                self.dlp._REF = ref
                if viol:
                    self.dlp.new_read_slot("train").value = _RUN_START - 1.0
                for rank in (1, 7, 31):
                    self.dlp._REPORTED = False
                    ret, msgs = self._report(rank)
                    self.assertIsNone(ret)
                    tl = [m for lv, m in msgs if lv == logging.INFO]
                    self.assertEqual(len(tl), 1)
                    head = "[dlstart] rank-timeline rank=%d" % rank
                    self.assertTrue(tl[0].startswith(head))
                    body = tl[0][len(head):]
                    for bad in (" | ", "=", "[dlstart] timeline", "timeline s rel", "compliance OK",
                                "COMPLIANCE VIOLATION", "| split", "| compliance"):
                        self.assertNotIn(bad, body, (viol, ref, rank, bad))

    def test_idempotent(self) -> None:
        self.dlp.report(3, prestarted=True)
        with _NoLogs(self, self.dlp.logger):
            self.assertIsNone(self.dlp.report(3, prestarted=True))

    def test_violation_on_other_rank(self) -> None:
        self.dlp.new_read_slot("train").value = _RUN_START - 1.0  # a sample read before RUN_START
        ret, msgs = self._report(2)
        self.assertIsNone(ret)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0][0], logging.ERROR)
        self.assertEqual(msgs[0][1], "[dlstart] COMPLIANCE VIOLATION rank=2: sample read before RUN_START: train@-1.000s")
        self.assertTrue(msgs[1][1].startswith("[dlstart] rank-timeline rank=2 s rel. run_start (prestart:on): "))
        self.assertIn(" train_first_read:-1.000", msgs[1][1])
        self.assertTrue(msgs[1][1].endswith(" ; verdict:VIOLATION"))

    def test_non_int_rank_does_not_raise(self) -> None:
        ret, msgs = self._report("3")  # pyre-ignore[6]
        self.assertIsNone(ret)
        self.assertTrue(msgs[-1][1].startswith("[dlstart] rank-timeline rank=3 s rel. "))


class TimelineDefaultOffTest(_Base):
    """OFF (knob absent: the gin default False never reaches report() as "1"):
    byte-identical to 544e691."""

    KNOB_VALUE = None

    def test_disabled(self) -> None:
        self.assertFalse(self.dlp.rank_timeline_enabled())

    def test_rank0_line_unchanged(self) -> None:
        ret, msgs = self._report(0)
        self.assertEqual(ret, _RANK0_LINE)
        self.assertEqual(msgs, [(logging.INFO, _RANK0_LINE)])

    def test_rank0_violation_unchanged(self) -> None:
        self.dlp.new_read_slot("train").value = _RUN_START - 1.0
        ret, msgs = self._report(0)
        self.assertEqual(ret, _RANK0_VIOL)
        self.assertEqual(
            msgs,
            [
                (logging.ERROR, "[dlstart] COMPLIANCE VIOLATION rank=0: sample read before RUN_START: train@-1.000s"),
                (logging.INFO, _RANK0_VIOL),
            ],
        )

    def test_rank0_loop_entry_fallback_unchanged(self) -> None:
        self.dlp._REF = None
        ret, msgs = self._report(0, prestarted=False)
        self.assertEqual(ret, _RANK0_LOOP)
        self.assertEqual(msgs, [(logging.INFO, _RANK0_LOOP)])

    def test_other_ranks_log_nothing(self) -> None:
        for ref in (("run_start", _RUN_START), None):
            for rank in (1, 5, 7, 31, "3"):
                self.dlp._REF = ref
                self.dlp._REPORTED = False
                with _NoLogs(self, self.dlp.logger):
                    self.assertIsNone(self.dlp.report(rank, prestarted=True))  # pyre-ignore[6]
                self.assertTrue(self.dlp._REPORTED)
                self.assertFalse(self.dlp.ACTIVE)

    def test_violation_on_other_rank_logs_only_the_error(self) -> None:
        self.dlp.new_read_slot("train").value = _RUN_START - 1.0
        ret, msgs = self._report(2)
        self.assertIsNone(ret)
        self.assertEqual(
            msgs,
            [(logging.ERROR, "[dlstart] COMPLIANCE VIOLATION rank=2: sample read before RUN_START: train@-1.000s")],
        )

    def test_idempotent(self) -> None:
        with _NoLogs(self, self.dlp.logger):
            self.assertIsNone(self.dlp.report(3, prestarted=True))
            self.assertIsNone(self.dlp.report(3, prestarted=True))


class TimelineKillSwitchTest(TimelineDefaultOffTest):
    """DL_PRESTART_RANK_TIMELINE=0 (what the gin False binds; the kill switch): same as absent."""

    KNOB_VALUE = "0"


class TimelineKnobValuesTest(_Base):
    def test_only_one_arms(self) -> None:
        for raw, on in (("1", True), (" 1 ", True), ("0", False), ("", False), ("2", False),
                        ("off", False), ("false", False)):
            os.environ[_KNOB] = raw
            self.assertEqual(self.dlp.rank_timeline_enabled(), on, repr(raw))
        os.environ.pop(_KNOB, None)
        self.assertFalse(self.dlp.rank_timeline_enabled())

    def test_read_at_report_time(self) -> None:
        # the module was loaded with the knob absent; setting it before the report arms it
        os.environ[_KNOB] = "1"
        ret, msgs = self._report(5)
        self.assertIsNone(ret)
        self.assertEqual(msgs, [(logging.INFO, _RANK5_LINE)])


if __name__ == "__main__":
    unittest.main()
