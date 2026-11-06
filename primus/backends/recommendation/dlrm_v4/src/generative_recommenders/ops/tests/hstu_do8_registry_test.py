#!/usr/bin/env python3

# pyre-unsafe

"""CPU tests of the R15 A11 consumer hook (ops/flydsl/hstu_do8_registry.py,
contract v1.2) and of its wiring.

No GPU, no kernel: dO8/E8 are stand-in tensors of the quantizer's dtypes and
shapes (the registry never looks at their bytes). The forward-gate tests of
hstu_flydsl_mha.c8_do_blk_planned need that module importable (triton
installed, as in the training image); they are skipped otherwise. The gin
binding test needs gin-config (skipped otherwise) and runs its body in a
child interpreter: it clears gin's global config and rewrites the
environment. The Arbor correctness gate collects these classes through
dlrm_v4/tests/hstu_do8_registry_gate_test.py.
"""

import ast
import contextlib
import gc
import io
import os
import re
import subprocess
import sys
import textwrap
import unittest
import weakref

import torch
from generative_recommenders.ops.flydsl import hstu_do8_registry as reg

FP8 = getattr(torch, "float8_e4m3fn", None)
OPS = os.path.dirname(os.path.dirname(os.path.abspath(reg.__file__)))
GR = os.path.dirname(OPS)
ROOT = os.path.dirname(GR)      # the directory that holds generative_recommenders
SHIM = os.path.join(GR, "dlrm_v4", "tests", "hstu_do8_registry_gate_test.py")

try:
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
except Exception as error:  # noqa: BLE001 - no triton in this environment
    mha = None
    MHA_IMPORT_ERROR = repr(error)
else:
    MHA_IMPORT_ERROR = ""

try:
    import gin
except Exception:  # noqa: BLE001 - gin-config not installed
    gin = None

T, H, D, B = 300, 4, 8, 5
SUMMARY = "[hstu-attn] dO8 registry: rank="
ARMED = "[hstu-attn] dO8 registry ARMED: "
REQUESTED = "[hstu-attn] dO8 registry REQUESTED: "


def offsets(lens=(1, 63, 64, 65, 107)):
    o = [0]
    for n in lens:
        o.append(o[-1] + n)
    assert o[-1] == T
    return torch.tensor(o, dtype=torch.int64)


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Base(unittest.TestCase):
    """The knob ON (DLRM_HSTU_FP8_BWD_DO8=1) unless KNOB is None; the retired
    DLRM_HSTU_FP8_BWD_DO8_TAKE unset; stdout captured."""

    KNOB = "1"

    def setUp(self) -> None:
        reg.reset_for_testing()
        self._env = {k: os.environ.pop(k, None) for k in (reg.KNOB_ENV, reg.TAKE_ENV)}
        if self.KNOB is not None:
            os.environ[reg.KNOB_ENV] = self.KNOB
        self.out = io.StringIO()
        self._redirect = contextlib.redirect_stdout(self.out)
        self._redirect.__enter__()

    def tearDown(self) -> None:
        self._redirect.__exit__(None, None, None)
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        reg.reset_for_testing()

    def lines(self):
        return self.out.getvalue().splitlines()

    def summaries(self):
        return [x for x in self.lines() if x.startswith(SUMMARY)]

    def site(self, N=128):
        out = torch.zeros(T, H, D, dtype=torch.bfloat16)
        offs = offsets()
        site = reg.new_site(out, offs, N)
        self.assertIsNotNone(site)
        return out, offs, site

    def produce(self, site, two_d=True, writes=0):
        dx = torch.randn(T, H * D).to(torch.bfloat16)
        for _ in range(writes):          # tracked in-place writes before publish
            dx.mul_(1)
        if not two_d:
            dx = dx.view(T, H, D)
        do8 = torch.randn(T, H * D).to(FP8)
        e8 = torch.randint(0, 255, (T, H), dtype=torch.uint8)
        return dx, do8, e8

    def publish(self, two_d=True, N=128, writes=0):
        out, offs, site = self.site(N)
        dx, do8, e8 = self.produce(site, two_d, writes)
        rec = reg.publish(site, dx, do8, e8, producer="test")
        self.assertIsNotNone(rec)
        dout = dx.view(T, H, D)
        self.assertTrue(reg.bind(site, dout))
        return site, offs, dx, do8, e8, rec, dout

    def assert_declined(self, why, dout, offs, N=None):
        before = reg.stats()["declined"].get(why, 0)
        self.assertIsNone(reg.take(dout, offs, N))
        self.assertEqual(reg.stats()["declined"].get(why, 0), before + 1, reg.stats())


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class RoundTripTest(_Base):
    def test_round_trip_returns_the_published_bytes(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        got = reg.take(dout, offs, 128)
        self.assertIsNotNone(got)
        g8, ge = got
        self.assertEqual(tuple(g8.shape), (T, H, D))
        self.assertIs(g8.dtype, FP8)
        self.assertEqual(g8.data_ptr(), do8.data_ptr())
        self.assertTrue(torch.equal(g8.view(torch.uint8).view(T, H * D), do8.view(torch.uint8)))
        self.assertIs(ge, e8)
        self.assertEqual(rec.state, "consumed")
        self.assertIsNone(rec.do8)
        self.assertIsNone(rec.e8)
        s = reg.stats()
        self.assertEqual((s["sites"], s["published"], s["taken"]), (1, 1, 1))
        self.assertEqual(s["declined"], {})
        self.assertEqual(s["discarded"], {})

    def test_3d_dx_is_refused_shape(self) -> None:
        # v1.2: dx must be the d(attn) tensor itself, attn's [T, H*D] shape.
        # A [T, H, D] dx (or a 3-D view of the real one) is refused, nothing
        # is tagged, and the consumer quantizes as today.
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site, two_d=False)
        self.assertIsNone(reg.publish(site, dx, do8.view(T, H, D), e8))
        self.assertEqual(reg.stats()["refused"], {"shape": 1})
        self.assertIsNone(site.record)
        self.assertFalse(hasattr(dx, reg.HSTU_DO8_RECORD_ATTR))
        self.assertIsNone(reg.take(dx, offs))
        # A [T, H, D] do8 is fine: only its element count is checked.
        dx2 = dx.reshape(T, H * D)
        self.assertIsNotNone(reg.publish(site, dx2, do8.view(T, H, D), e8))
        reg.bind(site, dx2.view(T, H, D))

    def test_take_of_the_published_dx_object_itself(self) -> None:
        # The [T, H*D] dx object itself as dO passes the identity clause
        # (dout is dx) and declines on shape: the consumer only takes a
        # [T, H, D] dO.
        site, offs, dx, do8, e8, rec, dout = self.publish()
        self.assert_declined("shape", dx, offs)

    def test_view_of_a_view_of_dx(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        vv = dx.view(-1).view(T, H, D)        # _base is dx (views flatten to the root)
        self.assertIs(vv._base, dx)
        setattr(vv, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assertIsNotNone(reg.take(vv, offs))

    def test_dx_itself_a_view_of_a_larger_buffer(self) -> None:
        # The producer may return a contiguous view of a bigger allocation:
        # dout (autograd's view of dx) then has the buffer as its _base.
        out, offs, site = self.site()
        buf = torch.randn(2 * T * H * D).to(torch.bfloat16)
        dx = buf[:T * H * D].view(T, H * D)
        dx2, do8, e8 = self.produce(site)
        rec = reg.publish(site, dx, do8, e8)
        self.assertIsNotNone(rec)
        dout = dx.view(T, H, D)
        self.assertIs(dout._base, buf)
        reg.bind(site, dout)
        self.assertIsNotNone(reg.take(dout, offs))
        # A tracked write anywhere in the buffer bumps the shared counter:
        # the next record on the same buffer declines on version.
        rec2 = reg.publish(site, dx, do8, e8)
        buf[T * H * D:].zero_()
        dout2 = dx.view(T, H, D)
        reg.bind(site, dout2)
        self.assert_declined("version", dout2, offs)
        self.assertEqual(rec2.state, "declined")

    def test_one_shot_second_take_declines_consumed(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        first = reg.take(dout, offs, 128)
        self.assertIsNotNone(first)
        # The M-E1 exact re-run takes the same dout again: quantizer, not
        # the record.
        self.assert_declined("consumed", dout, offs, 128)
        self.assertEqual(reg.stats()["taken"], 1)
        self.assertTrue(torch.equal(first[0].view(torch.uint8).view(T, H * D),
                                    do8.view(torch.uint8)))

    def test_no_record_and_no_site(self) -> None:
        t = torch.zeros(T, H, D, dtype=torch.bfloat16)
        self.assertIsNone(reg.take(t, offsets(), 128))
        self.assertIsNone(reg.site_of(t))
        self.assertFalse(reg.discard(t, "x"))
        self.assertFalse(reg.bind(None, t))
        dst = t.view(T, H * D)
        self.assertIs(reg.carry_site(t, dst), dst)
        self.assertIsNone(reg.site_of(dst))
        self.assertEqual(reg.stats()["events"], 0)
        self.assertEqual(self.lines(), [])

    def test_site_tag_carry_and_site_of(self) -> None:
        out, offs, site = self.site(N=96)
        self.assertIs(getattr(out, reg.HSTU_DO8_SITE_ATTR), site)
        attn = out.view(-1, H * D)
        self.assertIsNone(reg.site_of(attn))       # a view carries nothing
        attn = reg.carry_site(out, out.view(-1, H * D))
        self.assertIs(reg.site_of(attn), site)
        self.assertIs(site.seq_offsets, offs)
        self.assertEqual((site.N, site.T, site.H, site.D), (96, T, H, D))
        # A tensor of another size is not this site's attn.
        other = torch.zeros(T - 1, H * D, dtype=torch.bfloat16)
        setattr(other, reg.HSTU_DO8_SITE_ATTR, site)
        self.assertIsNone(reg.site_of(other))
        # fx Proxy-like objects pass through untouched.
        self.assertEqual(reg.carry_site("proxy", "proxy2"), "proxy2")

    def test_new_site_rejects_malformed_calls(self) -> None:
        offs = offsets()
        self.assertIsNone(reg.new_site(torch.zeros(T, H * D), offs, 1))
        self.assertIsNone(reg.new_site(torch.zeros(T, H, D), offs.view(1, -1), 1))
        self.assertIsNone(reg.new_site(torch.zeros(T, H, D), torch.zeros(0), 1))
        self.assertEqual(reg.stats()["sites"], 0)


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class DeclineTest(_Base):
    """One decline per take() check; each releases the record's tensors."""

    def test_released_after_discard(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        self.assertTrue(reg.discard(dout, "unit"))
        self.assert_declined("released", dout, offs)

    def test_kill_switch(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        with env(**{reg.KNOB_ENV: "0"}):
            self.assert_declined("kill_switch", dout, offs)
        self.assertIsNone(rec.do8)

    def test_retired_take_switch_still_forces_off(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        with env(**{reg.TAKE_ENV: "0"}):
            self.assert_declined("kill_switch", dout, offs)
        retired = [x for x in self.lines() if "(retired v1 kill switch)" in x]
        self.assertEqual(len(retired), 1, self.lines())

    def test_producer_freed(self) -> None:
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site)
        rec = reg.publish(site, dx, do8, e8)
        other = torch.zeros(T, H, D, dtype=torch.bfloat16)
        del dx
        gc.collect()
        self.assertIsNone(rec.dx_ref())
        self.assertTrue(reg.bind(site, other))
        self.assert_declined("producer_freed", other, offs)

    def test_data_ptr_reuse_after_the_producer_tensor_died(self) -> None:
        # The allocator may hand dx's address to a new tensor of the same
        # size: emulate exactly that (same data_ptr, numel and version) and
        # require a decline, because the producer tensor is gone.
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site)
        rec = reg.publish(site, dx, do8, e8)
        del dx
        gc.collect()
        new = torch.zeros(T, H, D, dtype=torch.bfloat16)
        rec.dx_ptr = new.data_ptr()
        rec.version = new._version
        reg.bind(site, new)
        self.assert_declined("producer_freed", new, offs)

    def test_storage_other_tensor(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        other = dout.clone()
        setattr(other, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("storage", other, offs)

    def test_storage_offset_view(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        part = dx.view(-1)[H * D:].view(T - 1, H, D)
        setattr(part, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("storage", part, offs)

    def test_storage_data_alias_with_its_own_version_counter(self) -> None:
        # Review finding (v1 fail-open): x.data shares dx's storage but has
        # its own version counter. dx was written once before publish
        # (version 1) and the alias once after (its own counter: also 1), so
        # data_ptr, numel, shape, dtype AND the _version number all match;
        # the bytes behind the record are stale. v1.1 declines on identity.
        site, offs, dx, do8, e8, rec, dout = self.publish(writes=1)
        self.assertEqual(rec.version, 1)
        alias = dx.data
        alias.mul_(2)                      # tracked only on the alias
        stale = alias.view(T, H, D)
        self.assertEqual((stale.data_ptr(), stale._version, dx._version), (dx.data_ptr(), 1, 1))
        setattr(stale, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("storage", stale, offs)

    def test_storage_set_alias_with_its_own_version_counter(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        alias = torch.empty(0, dtype=torch.bfloat16)
        alias.set_(dx.untyped_storage(), dx.storage_offset(), (T, H, D), (H * D, D, 1))
        self.assertEqual(alias.data_ptr(), dx.data_ptr())
        setattr(alias, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("storage", alias, offs)

    def test_storage_detached_alias_declines(self) -> None:
        # detach() shares dx's counter but is no autograd view of dx: declined
        # (fail-closed; no production path hands the consumer a detached dO).
        site, offs, dx, do8, e8, rec, dout = self.publish()
        det = dx.detach().view(T, H, D)
        setattr(det, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("storage", det, offs)

    def test_version_bump_after_publish(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        dx.mul_(1)        # any tracked write after publish
        self.assert_declined("version", dout, offs)

    def test_dtype_view_of_dx_declines_storage(self) -> None:
        # A dtype view is a non-differentiable view: its _base is not dx, so
        # it declines on identity before the dtype clause.
        site, offs, dx, do8, e8, rec, dout = self.publish()
        as_int = dx.view(torch.int16).view(T, H, D)  # same bytes, other dtype
        setattr(as_int, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("storage", as_int, offs)

    def test_dtype_of_the_record(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        rec.e8 = rec.e8.view(torch.int8)     # a record that slipped past publish
        self.assert_declined("dtype", dout, offs)

    def test_shape_dout(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        flat = dx.view(T * H, D)
        setattr(flat, reg.HSTU_DO8_RECORD_ATTR, rec)
        self.assert_declined("shape", flat, offs)

    def test_shape_e8(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        rec.e8 = e8[:-1]   # a record that slipped past publish's checks
        self.assert_declined("shape", dout, offs)

    def test_layout_non_contiguous_dout(self) -> None:
        # H == D so the transposed view keeps dout's [T, H, D] shape and
        # data_ptr: only the element order differs.
        global H, D
        h, d = H, D
        H = D = 4
        try:
            site, offs, dx, do8, e8, rec, dout = self.publish()
            tr = dx.view(T, H, D).transpose(1, 2)
            setattr(tr, reg.HSTU_DO8_RECORD_ATTR, rec)
            self.assert_declined("layout", tr, offs)
        finally:
            H, D = h, d

    def test_align(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        raw = torch.zeros(T * H * D + 1, dtype=torch.uint8)
        rec.do8 = raw[1:].view(FP8)          # contiguous, misaligned
        self.assert_declined("align", dout, offs)

    def test_device(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        rec.device = torch.device("meta")
        self.assert_declined("device", dout, offs)

    def test_stream(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        real = reg._stream_key
        reg._stream_key = lambda device: 12345
        try:
            self.assert_declined("stream", dout, offs)
        finally:
            reg._stream_key = real

    def test_seq_offsets_other_tensor(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        self.assert_declined("seq_offsets", dout, offs.clone())

    def test_seq_offsets_written_after_publish(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        offs.add_(0)
        self.assert_declined("seq_offsets", dout, offs)

    def test_n_mismatch(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish(N=128)
        self.assert_declined("n", dout, offs, 129)

    def test_decline_releases_the_payload(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        w8, we = weakref.ref(do8), weakref.ref(e8)
        del do8, e8
        self.assertIsNotNone(w8())
        self.assert_declined("seq_offsets", dout, offs.clone())
        gc.collect()
        self.assertIsNone(w8())
        self.assertIsNone(we())


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class PublishTest(_Base):
    def test_only_mode_refused_with_error(self) -> None:
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site)
        with self.assertRaisesRegex(ValueError, "contract v2"):
            reg.publish(site, dx, do8, e8, bf16_written=False)
        self.assertIsNone(site.record)
        self.assertFalse(hasattr(dx, reg.HSTU_DO8_RECORD_ATTR))
        self.assertEqual(reg.stats()["refused"], {"only": 1})

    def test_refusals(self) -> None:
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site)
        cases = {
            "site": (object(), dx, do8, e8),
            "type": (site, dx, do8, "e8"),
            "dtype": (site, dx.float(), do8, e8),
            "shape": (site, dx, do8, e8[:-1]),
            "layout": (site, dx.t().contiguous().t(), do8, e8),
            "align": (site, dx, torch.zeros(T * H * D + 1, dtype=torch.uint8)[1:].view(FP8), e8),
        }
        for why, args in cases.items():
            self.assertIsNone(reg.publish(*args), why)
            self.assertEqual(reg.stats()["refused"].get(why), 1, why)
        with env(**{reg.KNOB_ENV: "0"}):
            self.assertIsNone(reg.publish(site, dx, do8, e8))
        self.assertEqual(reg.stats()["refused"].get("kill_switch"), 1)
        # v1.2: dx of the right size but not attn's [T, H*D] shape
        self.assertIsNone(reg.publish(site, dx.view(T, H, D), do8, e8))
        self.assertIsNone(reg.publish(site, dx.view(T * H, D), do8, e8))
        self.assertEqual(reg.stats()["refused"].get("shape"), 3)
        self.assertIsNone(site.record)
        self.assertEqual(reg.stats()["published"], 0)

    def test_superseded_record_is_discarded(self) -> None:
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site)
        first = reg.publish(site, dx, do8, e8)
        second = reg.publish(site, dx, do8, e8)
        self.assertEqual(first.state, "discarded")
        self.assertIs(site.record, second)
        self.assertEqual(reg.stats()["discarded"], {"superseded": 1})

    def test_record_holds_no_strong_reference_to_dx(self) -> None:
        out, offs, site = self.site()
        dx, do8, e8 = self.produce(site)
        reg.publish(site, dx, do8, e8)
        w = weakref.ref(dx)
        del dx
        gc.collect()
        self.assertIsNone(w())


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class DiscardBindTest(_Base):
    def test_discard_by_dout_and_by_site(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        self.assertTrue(reg.discard(dout, "not_c8"))
        self.assertIsNone(rec.do8)
        self.assertFalse(reg.discard(dout, "not_c8"))      # once
        self.assertFalse(reg.discard(site, "not_taken"))   # already released
        site2, offs2, *_ = self.publish()
        self.assertTrue(reg.discard(site2, "not_taken"))
        self.assertEqual(reg.stats()["discarded"], {"not_c8": 1, "not_taken": 1})

    def test_discard_after_take_is_a_noop(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        self.assertIsNotNone(reg.take(dout, offs))
        self.assertFalse(reg.discard(site, "not_taken"))
        self.assertEqual(reg.stats()["discarded"], {})

    def test_bind_without_pending_record(self) -> None:
        out, offs, site = self.site()
        t = torch.zeros(T, H, D, dtype=torch.bfloat16)
        self.assertFalse(reg.bind(site, t))
        self.assertFalse(hasattr(t, reg.HSTU_DO8_RECORD_ATTR))
        self.assertEqual(reg.stats()["unbound"], 1)

    def test_unbound_is_quiet_exact_and_on_every_summary_line(self) -> None:
        # v1.2: a site reaching the attention backward without a pending
        # record counts `unbound`; no event, no line, no exit hook.
        saved = reg._STATE["atexit"]
        reg._STATE["atexit"] = False      # a process-wide latch: restored below
        try:
            out, offs, site = self.site()
            t = torch.zeros(T, H, D, dtype=torch.bfloat16)
            self.assertFalse(reg.bind(site, t))          # never published
            self.assertFalse(reg.bind(None, t))          # no site: not counted
            self.assertFalse(reg.bind(site, "not a tensor"))
            self.assertEqual(self.lines(), [])
            self.assertFalse(reg._STATE["atexit"])
            s = reg.stats()
            self.assertEqual((s["sites"], s["unbound"], s["events"]), (1, 1, 0))
        finally:
            reg._STATE["atexit"] = saved
        site2, offs2, dx, do8, e8, rec, dout = self.publish()   # bound: not counted
        self.assertIsNotNone(reg.take(dout, offs2))
        self.assertFalse(reg.bind(site2, dx.view(T, H, D)))     # consumed record
        self.assertEqual(reg.stats()["unbound"], 2)
        self.assertIn(" taken=1 unbound=2 declined={} ", reg.summary_line())
        self.assertTrue(all(" unbound=" in x for x in self.summaries()), self.lines())


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class KnobTest(_Base):
    KNOB = None

    def test_values(self) -> None:
        for raw, on in ((None, False), ("", False), ("0", False), (" 0 ", False),
                        ("1", True), (" 1 ", True)):
            with env(**{reg.KNOB_ENV: raw}):
                self.assertEqual(reg.take_enabled(), on, raw)
                self.assertEqual(reg.sites_enabled(), on, raw)
        # Only "1" arms (v1.2 MINOR 5): the truthy spellings some other
        # DLRM_HSTU_FP8_* knobs accept stay off here, with one line each.
        invalid = ("true", "True", "on", "yes", "off", "2", "01")
        for raw in invalid:
            with env(**{reg.KNOB_ENV: raw}):
                self.assertFalse(reg.take_enabled())
                self.assertFalse(reg.sites_enabled())
        for raw in invalid:
            warn = [x for x in self.lines() if f"{reg.KNOB_ENV}={raw!r} is neither 0 nor 1" in x]
            self.assertEqual(len(warn), 1, (raw, self.lines()))
        req = [x for x in self.lines() if x.startswith(REQUESTED)]
        self.assertEqual(len(req), 1, self.lines())
        self.assertRegex(req[0], r"^\[hstu-attn\] dO8 registry REQUESTED: rank=-?\d+ pid=\d+ dev=\S+ "
                         r"DLRM_HSTU_FP8_BWD_DO8=1: ")
        self.assertIn("contract v1.2 'plus'", req[0])
        self.assertEqual(len(self.lines()), 1 + len(invalid), self.lines())
        self.assertEqual(reg.stats()["events"], 0)

    def test_default_is_off_and_silent(self) -> None:
        # The consumer-only tree with the default (no binding, no env): no
        # site, no line, no event, no exit line.
        self.assertNotIn(reg.KNOB_ENV, os.environ)
        self.assertFalse(reg.sites_enabled())
        self.assertFalse(reg.take_enabled())
        t = torch.zeros(T, H, D, dtype=torch.bfloat16)
        self.assertIsNone(reg.take(t, offsets(), 128))
        self.assertFalse(reg.discard(t, "not_c8"))
        self.assertFalse(reg.bind(None, t))
        self.assertIs(reg.carry_site(t, t.view(T, H * D))._base, t)
        reg._final()
        self.assertEqual(self.lines(), [])
        self.assertEqual(reg.stats()["events"], 0)
        self.assertIn("take=off", reg.summary_line())

    def test_retired_take_switch(self) -> None:
        cases = (("1", "0", False), ("1", "garbage", False), ("1", "1", True), ("1", "", True),
                 (None, "1", False), (None, "0", False))
        for knob, take, on in cases:
            with env(**{reg.KNOB_ENV: knob, reg.TAKE_ENV: take}):
                self.assertEqual(reg.take_enabled(), on, (knob, take))
        retired = [x for x in self.lines() if "(retired v1 kill switch)" in x]
        self.assertEqual(len(retired), 2, self.lines())   # '0' and 'garbage', each once

    def test_requested_line_once(self) -> None:
        with env(**{reg.KNOB_ENV: "1"}):
            for _ in range(3):
                self.assertTrue(reg.take_enabled())
            reg.summary_line()
        self.assertEqual(len([x for x in self.lines() if x.startswith(REQUESTED)]), 1, self.lines())
        self.assertEqual(self.summaries(), [])


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class CounterLogTest(_Base):
    RE = re.compile(
        r"^\[hstu-attn\] dO8 registry: rank=(?P<rank>-?\d+) pid=(?P<pid>\d+) dev=(?P<dev>\S+) "
        r"events=(?P<events>\d+) sites=(?P<sites>\d+) published=(?P<published>\d+) "
        r"taken=(?P<taken>\d+) unbound=(?P<unbound>\d+) "
        r"declined=\{(?P<declined>.*?)\} discarded=\{(?P<discarded>.*?)\} "
        r"refused=\{(?P<refused>.*?)\} take=(?P<take>on|off) contract=v1\.2(?P<final> final=1)?$")

    def events(self):
        return [int(m.group("events")) for m in map(self.RE.match, self.lines()) if m]

    def test_sites_are_quiet(self) -> None:
        saved = reg._STATE["atexit"]
        reg._STATE["atexit"] = False      # a process-wide latch: restored below
        try:
            for _ in range(9):
                self.site()
            self.assertFalse(reg._STATE["atexit"])   # no event, so no exit hook
        finally:
            reg._STATE["atexit"] = saved
        self.assertEqual(self.lines(), [])
        s = reg.stats()
        self.assertEqual((s["sites"], s["events"]), (9, 0))
        reg._final()
        self.assertEqual(self.lines(), [])

    def test_log_spaced_summary_and_one_arming_line(self) -> None:
        for _ in range(9):
            self.site()                                       # quiet
        for _ in range(4):
            site, offs, dx, do8, e8, rec, dout = self.publish()   # events 1, 3, 5, 7
            self.assertIsNotNone(reg.take(dout, offs))           # events 2, 4, 6, 8
        self.assertTrue(all(map(self.RE.match, self.summaries())), self.lines())
        armed = [x for x in self.lines() if x.startswith(ARMED)]
        self.assertEqual(len(armed), 1, self.lines())
        self.assertIn("producer=test", armed[0])
        self.assertIn("kill switch DLRM_HSTU_FP8_BWD_DO8=0", armed[0])
        # 1, the forced line right after the arming line (2), 4, 8
        self.assertEqual(self.events(), [1, 2, 4, 8])
        self.assertEqual(len([x for x in self.lines() if x.startswith(REQUESTED)]), 1)
        s = reg.stats()
        self.assertEqual((s["sites"], s["published"], s["taken"], s["events"]),
                         (13, 4, 4, 8))

    def test_first_occurrence_of_a_reason_prints(self) -> None:
        for _ in range(2):
            site, offs, dx, do8, e8, rec, dout = self.publish()   # 1, 3
            reg.take(dout, offs)                                  # 2, 4
        for _ in range(3):
            site, offs, dx, do8, e8, rec, dout = self.publish()   # 5, 7, 9
            reg.take(dout, offs.clone())                          # 6 (new reason), 8, 10
        ms = [m for m in map(self.RE.match, self.lines()) if m]
        self.assertEqual([int(m.group("events")) for m in ms], [1, 2, 4, 6, 8])
        self.assertEqual(ms[-1].group("declined"), "seq_offsets:2")

    def test_summary_fields_and_final(self) -> None:
        site, offs, dx, do8, e8, rec, dout = self.publish()
        reg.discard(dout, "not_c8")
        with env(RANK="3"):
            line = reg.summary_line(final=True)
        m = self.RE.match(line)
        self.assertIsNotNone(m, line)
        self.assertEqual(m.group("rank"), "3")
        self.assertEqual(m.group("dev"), "cpu")
        self.assertEqual(m.group("unbound"), "0")
        self.assertEqual(m.group("discarded"), "not_c8:1")
        self.assertEqual(m.group("take"), "on")
        self.assertEqual(m.group("final"), " final=1")

    def test_final_only_after_a_publish_or_a_refusal(self) -> None:
        out, offs, site = self.site()
        reg._final()
        self.assertEqual([x for x in self.lines() if x.endswith(" final=1")], [])
        dx, do8, e8 = self.produce(site)
        self.assertIsNone(reg.publish(site, dx.float(), do8, e8))      # refused{dtype}
        reg._final()
        finals = [x for x in self.lines() if x.endswith(" final=1")]
        self.assertEqual(len(finals), 1, self.lines())
        self.assertIn("refused={dtype:1}", finals[0])
        reg.reset_for_testing()
        self.publish()
        reg._final()
        finals = [x for x in self.lines() if x.endswith(" final=1")]
        self.assertEqual(len(finals), 2, self.lines())
        self.assertIn(" published=1 ", finals[1])


@unittest.skipIf(FP8 is None, "torch without float8_e4m3fn")
class AutogradRoundTripTest(_Base):
    """The production chain on CPU autograd: the attention Function tags out
    with a site, hstu_compute's view carries it, the producer Function
    publishes on the dx it returns, autograd's view backward hands the
    attention backward a NEW view of dx, which binds and takes."""

    def run_chain(self, publish=True, hook=None, write_in_place=False):
        seen = {}
        offs = offsets()

        class Attn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = x.detach().view(T, H, D).clone()
                ctx.do8_site = reg.new_site(out, offs, 128)
                return out

            @staticmethod
            def backward(ctx, dout):
                reg.bind(ctx.do8_site, dout)
                seen["dout_is_dx"] = dout is seen["dx_ref"]()
                seen["dout_base_is_dx"] = dout._base is seen["dx_ref"]()
                seen["got"] = reg.take(dout.contiguous(), offs, 128)
                reg.discard(ctx.do8_site, "not_taken")
                # A clone: x.grad must not keep a view of dx alive.
                return dout.reshape(T, H * D).clone()

        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, attn):
                ctx.site = reg.site_of(attn)
                return attn * 2

            @staticmethod
            def backward(ctx, g):
                if write_in_place:          # dx written by a tracked op (version 1)
                    dx = torch.empty_like(g)
                    dx.copy_(g * 2)
                else:                       # a fresh result (version 0), like a raw kernel
                    dx = (g * 2).contiguous()
                seen["dx_ref"] = weakref.ref(dx)
                if publish and ctx.site is not None:
                    seen["do8"] = torch.zeros(T, H * D, dtype=torch.uint8).view(FP8)
                    seen["e8"] = torch.zeros(T, H, dtype=torch.uint8)
                    seen["rec"] = reg.publish(ctx.site, dx, seen["do8"], seen["e8"])
                return dx

        x = torch.randn(T, H * D).to(torch.bfloat16).requires_grad_()
        out = Attn.apply(x)
        attn = reg.carry_site(out, out.view(-1, H * D))
        y = Producer.apply(attn)
        if hook is not None:
            y = hook(y, attn)
        y.float().sum().backward()
        return seen

    def test_take_through_autograd(self) -> None:
        for write_in_place in (False, True):
            reg.reset_for_testing()
            seen = self.run_chain(write_in_place=write_in_place)
            self.assertFalse(seen["dout_is_dx"])          # a new view object ...
            self.assertTrue(seen["dout_base_is_dx"])      # ... of dx
            self.assertIsNotNone(seen["got"])
            self.assertEqual(seen["got"][0].data_ptr(), seen["do8"].data_ptr())
            self.assertEqual(seen["rec"].state, "consumed")
            s = reg.stats()
            self.assertEqual((s["sites"], s["published"], s["taken"]), (1, 1, 1))
            self.assertEqual((s["declined"], s["discarded"]), ({}, {}))
            gc.collect()
            self.assertIsNone(seen["dx_ref"]())            # nothing pins dx

    def test_no_producer_no_record(self) -> None:
        seen = self.run_chain(publish=False)
        self.assertIsNone(seen["got"])
        s = reg.stats()
        self.assertEqual((s["sites"], s["published"], s["taken"], s["events"]), (1, 0, 0, 0))
        self.assertEqual(s["unbound"], 1)       # v1.2: the site reached bind() without a record
        reg._final()
        self.assertEqual(self.summaries(), [])

    def test_partial_arming_counts_unbound(self) -> None:
        # Two ordinary layers; the producer publishes for one of them only
        # (L11 declined or latched on the other): one take, one unbound.
        offs = offsets()

        class Attn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = x.detach().view(T, H, D).clone()
                ctx.do8_site = reg.new_site(out, offs, 128)
                return out

            @staticmethod
            def backward(ctx, dout):
                reg.bind(ctx.do8_site, dout)
                reg.take(dout.contiguous(), offs, 128)
                reg.discard(ctx.do8_site, "not_taken")
                return dout.reshape(T, H * D).clone()

        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, attn, emit):
                ctx.site, ctx.emit = reg.site_of(attn), emit
                return attn * 2

            @staticmethod
            def backward(ctx, g):
                dx = (g * 2).contiguous()
                if ctx.emit and ctx.site is not None:
                    reg.publish(ctx.site, dx, torch.zeros(T, H * D, dtype=torch.uint8).view(FP8),
                                torch.zeros(T, H, dtype=torch.uint8))
                return dx, None

        loss = 0
        for emit in (True, False):
            x = torch.randn(T, H * D).to(torch.bfloat16).requires_grad_()
            out = Attn.apply(x)
            y = Producer.apply(reg.carry_site(out, out.view(-1, H * D)), emit)
            loss = loss + y.float().sum()
        loss.backward()
        s = reg.stats()
        self.assertEqual((s["sites"], s["published"], s["taken"], s["unbound"]), (2, 1, 1, 1), s)
        self.assertEqual((s["declined"], s["discarded"], s["refused"]), ({}, {}, {}), s)

    def test_eval_forward_creates_no_site(self) -> None:
        # v1.2 MINOR 1: the tpa site guard reads ctx.needs_input_grad[0] (x,
        # the activation input; pinned by WiringTest). Under no_grad the
        # Function's needs_input_grad still reflects requires_grad, so a
        # Parameter input keeps any(needs_input_grad) True (v1.1's guard) while
        # x, computed under no_grad, needs none.
        offs = offsets()
        w = torch.nn.Parameter(torch.ones(H * D, dtype=torch.bfloat16))
        seen = []

        class Attn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                out = (x.detach() * weight.detach()).view(T, H, D).clone()
                seen.append((bool(ctx.needs_input_grad[0]), any(ctx.needs_input_grad)))
                if reg.sites_enabled() and bool(ctx.needs_input_grad[0]):
                    ctx.do8_site = reg.new_site(out, offs, 128)
                return out

            @staticmethod
            def backward(ctx, dout):
                return dout.reshape(T, H * D).clone(), None

        with torch.no_grad():
            x = torch.randn(T, H * D).to(torch.bfloat16)
            Attn.apply(x, w)
        self.assertEqual(seen[-1], (False, True))
        self.assertEqual(reg.stats()["sites"], 0)
        if hasattr(torch, "inference_mode"):
            with torch.inference_mode():
                Attn.apply(torch.randn(T, H * D).to(torch.bfloat16), w)
            self.assertEqual(reg.stats()["sites"], 0)
        x = torch.randn(T, H * D).to(torch.bfloat16).requires_grad_()
        Attn.apply(x, w)                       # a training forward
        self.assertEqual(seen[-1], (True, True))
        self.assertEqual(reg.stats()["sites"], 1)

    def test_accumulated_gradient_declines(self) -> None:
        # A second consumer of the [T, H*D] attention view makes autograd sum
        # two gradients: the dO the attention backward sees is no longer the
        # published dx.
        seen = self.run_chain(hook=lambda y, attn: y + attn.float())
        self.assertIsNone(seen["got"])
        s = reg.stats()
        self.assertEqual(s["taken"], 0)
        self.assertEqual(sum(s["declined"].values()) + sum(s["discarded"].values()), 1, s)

    def test_hook_returning_a_data_alias_declines(self) -> None:
        # Review finding (v1 fail-open): a gradient hook on attn keeps dx
        # alive and REPLACES the gradient by g.data after an in-place write.
        # The alias has dx's data_ptr, numel, shape, dtype and (dx written
        # once in place, the alias once) the same _version number, but its
        # own version counter: the record's bytes are stale. Declined.
        keep = []

        def h(g):
            keep.append(g)
            return g.data.mul_(2)

        def hook(y, attn):
            attn.register_hook(h)
            return y

        seen = self.run_chain(hook=hook, write_in_place=True)
        self.assertIsNone(seen["got"])
        self.assertEqual(reg.stats()["taken"], 0)
        self.assertEqual(reg.stats()["declined"], {"storage": 1}, reg.stats())


@unittest.skipIf(mha is None, f"hstu_flydsl_mha not importable: {MHA_IMPORT_ERROR}")
class ForwardGateTest(unittest.TestCase):
    """hstu_flydsl_mha.c8_do_blk_planned on CPU, with the host check and the
    kernel module stubbed (the tree's real SHIPPED dict, read from source)."""

    ENV = {
        "HSTU_ATTN_BWD": "flydsl",
        "DLRM_HSTU_FP8_BWD_FULL": "1",
        "DLRM_HSTU_FP8_BWD_QK_MFMA": "fp8",
        "DLRM_HSTU_FP8_BWD_K128": None,
        "DLRM_HSTU_FP8_BWD_FULL_HYBRID": None,
        "DLRM_HSTU_FP8_BWD_FULL_DOBLK": None,
        "DLRM_HSTU_FP8_BWD_FULL_SAT": None,
    }

    @staticmethod
    def shipped():
        src = open(os.path.join(OPS, "flydsl", "hstu_bwd_dkdv.py")).read()
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "SHIPPED" for t in node.targets):
                return ast.literal_eval(node.value)
        raise AssertionError("no SHIPPED in hstu_bwd_dkdv.py")

    def setUp(self) -> None:
        self.kernel = type("K", (), {"SHIPPED": self.shipped()})
        self._saved = (mha.unsupported_hardware_reason, mha._load_kernel,
                       mha._host_ok, mha._targets_max_memo)
        mha.unsupported_hardware_reason = lambda: None
        mha._load_kernel = lambda: self.kernel
        mha._targets_max_memo = (None, -1, 0)
        self._env = env(**self.ENV)
        self._env.__enter__()

    def tearDown(self) -> None:
        self._env.__exit__(None, None, None)
        (mha.unsupported_hardware_reason, mha._load_kernel, mha._host_ok,
         mha._targets_max_memo) = self._saved

    def args(self, q_dtype=FP8, kv_dtype=FP8, Tq=64, heads=4, num_targets=None):
        q = torch.zeros(Tq, heads, 128).to(q_dtype)
        k = torch.zeros(64, heads, 128).to(kv_dtype)
        v = torch.zeros(64, heads, 128).to(kv_dtype)
        out = torch.zeros(Tq, heads, 128, dtype=torch.bfloat16)
        return dict(q=q, k=k, v=v, out=out, num_targets=num_targets,
                    max_attn_len=0, contextual_seq_len=0, num_softmax_heads=0,
                    qkv_scale=0.0173)

    def test_eligible_call(self) -> None:
        self.assertTrue(mha.c8_do_blk_planned(**self.args()))

    def test_bf16_qk_calibration_step(self) -> None:
        self.assertFalse(mha.c8_do_blk_planned(
            **self.args(q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16)))

    def test_bf16_q_e4m3_kv(self) -> None:
        self.assertFalse(mha.c8_do_blk_planned(**self.args(q_dtype=torch.bfloat16)))

    def test_targets_only_compact_q(self) -> None:
        self.assertFalse(mha.c8_do_blk_planned(**self.args(Tq=5)))

    def test_knobs(self) -> None:
        for kv in ({"DLRM_HSTU_FP8_BWD_FULL": "0"},
                   {"DLRM_HSTU_FP8_BWD_QK_MFMA": "bf16"},
                   {"HSTU_ATTN_BWD": "triton"},
                   {"DLRM_HSTU_FP8_BWD_FULL_HYBRID": "dq"},
                   {"DLRM_HSTU_FP8_BWD_FULL_DOBLK": "0"}):
            with env(**kv):
                self.assertFalse(mha.c8_do_blk_planned(**self.args()), kv)
        with env(DLRM_HSTU_FP8_BWD_QK_MFMA="fp8_strict"):
            self.assertTrue(mha.c8_do_blk_planned(**self.args()))

    def test_shape_and_config(self) -> None:
        self.assertFalse(mha.c8_do_blk_planned(**self.args(heads=3)))
        a = self.args()
        a["qkv_scale"] = 0.0
        self.assertFalse(mha.c8_do_blk_planned(**a))
        a = self.args()
        a["max_attn_len"] = 8
        self.assertFalse(mha.c8_do_blk_planned(**a))
        a = self.args()
        a["out"] = a["out"].float()
        self.assertFalse(mha.c8_do_blk_planned(**a))
        self.kernel.SHIPPED = dict(self.kernel.SHIPPED, dq16="f16")
        self.assertFalse(mha.c8_do_blk_planned(**self.args()))
        self.kernel.SHIPPED = dict(self.kernel.SHIPPED, dq16="bf16p", BN=64)
        self.assertFalse(mha.c8_do_blk_planned(**self.args()))

    def test_host_cannot_run(self) -> None:
        mha.unsupported_hardware_reason = lambda: "no GPU"
        mha._host_ok = False
        self.assertFalse(mha.c8_do_blk_planned(**self.args()))

    def test_num_targets_only_from_the_memo(self) -> None:
        nt = torch.tensor([0, 1, 1])
        # Cold memo: no device read in the forward -> no site.
        self.assertFalse(mha.c8_do_blk_planned(**self.args(num_targets=nt)))
        self.assertEqual(mha._targets_max_memo, (None, -1, 0))
        mha.seed_max_num_targets(nt, 1)
        self.assertTrue(mha.c8_do_blk_planned(**self.args(num_targets=nt)))
        mha.seed_max_num_targets(nt, 2)
        self.assertFalse(mha.c8_do_blk_planned(**self.args(num_targets=nt)))

    def test_backward_uses_the_same_predicates(self) -> None:
        import inspect

        src = inspect.getsource(mha.flydsl_hstu_attention_bwd)
        self.assertIn("_blk_ok = _c8_blk_ok(config, heads, dim)", src)
        self.assertIn("if _c8_do_blk_selected(f8b_full, fp8_mfma, input_fp8, q_bf16, config,", src)
        self.assertIn("_do8 = hstu_do8_registry.take(dout, seq_offsets, N)", src)
        self.assertIn('hstu_do8_registry.discard(dout, "not_c8")', src)
        self.assertEqual(src.count("quantize_do_e4m3_blk(dout, seq_offsets, N)"), 1)
        tsrc = inspect.getsource(mha.try_flydsl_attention_bwd)
        self.assertIn('hstu_do8_registry.discard(dout, "flydsl_not_selected")', tsrc)
        self.assertIn('hstu_do8_registry.discard(dout, "shape_fallback")', tsrc)


class WiringTest(unittest.TestCase):
    """Source pins of the tree-specific hunks (read as text: importing these
    modules needs the full training stack)."""

    def read(self, *parts):
        with open(os.path.join(OPS, *parts)) as fh:
            return fh.read()

    def test_site_only_in_the_ordinary_layer_forward(self) -> None:
        src = self.read("triton", "triton_hstu_preprocess_and_attention.py")
        lo = src.index("class _HSTUPreprocessAndAttentionFunction(")
        hi = src.index("class _HSTUPreprocessAndAttentionTargetsOnlyFunction(")
        self.assertEqual(src.count("hstu_do8_registry.new_site("), 1)
        i = src.index("hstu_do8_registry.new_site(")
        self.assertTrue(lo < i < hi)
        self.assertLess(i, src.index("        return silu_u, out, x\n"))
        # The knob is the guard's first, cheapest test (default off); v1.2:
        # training forwards only (x needs a gradient), as B1a's guards.
        self.assertEqual(src.count("hstu_do8_registry.sites_enabled()"), 1)
        self.assertIn("            hstu_do8_registry.sites_enabled()\n"
                      "            and bool(ctx.needs_input_grad[0])\n"
                      "            and _flydsl_do8_planned()(\n", src[lo:hi])
        self.assertNotIn("any(ctx.needs_input_grad)", src)
        self.assertNotIn("do8", src[hi:])
        self.assertIn("hstu_do8_registry.bind(do8_site, dout)", src[lo:hi])
        self.assertIn('hstu_do8_registry.discard(do8_site, "not_taken")', src[lo:hi])

    def test_site_guard_input_0_is_x(self) -> None:
        # needs_input_grad[0] is the activation input x of the ordinary-layer
        # forward (its first parameter after ctx).
        src = self.read("triton", "triton_hstu_preprocess_and_attention.py")
        fwd = [n for c in ast.parse(src).body
               if isinstance(c, ast.ClassDef) and c.name == "_HSTUPreprocessAndAttentionFunction"
               for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
        self.assertEqual(len(fwd), 1)
        self.assertEqual([a.arg for a in fwd[0].args.args[:2]], ["ctx", "x"])

    def test_compute_output_view_carries_the_site(self) -> None:
        src = self.read("hstu_compute.py")
        self.assertIn(
            "attn_output = carry_site(\n"
            "            attn_output, attn_output.view(-1, hidden_dim * num_heads)\n"
            "        )", src)
        # The targets-only path views its output too, and gets no site.
        self.assertEqual(src.count("carry_site("), 1)

    def test_env_bootstrap_declares_and_binds_the_knob(self) -> None:
        with open(os.path.join(GR, "dlrm_v4", "train", "_env_bootstrap.py")) as fh:
            src = fh.read()
        self.assertEqual(src.count("    DLRM_HSTU_FP8_BWD_DO8: Optional[bool] = None,\n"), 1)
        self.assertEqual(src.count('_bind_bool("DLRM_HSTU_FP8_BWD_DO8", DLRM_HSTU_FP8_BWD_DO8)'), 1)

    @unittest.skipIf(gin is None, "gin-config not installed")
    def test_gin_binds_the_knob_and_the_environment_wins(self) -> None:
        # gin.clear_config() and the environment rewrite are process-global,
        # and the gate runs this next to tests that use gin: so the body runs
        # in a child interpreter (gin + the bootstrap only, no torch import).
        child = textwrap.dedent("""
            import os, sys
            import gin
            from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
            assert os.path.abspath(boot.__file__).startswith(sys.argv[1]), boot.__file__
            K = "DLRM_HSTU_FP8_BWD_DO8"
            got = []
            def run(binding, preset=None):
                os.environ.pop(K, None)
                if preset is not None:
                    os.environ[K] = preset
                gin.clear_config()
                if binding is not None:
                    gin.parse_config("apply_env_bootstrap.%s = %s" % (K, binding))
                boot.apply_env_bootstrap()
                got.append(os.environ.get(K))
            run(None)             # no binding: stays absent = off
            run("True")           # _bind_bool writes "1"
            run("False")          # and "0"
            run("True", "0")      # a pre-set env var wins (kill switch)
            print("R15_GIN_CHILD", repr(got))
        """)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("DLRM_", "HSTU_ATTN_", "TRITON_", "GR_PIN_", "PYTORCH_TUNABLEOP"))}
        env["PYTHONPATH"] = os.pathsep.join(p for p in (ROOT, env.get("PYTHONPATH", "")) if p)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run([sys.executable, "-B", "-c", child, ROOT], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-3000:])
        line = [x for x in r.stdout.splitlines() if x.startswith("R15_GIN_CHILD ")]
        self.assertEqual(line, ["R15_GIN_CHILD [None, '1', '0', '0']"], r.stdout[-1500:])

    def test_gate_shim_collects_every_test_class(self) -> None:
        # The Arbor correctness gate runs pytest on dlrm_v4/tests and
        # dlrm_v4/train/tests only; the shim there must re-export every
        # TestCase class of this module, or the gate silently drops it.
        if not os.path.exists(SHIM):
            self.skipTest("no gate shim in this tree")
        with open(SHIM) as fh:
            tree = ast.parse(fh.read())
        exported = set()
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
                exported = set(ast.literal_eval(node.value))
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "ops.tests.hstu_do8_registry_test"):
                imported |= {a.asname or a.name for a in node.names}
        mine = {n for n, v in globals().items()
                if isinstance(v, type) and issubclass(v, unittest.TestCase)
                and v.__module__ == __name__ and not n.startswith("_")}
        self.assertEqual(exported, mine)
        self.assertEqual(imported, mine)

    def test_gin_does_not_arm_without_a_producer(self) -> None:
        # Binding the knob ON is meaningful only together with a producer
        # (a publish() call outside the registry and the tests).
        path = os.path.join(GR, "dlrm_v4", "train", "gin", "yambda_5b.gin")
        if not os.path.exists(path):
            self.skipTest("no yambda_5b.gin")
        with open(path) as fh:
            armed = re.search(r"^apply_env_bootstrap\.DLRM_HSTU_FP8_BWD_DO8\s*=\s*True\b",
                              fh.read(), re.M)
        if armed is None:
            return
        producers = []
        for root, _dirs, files in os.walk(GR):
            for f in files:
                if f.endswith(".py") and not f.endswith("_test.py") and f != "hstu_do8_registry.py":
                    with open(os.path.join(root, f)) as fh:
                        text = fh.read()
                    if "hstu_do8_registry" in text and re.search(r"\bpublish\(", text):
                        producers.append(f)
        self.assertTrue(producers, "yambda_5b.gin binds DLRM_HSTU_FP8_BWD_DO8 = True but no "
                        "module publishes to hstu_do8_registry")


if __name__ == "__main__":
    unittest.main()
