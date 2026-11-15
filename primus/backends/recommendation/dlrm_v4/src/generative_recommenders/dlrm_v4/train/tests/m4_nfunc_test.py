# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""The M4-OTF N function row (dlrm_v4/train/m4_nfunc.py, x4c19 iter67 delivery, verbatim), CPU only.

Synthetic data only (m4_otf_test.make_data: users with uid holes, a one-event user, histories
longer than the 20000-event scan, a saturated pool). Checked:
  * the file is the delivered one (sha256) and is what the builder imports (NFUNC_MODULE/NAME);
  * compute_window_n_incremental equals the reference rule of m4_otf_test (ref_n) byte for byte:
    windows in order with one carried state, out of order with gaps and a repeat, a fresh state,
    unsorted anchors; int16, the window's length, values in [9, 4095]; the stateless
    compute_window_n equals it too;
  * end to end: the node builder with the REAL m4_nfunc (no injected reference) publishes every
    window's N equal to the reference, and the owner arrays dealt from it equal the withdrawn
    table form.
x4c19's offline byte-equality against the full data set (299/299 windows) is their delivery's
test, not part of this tree.
"""

import hashlib
import os
import tempfile
import unittest

import numpy as np

from generative_recommenders.dlrm_v4.train import m4_nfunc, m4_otf
from generative_recommenders.dlrm_v4.train import m4_partition as m4p
from generative_recommenders.dlrm_v4.train.tests import m4_otf_test as T

# sha256 of the x4c19 iter67 delivery (probes/iter67/m4otf/deliver/m4_nfunc.py); any edit fails here
M4_NFUNC_SHA256 = "dfcba44986c32c1b6ea07bb3824555128bd3d82d92edfad9d72005307f051ab1"
_REAL = "generative_recommenders.dlrm_v4.train.m4_nfunc"


def _arrays(seed: int, n_windows: int = 12, n_anchors: int = 20000):
    with tempfile.TemporaryDirectory(prefix="m4nfunc_") as d:
        data = T.make_data(d, seed=seed, n_windows=n_windows, n_anchors=n_anchors)
    flags = tuple(np.asarray(f).view(np.uint8) for f in data["flags"])
    return data, flags


class NFuncRowTest(unittest.TestCase):
    def test_file_is_the_delivery_and_the_builder_imports_it(self) -> None:
        with open(m4_nfunc.__file__, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), M4_NFUNC_SHA256)
        self.assertEqual(m4_otf.NFUNC_MODULE, _REAL)
        self.assertTrue(callable(getattr(m4_nfunc, m4_otf.NFUNC_NAME)))

    def _check(self, got, p, csr, flags) -> None:
        want = T.ref_n(p, csr, flags)
        self.assertEqual(got.dtype, np.int16)
        self.assertEqual(got.shape, (p.shape[0],))
        if p.shape[0]:
            self.assertGreaterEqual(int(got.min()), 9)
            self.assertLessEqual(int(got.max()), 4095)
        self.assertEqual(got.tobytes(), want.tobytes())

    def test_synthetic_equivalence_with_the_reference_rule(self) -> None:
        fn = m4_nfunc.compute_window_n_incremental
        for seed in range(3):
            data, flags = _arrays(seed)
            csr, pos, tsi, off = data["csr"], data["pos"], data["tsi"], data["off"]
            nu = csr.shape[0] - 1
            win = [pos[tsi[off[t]:off[t + 1]]] for t in range(off.shape[0] - 1)]
            # the data reach the per-pool cap (1362); an edge window adds every user's first event (empty
            # history: N = 9, incl. a one-event user) and last event (history up to the 20000-event scan)
            self.assertGreaterEqual(max(int(T.ref_n(w, csr, flags).max()) for w in win if w.size), 1362 + 9)
            edge = np.unique(np.concatenate([csr[:-1], csr[1:] - 1])).astype(np.int64)
            win.append(edge)
            self.assertEqual(int(T.ref_n(edge, csr, flags).min()), 9)
            state = None
            for p in win:  # increasing windows, one state (the builder's normal order)
                n, state = fn(p, csr, flags, state, user_lo=0, user_hi=nu)
                self._check(n, p, csr, flags)
            state = None
            for t in (0, 1, 5, 3, 3, 9, 2, 11):  # out of order, gaps, a repeat (the builder's jumps)
                n, state = fn(win[t], csr, flags, state, user_lo=0, user_hi=nu)
                self._check(n, win[t], csr, flags)
            rng = np.random.default_rng(seed)
            p = win[7][rng.permutation(win[7].shape[0])]  # unsorted anchors, fresh state
            n, _ = fn(p, csr, flags, None, user_lo=0, user_hi=nu)
            self._check(n, p, csr, flags)
            self._check(m4_nfunc.compute_window_n(win[4], csr, flags), win[4], csr, flags)
            n, _ = fn(np.zeros(0, np.int64), csr, flags, None, user_lo=0, user_hi=nu)
            self._check(n, np.zeros(0, np.int64), csr, flags)

    def test_builder_end_to_end_with_the_real_n_function(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4nfunc_") as d, T._Env(nfunc=_REAL):
            data = T.make_data(d, seed=21)
            nw = len(data["off"]) - 1
            rt = T._start(0, data, d, T._name(), lws=1)
            try:
                rt.progress(1, 10 ** 6)
                want = list(range(2, nw))
                self.assertTrue(T._wait(lambda: (rt.drain(1), all(rt.holds(j) for j in want))[1]))
                lut = m4p.cost_lut()
                W, bs = 4, 16
                wq = m4p.weights_from_factors(np.linspace(0.95, 1.05, W), W)
                for j in want:
                    fd = rt.dup_fd(j, j)
                    try:
                        lo, hi = int(data["off"][j]), int(data["off"][j + 1])
                        win = m4p.window_array(fd)
                        self.assertEqual(win.tobytes(), data["table"][lo:hi].tobytes(), j)
                        nb = (hi - lo) // W // bs
                        if nb:
                            for r in range(W):
                                a = m4p.compute_m4(win, lut, j, j, hi - lo, nb, W, bs, wq, r)
                                b = T._ref_table_m4(data["table"], data["off"], lut, j, j, hi - lo, nb, W, bs, wq, r)
                                self.assertTrue(np.array_equal(a[0], b[0]))
                                self.assertEqual(a[2], b[2])
                        del win
                    finally:
                        os.close(fd)
            finally:
                rt.close()
            self.assertEqual(rt.builder._p.returncode, 0)  # no "m4otf builder failed": exited on EOF


if __name__ == "__main__":
    unittest.main()
