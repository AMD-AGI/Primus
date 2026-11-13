#!/usr/bin/env python3

# pyre-unsafe

"""FusedDenseAdam (yambda_5b.gin item (40)) must equal torch.optim.Adam (foreach
default) + torch.nn.utils.clip_grad_norm_ bit for bit: params, exp_avg,
exp_avg_sq, clipped grads, state['step'] and the state_dict layout."""

import math
import unittest

import torch
from generative_recommenders.common import gpu_unavailable

# a subset of the DLRMv4 dense parameter shapes (incl. 1-element and odd sizes)
_SHAPES = [
    [512, 512], [512], [1, 512], [1], [512, 2048], [2048], [1536, 512], [8, 512, 512],
    [8, 512], [256, 512], [3, 8], [1, 24], [256, 24], [512, 516], [2049, 512], [512, 1536],
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().view(torch.int32)


def _grads(step: int, edge: bool, scale: float):
    g = torch.Generator(device="cuda").manual_seed(1000 + step)
    out = []
    for i, s in enumerate(_SHAPES):
        x = torch.randn(s, device="cuda", generator=g) * [1e-3, 1e-2, 1e-1][(i + step) % 3] * scale
        if edge:
            r = torch.rand(s, device="cuda", generator=g)
            den = torch.randint(1, 0x7FFFFF, s, device="cuda", generator=g, dtype=torch.int32).view(torch.float32)
            x = torch.where(r < 0.05, torch.zeros_like(x), x)
            x = torch.where((r >= 0.05) & (r < 0.08), den, x)
            x = torch.where((r >= 0.08) & (r < 0.10), x * 1e-19, x)
            if (i + step) % 7 == 0:
                x.zero_()
        out.append(x)
    return out


class FusedDenseAdamTest(unittest.TestCase):
    def _run(self, clip: bool, edge: bool, steps: int, scale, bucket: bool = False) -> None:
        from generative_recommenders.ops.triton import dense_adam_fused as daf

        torch.manual_seed(0)
        ref = [torch.nn.Parameter(torch.randn(s, device="cuda") * 0.02) for s in _SHAPES]
        fus = [torch.nn.Parameter(p.detach().clone()) for p in ref]
        kw = dict(lr=1e-3, betas=(0.95, 0.999), eps=1e-8, weight_decay=0)
        ro = torch.optim.Adam(ref, **kw)
        fo = daf.FusedDenseAdam(fus, **kw)
        flat_r = torch.empty(sum(math.prod(s) for s in _SHAPES) + 1, device="cuda")
        flat_f = torch.empty_like(flat_r)
        for step in range(1, steps + 1):
            lr = 1e-3 * min(1.0, step / 10.0) * (1.0 + 0.1 * math.sin(step))
            ro.param_groups[0]["lr"] = lr
            fo.param_groups[0]["lr"] = lr
            o = 1  # odd offset: grads are 4-byte-aligned bucket views when bucket=True
            for pr, pf, g in zip(ref, fus, _grads(step, edge, scale(step))):
                if bucket:
                    n = g.numel()
                    vr, vf = flat_r[o : o + n].view(g.shape), flat_f[o : o + n].view(g.shape)
                    vr.copy_(g)
                    vf.copy_(g)
                    pr.grad, pf.grad = vr, vf
                    o += n
                else:
                    pr.grad, pf.grad = g.clone(), g.clone()
            if clip:
                nr = torch.nn.utils.clip_grad_norm_(ref, max_norm=1.0)
                nf = daf.clip_grad_norm_fold_(fus, 1.0, fo)
                self.assertTrue(torch.equal(_bits(nr), _bits(nf)))
            ro.step()
            fo.step()
            for pr, pf in zip(ref, fus):
                self.assertTrue(torch.equal(_bits(pr), _bits(pf)), f"param step {step}")
                self.assertTrue(torch.equal(_bits(pr.grad), _bits(pf.grad)), f"grad step {step}")
                for k in ("exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(_bits(ro.state[pr][k]), _bits(fo.state[pf][k])), f"{k} step {step}")
                self.assertEqual(ro.state[pr]["step"].item(), fo.state[pf]["step"].item())
                self.assertEqual(fo.state[pf]["step"].device.type, "cpu")
                self.assertEqual(fo.state[pf]["step"].dtype, torch.float32)
            ro.zero_grad()
            fo.zero_grad()
        a, b = ro.state_dict(), fo.state_dict()
        self.assertEqual(a["param_groups"], b["param_groups"])
        self.assertEqual(sorted(a["state"].keys()), sorted(b["state"].keys()))
        for k in a["state"]:
            self.assertEqual(sorted(a["state"][k].keys()), sorted(b["state"][k].keys()))

    @unittest.skipIf(*gpu_unavailable)
    def test_no_clip(self) -> None:
        self._run(clip=False, edge=False, steps=50, scale=lambda s: 1.0)

    @unittest.skipIf(*gpu_unavailable)
    def test_clip_active_and_inactive_edge(self) -> None:
        self._run(clip=True, edge=True, steps=50, scale=lambda s: 1.0 if s % 2 else 1e-5)

    @unittest.skipIf(*gpu_unavailable)
    def test_bucket_view_grads(self) -> None:
        self._run(clip=True, edge=True, steps=20, scale=lambda s: 1.0, bucket=True)

    @unittest.skipIf(*gpu_unavailable)
    def test_fallback_weight_decay(self) -> None:
        from generative_recommenders.ops.triton import dense_adam_fused as daf

        ref = [torch.nn.Parameter(torch.randn(s, device="cuda")) for s in _SHAPES[:4]]
        fus = [torch.nn.Parameter(p.detach().clone()) for p in ref]
        ro = torch.optim.Adam(ref, lr=1e-3, weight_decay=0.1)
        fo = daf.FusedDenseAdam(fus, lr=1e-3, weight_decay=0.1)
        for step in range(5):
            for pr, pf, g in zip(ref, fus, _grads(step, False, 1.0)[:4]):
                pr.grad, pf.grad = g.clone(), g.clone()
            torch.nn.utils.clip_grad_norm_(ref, max_norm=1.0)
            daf.clip_grad_norm_fold_(fus, 1.0, fo)
            ro.step()
            fo.step()
        for pr, pf in zip(ref, fus):
            self.assertTrue(torch.equal(_bits(pr), _bits(pf)))

    @unittest.skipIf(*gpu_unavailable)
    def test_version_counters_bumped(self) -> None:
        # Caches keyed on weight._version (FP8 weight shadows) go stale if the raw-pointer kernel skips the bump.
        from generative_recommenders.ops.triton import dense_adam_fused as daf

        fus = [torch.nn.Parameter(torch.randn(s, device="cuda") * 0.02) for s in _SHAPES]
        fo = daf.FusedDenseAdam(fus, lr=1e-3, betas=(0.95, 0.999), eps=1e-8, weight_decay=0)
        for step in range(1, 4):
            clip = step % 2 == 0
            for pf, g in zip(fus, _grads(step, False, 1.0)):
                pf.grad = g.clone()
            if clip:
                daf.clip_grad_norm_fold_(fus, 1e-3, fo)
            before = [(pf._version, pf.grad._version) for pf in fus]
            state = [(fo.state[pf]["exp_avg"]._version, fo.state[pf]["exp_avg_sq"]._version) for pf in fus if pf in fo.state]
            fo.step()
            for i, pf in enumerate(fus):
                self.assertGreater(pf._version, before[i][0], f"param {i} step {step}")
                if clip:
                    self.assertGreater(pf.grad._version, before[i][1], f"grad {i} step {step}")
            for (va, vs), pf in zip(state, fus):
                self.assertGreater(fo.state[pf]["exp_avg"]._version, va)
                self.assertGreater(fo.state[pf]["exp_avg_sq"]._version, vs)
            fo.zero_grad()


if __name__ == "__main__":
    unittest.main()
