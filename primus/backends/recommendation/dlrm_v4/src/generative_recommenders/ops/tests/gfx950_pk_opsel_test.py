"""gfx950 packed-FP32 op_sel guard (ROCm 10.2 port, opsel-fix).

RuleTest (CPU): gfx950_pk_opsel.invalid_pk_f32_opsel flags exactly the encodings AMD
LLVM 24's assembler rejects for gfx950 ("invalid op_sel operand"): v_pk_add_f32 /
v_pk_mul_f32 with op_sel == [0,1] and v_pk_fma_f32 with op_sel[1] or op_sel[2] set,
for any op_sel_hi / neg modifier and any source kind (VGPR, SGPR, constant).

LayerNormSavedStatsTest (gfx950 GPU, Triton with llvm_fn_attrs, i.e. 3.9): the
saved mean/rstd launch of triton_weighted_layer_norm_fwd (the backward recompute,
COMPUTE_MEAN_AND_RSTD=False) compiles without the invalid form (and without packed
FP32), and its output equals the same launch without the option bit for bit.

FlyDSL bwd_dkdv (vf3p / vf4 constants) is checked by compiling the production
configs (opsel-fix tools/fly_compile.sh + pk_census.py); see hstu_bwd_dkdv.py at
pk_cx2.
"""

import unittest

import torch

from generative_recommenders.ops.triton.gfx950_pk_opsel import (
    NO_PACKED_FP32_FN_ATTRS,
    invalid_pk_f32_opsel,
    no_packed_fp32_launch_opts,
)


class RuleTest(unittest.TestCase):
    INVALID = [
        "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] op_sel:[0,1]",
        "v_pk_add_f32 v[4:5], v[0:1], v[2:3] op_sel:[0,1] neg_lo:[0,1] neg_hi:[0,1]",
        "v_pk_mul_f32 v[4:5], v[0:1], s[6:7] op_sel:[0,1] op_sel_hi:[1,0]",
        "v_pk_fma_f32 v[188:189], v[154:155], s[6:7], v[102:103] op_sel:[0,1,0]",
        "v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7] op_sel:[0,0,1]",
        "v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7] op_sel:[1,1,0] op_sel_hi:[0,0,1]",
        "\tv_pk_add_f32 v[10:11], v[2:3], v[4:5] op_sel:[0,1] ; comment",
    ]
    VALID = [
        "v_pk_mul_f32 v[4:5], v[0:1], v[2:3]",
        "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] op_sel:[1,0]",
        "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] op_sel:[1,1]",
        "v_pk_add_f32 v[4:5], v[0:1], v[2:3] op_sel_hi:[1,0]",
        "v_pk_fma_f32 v[188:189], s[6:7], v[154:155], v[102:103] op_sel:[1,0,0]",
        "v_pk_fma_f32 v[0:1], v[2:3], s[2:3], v[6:7] op_sel_hi:[1,0,1]",
        "v_pk_mov_b32 v[0:1], v[2:3], v[4:5] op_sel:[0,1]",
        "v_pk_mul_f16 v0, v1, v2 op_sel:[0,1]",
        "v_mul_f32_e32 v0, v1, v2",
    ]

    def test_rule(self) -> None:
        for line in self.INVALID:
            self.assertEqual(len(invalid_pk_f32_opsel(line)), 1, line)
        for line in self.VALID:
            self.assertEqual(invalid_pk_f32_opsel(line), [], line)
        self.assertEqual(len(invalid_pk_f32_opsel("\n".join(self.INVALID + self.VALID))),
                         len(self.INVALID))


def _gfx950_with_fn_attrs() -> bool:
    if not torch.cuda.is_available():
        return False
    return bool(no_packed_fp32_launch_opts())


@unittest.skipUnless(_gfx950_with_fn_attrs(), "needs a gfx950 GPU and Triton with llvm_fn_attrs (3.9)")
class LayerNormSavedStatsTest(unittest.TestCase):
    def test_no_invalid_op_sel_and_bitwise(self) -> None:
        import triton

        from generative_recommenders.ops.triton import triton_layer_norm as lnm
        from generative_recommenders.ops.triton.autotune_pin_table import apply_autotune_pins

        apply_autotune_pins(force=True)  # the production config (BLOCK_N=4, num_warps=1)
        jf = lnm._weighted_layer_norm_fwd
        while not hasattr(jf, "device_caches"):
            jf = jf.fn
        dev = torch.device("cuda")
        D = 512
        for N in (1024, 1024 + 3, 4099):
            g = torch.Generator(device=dev).manual_seed(N)
            scale = torch.exp(torch.empty(N, device=dev).uniform_(-2.3, 2.3, generator=g))
            off = torch.empty(N, device=dev).uniform_(-5.0, 5.0, generator=g)
            x = (torch.randn(N, D, device=dev, generator=g) * scale[:, None]
                 + off[:, None]).to(torch.bfloat16)
            w = (torch.randn(D, device=dev, generator=g) * 0.1 + 1).to(torch.bfloat16)
            b = (torch.randn(D, device=dev, generator=g) * 0.1).to(torch.bfloat16)
            mean = x.float().mean(1).contiguous()
            rstd = (1.0 / torch.sqrt(x.float().var(1, unbiased=False) + 1e-5)).contiguous()
            before = set(jf.device_caches[torch.cuda.current_device()][0])
            y, _, _ = lnm.triton_weighted_layer_norm_fwd(x, w, b, 1e-5, mean, rstd)
            ents = jf.device_caches[torch.cuda.current_device()][0]
            new = [k for k in ents if k not in before and "llvm_fn_attrs" in k]
            if not new:  # already compiled by an earlier N of the same specialization
                new = [k for k in ents if "llvm_fn_attrs" in k]
            self.assertTrue(new, "the saved-stats launch did not carry llvm_fn_attrs")
            for k in new:
                asm = ents[k].asm["amdgcn"]
                self.assertEqual(invalid_pk_f32_opsel(asm), [], k)
                self.assertNotRegex(asm, r"\bv_pk_(add|mul|fma)_f32\b")
            # same launch without the option (the pre-fix binary): identical bits
            y0 = torch.empty_like(x)
            jf[(triton.cdiv(N, 4),)](
                x, y0, y0, w, b, mean, rstd, N, D, 1e-5, x.stride(0), y0.stride(0),
                y0.stride(0), 1.0, None, 0, IS_SWISH=False, TRAINING=True, BLOCK_D=512,
                BLOCK_N=4, COMPUTE_MEAN_AND_RSTD=False, OUTPUT_FP8=False, OUTPUT_BOTH=False,
                ADD_RESIDUAL=False, num_warps=1, num_ctas=1, num_stages=3)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(y.view(torch.int16), y0.view(torch.int16)), N)
            y1, _, _ = lnm.triton_weighted_layer_norm_fwd(x, w, b, 1e-5, mean, rstd)
            self.assertTrue(torch.equal(y.view(torch.int16), y1.view(torch.int16)), N)

    def test_option_value(self) -> None:
        self.assertEqual(no_packed_fp32_launch_opts(), {"llvm_fn_attrs": NO_PACKED_FP32_FN_ATTRS})


if __name__ == "__main__":
    unittest.main()
