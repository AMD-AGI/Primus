"""mxfp6_adaln_tunableop: the pinned TunableOp results load, change only their own GEMMs (bitwise), and nothing
is written back on exit. Runs in a subprocess: TunableOp's state is process-global."""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_SCRIPT = textwrap.dedent(
    """
    import types, torch, torch.cuda.tunable as tn
    from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates
    g = torch.Generator(device="cuda").manual_seed(0)
    dy = torch.randn(32, 18432, device="cuda", dtype=torch.bfloat16, generator=g)
    x = torch.randn(32, 3072, device="cuda", dtype=torch.bfloat16, generator=g)
    a = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16, generator=g)
    w9 = torch.randn(9216, 3072, device="cuda", dtype=torch.bfloat16, generator=g)
    b9 = torch.randn(9216, device="cuda", dtype=torch.bfloat16, generator=g)
    ref_w, ref_o, ref_f = torch.mm(dy.t(), x), a @ x.t(), torch.addmm(b9, x, w9.t())
    mxfp6_gates.configure(types.SimpleNamespace(mxfp6_adaln_tunableop=True))
    assert tn.is_enabled() and not tn.tuning_is_enabled()
    loaded = tn.get_results()
    if len(loaded) != 2:  # a different hipBLASLt / PyTorch: the gate must have switched TunableOp back off
        assert not tn.is_enabled(), loaded
        print("SKIPPED-VALIDATOR")
    else:
        out = torch.empty_like(ref_w)
        torch.mm(dy.t(), x, out=out)
        assert torch.equal(out.view(torch.int16), ref_w.view(torch.int16))
        assert torch.equal((a @ x.t()).view(torch.int16), ref_o.view(torch.int16))  # no entry: default GEMM
        assert torch.equal(torch.addmm(b9, x, w9.t()).view(torch.int16), ref_f.view(torch.int16))  # forward: no entry
        print("OK")
    """
)


def test_adaln_tunableop_loads_and_is_bitwise(tmp_path):
    env = dict(os.environ)
    for k in [k for k in env if k.startswith("PYTORCH_TUNABLEOP")]:
        del env[k]
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    if "SKIPPED-VALIDATOR" in r.stdout:
        pytest.skip("pinned results do not validate on this image")
    assert "OK" in r.stdout
    assert not list(tmp_path.iterdir()), "TunableOp wrote a results file on exit"
