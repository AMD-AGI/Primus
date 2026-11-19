"""Dump (mode=dump) or compare (mode=check) tensorwise FP8 quant outputs byte-for-byte."""

import sys
import zlib

import torch

from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quantize_fp8_tensorwise_pad_impl,
)

PATH = "/tmp/quant_bitexact_ref.pt"
CASES = []
for dt in (torch.bfloat16, torch.float16, torch.float32):
    for shape in [(262144, 768), (4096, 2048), (1000, 1536), (37, 100), (3, 5), (17,), (8, 64, 96), (513, 1000)]:
        for qdt in (torch.float8_e4m3fn, torch.float8_e5m2):
            for k_align, pad_n in ((1, False), (128, False), (128, True)):
                CASES.append((dt, shape, qdt, k_align, pad_n, 0))
# misaligned input: a view starting one element in
for dt in (torch.bfloat16, torch.float32):
    CASES.append((dt, (4096, 768), torch.float8_e4m3fn, 1, False, 1))


def run(case):
    dt, shape, qdt, k_align, pad_n, offset = case
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(repr((dt, shape, qdt)).encode()))
    n = 1
    for s in shape:
        n *= s
    base = torch.randn(n + offset, device="cuda", generator=g, dtype=torch.float32) * 4
    x = base.to(dt)[offset:].view(shape)  # offset > 0 leaves data_ptr off 16-byte alignment
    y, s = quantize_fp8_tensorwise_pad_impl(x, qdt, pad_n=pad_n, k_align=k_align)
    return y.view(torch.uint8).cpu(), s.cpu()


mode = sys.argv[1]
if mode == "dump":
    torch.save([run(c) for c in CASES], PATH)
    print(f"dumped {len(CASES)} cases")
else:
    ref = torch.load(PATH)
    bad = 0
    for c, (ry, rs) in zip(CASES, ref):
        y, s = run(c)
        if not (torch.equal(y, ry) and torch.equal(s, rs)):
            bad += 1
            print("MISMATCH", c)
    print(f"checked {len(CASES)} cases, {bad} mismatches")
