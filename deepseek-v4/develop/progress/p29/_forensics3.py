"""P29 forensics, third pass — inspect input shape + dtype of the
matching aten::sum events to understand why each call is 10 ms.
"""

from __future__ import annotations

import collections
import glob
import json
import sys

TARGET = (
    "reduce_kernel<512, 1, at::native::ReduceOp<float, "
    "at::native::func_wrapper_t<float, at::native::sum_functor<"
    "float, float, float>"
)


def main(argv: list[str]) -> int:
    fp = (
        argv[1]
        if len(argv) > 1
        else glob.glob(
            "output/amd/tas-mi355x-20260509/p28_profile_baseline_pp1_ep8_seq4096/tensorboard/*.pt.trace.json"
        )[0]
    )
    print(f"loading {fp} ...")
    data = json.load(open(fp))
    evs = data["traceEvents"]

    def _ext(e: dict) -> int | None:
        a = e.get("args") or {}
        for k in ("External id", "external id", "correlation"):
            v = a.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    kernels = [
        e
        for e in evs
        if e.get("ph") == "X"
        and e.get("cat") in ("kernel", "Kernel", "gpu_op")
        and TARGET in e.get("name", "")
    ]
    print(f"  {len(kernels)} matching kernels")

    ext_to_cpu: dict[int, list[dict]] = collections.defaultdict(list)
    for e in evs:
        if e.get("ph") == "X" and e.get("cat") == "cpu_op":
            eid = _ext(e)
            if eid is not None:
                ext_to_cpu[eid].append(e)

    # Aggregate aten::sum cpu_op input shapes + dtypes.
    shape_hist = collections.Counter()
    shape_dur_us = collections.defaultdict(float)
    sample_event = None
    for k in kernels:
        eid = _ext(k)
        if eid is None:
            continue
        for c in ext_to_cpu.get(eid, []):
            if c.get("name") != "aten::sum":
                continue
            args = c.get("args") or {}
            input_shapes = args.get("Input Dims") or args.get("input_dims") or args.get("Input dims") or []
            input_types = args.get("Input type") or args.get("input_type") or args.get("Input Types") or []
            try:
                shapes_t = tuple(tuple(s) if isinstance(s, list) else s for s in input_shapes)
            except TypeError:
                shapes_t = tuple()
            try:
                types_t = tuple(input_types) if isinstance(input_types, (list, tuple)) else ()
            except TypeError:
                types_t = ()
            key = (shapes_t, types_t)
            shape_hist[key] += 1
            shape_dur_us[key] += float(k.get("dur", 0))
            if sample_event is None:
                sample_event = (c, k)

    print(f"\nDistinct (input shapes, dtypes) of aten::sum -> reduce_kernel:")
    for (shapes, types), n in shape_hist.most_common(20):
        avg_us = shape_dur_us[(shapes, types)] / max(1, n)
        total_ms = shape_dur_us[(shapes, types)] / 1e3
        print(
            f"  count={n:4d}  total={total_ms:7.1f} ms  avg={avg_us / 1e3:.2f} ms  shapes={shapes}  types={types}"
        )

    if sample_event:
        c, k = sample_event
        print("\nSample aten::sum cpu_op:")
        print(
            json.dumps(
                {k_: v for k_, v in c.items() if k_ != "args"} | {"args": c.get("args", {})}, indent=2
            )[:1500]
        )
        print("\nSample matching kernel:")
        print(
            json.dumps(
                {k_: v for k_, v in k.items() if k_ != "args"} | {"args": k.get("args", {})}, indent=2
            )[:1500]
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
