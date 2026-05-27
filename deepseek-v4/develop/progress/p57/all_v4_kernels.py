#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

trace = json.loads(Path(sys.argv[1]).read_text())
by_name: dict[str, list[float]] = defaultdict(list)
for ev in trace.get("traceEvents", []):
    if ev.get("ph") != "X":
        continue
    cat = (ev.get("cat") or "").lower()
    if "kernel" not in cat and cat != "gpu_op":
        continue
    name = ev.get("name", "")
    if "v4_" in name.lower() or "_csa" in name.lower():
        by_name[name].append(float(ev["dur"]) * 1e-3)

rows = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))
for name, durs in rows:
    n = len(durs)
    tot = sum(durs)
    print(f"{name[:90]:90s}  {n:>3d}  total={tot:>8.3f} ms  mean={tot/n:>7.4f} ms")
