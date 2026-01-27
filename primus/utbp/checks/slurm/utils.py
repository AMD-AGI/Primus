from __future__ import annotations
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CmdResult:
    rc: int
    out: str
    err: str

def run_cmd(cmd: list[str], timeout_s: int = 15) -> CmdResult:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CmdResult(p.returncode, p.stdout, p.stderr)
    except Exception as e:
        return CmdResult(999, "", f"{type(e).__name__}: {e}")

def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def expand_nodelist(nodelist: str) -> List[str]:
    if not nodelist:
        return []
    if "[" not in nodelist:
        return [nodelist]
    m = re.match(r"^(?P<prefix>[^\[]+)\[(?P<body>[^\]]+)\]$", nodelist)
    if not m:
        return [nodelist]
    prefix = m.group("prefix")
    body = m.group("body")
    parts = body.split(",")
    nodes: List[str] = []
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            pad = len(a)
            start = int(a)
            end = int(b)
            for i in range(start, end + 1):
                nodes.append(f"{prefix}{str(i).zfill(pad)}")
        else:
            nodes.append(f"{prefix}{part}")
    return nodes

def try_scontrol_hostnames(nodelist: str) -> Tuple[bool, List[str], str]:
    r = run_cmd(["scontrol", "show", "hostnames", nodelist])
    if r.rc != 0:
        return False, [], (r.err or r.out).strip()
    names = [ln.strip() for ln in r.out.splitlines() if ln.strip()]
    return True, names, ""
