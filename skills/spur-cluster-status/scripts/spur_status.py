#!/usr/bin/env python3
"""Collect Spur cluster node-allocation status and print a Markdown report.

Targets the AMD "Spur" scheduler (SLURM-compatible CLI: sinfo/squeue/scontrol/
spur accounts). Read-only. Handles Spur quirks: `-o` delimiters are rendered as
spaces, positional filters are ignored, there is no `scontrol show hostnames`,
and QoS membership is not enforced per account (only the account is) even though
per-QoS node caps *are* enforced.

Usage:
    python3 spur_status.py [--user USER]

The report is printed to stdout (redirect it to save a file).
"""

import argparse
import getpass
import re
import subprocess
from collections import defaultdict
from datetime import datetime

# Node-state buckets used for the state breakdown / utilization math.
BUSY_STATES = {"alloc", "mix"}
FREE_STATES = {"idle"}
DOWN_STATES = {"down", "drain", "drained", "fail", "failing", "unknown"}

# Spur renders a TRES node entry as `node=LIMIT` or `node=LIMIT(IN_USE)`,
# where a LIMIT of `N` means unlimited.
NODE_TRES_RE = re.compile(r"node=([0-9]+|N)(?:\((\d+)\))?")


def run(cmd):
    """Run a command list; return stdout text, or '' on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        return out.stdout or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Hostlist expansion (no `scontrol show hostnames` on Spur, so do it locally).
# Handles: "p-030", "p-[036,089,110]", "p-[036-040]", and top-level commas.
# ---------------------------------------------------------------------------
def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
            cur += ch
        elif ch == "]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return parts


def _expand_one(tok):
    m = re.match(r"^(.*?)\[([^\]]*)\](.*)$", tok)
    if not m:
        return [tok] if tok else []
    pre, body, post = m.groups()
    out = []
    for item in body.split(","):
        item = item.strip()
        if "-" in item:
            a, b = item.split("-", 1)
            width = len(a)
            for i in range(int(a), int(b) + 1):
                out.append(f"{pre}{str(i).zfill(width)}{post}")
        elif item:
            out.append(f"{pre}{item}{post}")
    return out


def expand_hostlist(s):
    if not s or s in ("(null)", "N/A", "-"):
        return []
    res = []
    for tok in _split_top(s):
        res.extend(_expand_one(tok))
    return res


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_user_accounts(user):
    """Return (accounts, default_acct, def_qos) for the given user."""
    text = run(["spur", "accounts", "show", "user"])
    accounts, default_acct, def_qos = [], "", ""
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 2 or toks[0] in ("User", "----"):
            continue
        if toks[0] != user:
            continue
        accounts.append(toks[1])
        if len(toks) >= 4:
            default_acct = toks[3]
        if len(toks) >= 5:
            def_qos = toks[4]
    return sorted(set(accounts)), default_acct, def_qos


def parse_node_tres(tres):
    """Extract (limit, in_use) from the `node=` entry of a TRES string.

    Returns (None, None) when there is no node entry, and a limit of None when
    the limit is `N` (unlimited). in_use is None unless the source rendered it.
    """
    if not tres:
        return None, None
    m = NODE_TRES_RE.search(tres)
    if not m:
        return None, None
    limit = None if m.group(1) == "N" else int(m.group(1))
    in_use = int(m.group(2)) if m.group(2) else None
    return limit, in_use


def parse_qos():
    """Return list of dicts: {name, prio, preempt, max_jobs_pu, max_submit_pu, node_limit}.

    Uses `-P` (pipe-delimited) rather than the default space-aligned table: most
    QoS rows leave the limit columns blank, and blanks collapse in the aligned
    output, which shifts every later column and makes GrpTRES unparseable.
    """
    text = run(["spur", "accounts", "show", "qos", "-P"])
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = [x.strip() for x in line.split("|")]
        if f[0] in ("Name", "----"):
            continue

        def col(i):
            return f[i] if len(f) > i else ""

        # Name|Priority|PreemptMode|UsageFactor|MaxJobsPU|MaxSubmitPU|MaxWall|
        # GrpWall|Flags|MaxTRES|MaxTRESPU|GrpTRES
        rows.append(
            {
                "name": col(0),
                "prio": col(1),
                "preempt": col(2),
                "max_jobs_pu": col(4),
                "max_submit_pu": col(5),
                "node_limit": parse_node_tres(col(11))[0],
            }
        )
    return rows


def parse_qos_node_usage():
    """Return dict qos -> {limit, in_use} from `scontrol show assoc_mgr`.

    This is the scheduler's own accounting, so `in_use` is a DISTINCT node count:
    two jobs of the same QoS sharing one node count once here, while summing each
    job's NumNodes counts that node twice.

    Only the leading `QOS Records` block is read. The `Association Records` block
    that follows repeats `GrpTRES=` lines per account and per user, and would
    otherwise be attributed to whichever QoS was seen last.
    """
    text = run(["scontrol", "show", "assoc_mgr"])
    usage, cur = {}, None
    for line in text.splitlines():
        if line.startswith("Association Records"):
            break
        if line.startswith("QOS="):
            cur = line.split()[0][len("QOS=") :]
            continue
        if cur and "GrpTRES=" in line:
            tail = line.split("GrpTRES=", 1)[1].strip()
            limit, in_use = parse_node_tres(tail.split()[0] if tail else "")
            usage[cur] = {"limit": limit, "in_use": in_use or 0}
            cur = None
    return usage


def parse_pending_reasons():
    """Return dict jobid -> wait reason (e.g. `QOSGrpNodeLimit`) for pending jobs.

    Queried separately because `%R` carries the nodelist for running jobs and the
    wait reason for pending ones, so it cannot be mixed into the main format
    string without making the trailing columns ambiguous.
    """
    text = run(["squeue", "-h", "-t", "PENDING", "-o", "%i %R"])
    reasons = {}
    for line in text.splitlines():
        toks = line.split(None, 1)
        if len(toks) == 2:
            reasons[toks[0]] = toks[1].strip().strip("()")
    return reasons


def parse_nodes():
    """Return dict node -> {partition, state, gpus} from `sinfo -N`."""
    text = run(["sinfo", "-N", "-h", "-o", "%N %P %t %G"])
    nodes = {}
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 3:
            continue
        name, part, state = toks[0], toks[1], toks[2].rstrip("*")
        gres = toks[3] if len(toks) > 3 else ""
        gpus = gres.count("gpu:")
        nodes[name] = {"partition": part, "state": state, "gpus": gpus}
    return nodes


def parse_jobs(nodes):
    """Return list of job dicts from squeue.

    Spur renders `-o` fields space-separated and DROPS empty fields (e.g. a job
    with no QoS), which shifts columns and breaks naive positional parsing. So we
    only trust the always-present leading fields (jobid, partition, account, user,
    state, nnodes) and disambiguate the trailing optional tokens (qos and/or
    nodelist) against the known node set: the token that expands to a real node is
    the nodelist; the other is the qos. `account` is always present because the
    scheduler enforces it at submit time.
    """
    text = run(["squeue", "-h", "-o", "%i %P %a %u %T %D %q %N"])
    jobs = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 6:
            continue
        jobid, partition, account, user, state = toks[:5]
        nnodes = int(toks[5]) if toks[5].isdigit() else 0
        qos, nodelist = "", ""
        for t in toks[6:]:
            expanded = expand_hostlist(t)
            if expanded and expanded[0] in nodes:
                nodelist = t
            else:
                qos = t
        jobs.append(
            {
                "jobid": jobid,
                "partition": partition,
                "qos": qos,
                "account": account,
                "user": user,
                "state": state,
                "nnodes": nnodes,
                "nodelist": nodelist,
            }
        )
    return jobs


def parse_controller():
    """Extract the controller address from `scontrol show config`."""
    for line in run(["scontrol", "show", "config"]).splitlines():
        if "Addr=" in line:
            return line.split("Addr=", 1)[1].strip()
    return "n/a"


def parse_reservations():
    """Return list of reservation dicts from scontrol show reservation."""
    text = run(["scontrol", "show", "reservation"])
    resvs, cur = [], {}
    for line in text.splitlines():
        if line.startswith("ReservationName="):
            if cur:
                resvs.append(cur)
            cur = {}
        for m in re.finditer(r"(\w+)=([^\s]+)", line):
            cur[m.group(1)] = m.group(2)
    if cur:
        resvs.append(cur)
    return resvs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def group_usage(jobs, nodes, key, state):
    """Aggregate running/pending jobs by a key (qos/account).

    Returns dict key -> {jobs, node_slots, distinct_nodes(set), gpus}.
    distinct_nodes/gpus are only meaningful for RUNNING jobs.
    """
    agg = defaultdict(lambda: {"jobs": 0, "node_slots": 0, "nodes": set(), "gpus": 0})
    for j in jobs:
        if j["state"] != state:
            continue
        k = j[key] or "(none)"
        agg[k]["jobs"] += 1
        agg[k]["node_slots"] += j["nnodes"]
        for n in expand_hostlist(j["nodelist"]):
            if n in nodes:
                agg[k]["nodes"].add(n)
    for k, v in agg.items():
        v["gpus"] = sum(nodes[n]["gpus"] for n in v["nodes"])
    return agg


def h(title):
    return f"\n## {title}\n"


def group_by_user(jobs, nodes):
    """Aggregate every job per user across all states."""
    agg = defaultdict(
        lambda: {
            "total": 0,
            "running": 0,
            "pending": 0,
            "nodes": set(),
            "accounts": set(),
            "qos": set(),
            "partitions": set(),
        }
    )
    for j in jobs:
        a = agg[j["user"]]
        a["total"] += 1
        if j["state"] == "RUNNING":
            a["running"] += 1
        elif j["state"] == "PENDING":
            a["pending"] += 1
        a["accounts"].add(j["account"])
        a["qos"].add(j["qos"])
        a["partitions"].add(j["partition"])
        for n in expand_hostlist(j["nodelist"]):
            if n in nodes:
                a["nodes"].add(n)
    return agg


def build_report(user):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    accounts, default_acct, def_qos = parse_user_accounts(user)
    qos_rows = parse_qos()
    nodes = parse_nodes()
    jobs = parse_jobs(nodes)
    resvs = parse_reservations()
    qos_usage = parse_qos_node_usage()
    pending_reasons = parse_pending_reasons()

    L = []
    L.append("# Spur Cluster Node-Allocation Report")
    L.append("")
    L.append(f"- **Generated**: {now}")
    L.append(f"- **Cluster**: spur  (controller: {parse_controller()})")
    L.append(f"- **Report user**: {user}")

    # 1. Account / QoS
    L.append(h("1. Account / QoS (you)"))
    L.append(f"- **Accounts**: {', '.join(accounts) if accounts else '(none found)'}")
    L.append(f"- **Default account**: {default_acct or '-'}")
    L.append(f"- **Default QoS**: {def_qos or '(unset)'}")
    L.append("")
    L.append(
        "> Permission model (verified): the **account is enforced** - you can only "
        "submit under the accounts listed above (submitting under another account "
        "fails with `user ... is not associated with account`). **QoS is NOT restricted "
        "per account**; any QoS from the global list below is accepted, so QoS selects "
        "scheduling priority, not permission."
    )
    L.append("")
    L.append("Global QoS (higher Prio = higher priority):")
    L.append("")
    L.append("| QoS | Prio | Preempt | Node limit | MaxJobsPU | MaxSubmitPU |")
    L.append("|-----|------|---------|------|------|------|")
    for q in sorted(qos_rows, key=lambda x: -int(x["prio"]) if x["prio"].isdigit() else 0):
        nl = q["node_limit"] if q["node_limit"] is not None else "unlimited"
        L.append(
            f"| {q['name']} | {q['prio']} | {q['preempt']} | {nl} | "
            f"{q['max_jobs_pu'] or '-'} | {q['max_submit_pu'] or '-'} |"
        )

    # 2. Partitions & node states
    L.append(h("2. Partitions & Node States"))
    part_state = defaultdict(lambda: defaultdict(int))
    part_total = defaultdict(int)
    state_total = defaultdict(int)
    gpu_total = gpu_free = 0
    for n, info in nodes.items():
        part_state[info["partition"]][info["state"]] += 1
        part_total[info["partition"]] += 1
        state_total[info["state"]] += 1
        gpu_total += info["gpus"]
        if info["state"] in FREE_STATES:
            gpu_free += info["gpus"]
    all_states = sorted(state_total)
    L.append("| Partition | Total | " + " | ".join(all_states) + " |")
    L.append("|" + "---|" * (len(all_states) + 2))
    for p in sorted(part_total):
        cells = " | ".join(str(part_state[p].get(s, 0)) for s in all_states)
        L.append(f"| {p} | {part_total[p]} | {cells} |")
    total_nodes = sum(part_total.values())
    busy = sum(state_total.get(s, 0) for s in BUSY_STATES)
    L.append("")
    L.append(f"- **Total nodes**: {total_nodes}")
    L.append("- **State breakdown**: " + ", ".join(f"{s}={state_total[s]}" for s in all_states))
    if total_nodes:
        L.append(f"- **Utilization (alloc+mix)**: {busy}/{total_nodes} = {busy*100//total_nodes}%")
    L.append(f"- **GPU capacity (whole-node)**: {gpu_total} total, ~{gpu_free} free (on idle nodes)")

    # idle / unhealthy nodelist
    idle_nodes = sorted(n for n, i in nodes.items() if i["state"] in FREE_STATES)
    bad_nodes = sorted(n for n, i in nodes.items() if i["state"] in DOWN_STATES)
    L.append("")
    L.append(f"- **Idle nodes ({len(idle_nodes)})**: `{','.join(idle_nodes) if idle_nodes else 'none'}`")
    if bad_nodes:
        L.append(f"- **Drain/down nodes ({len(bad_nodes)})**: `{','.join(bad_nodes)}`")

    # 3. QoS node limits vs usage
    L.append(h("3. QoS Node Limits vs Usage"))
    L.append(
        "> `Limit` is the QoS group cap (`GrpTRES=node=`); `Run nodes` is the scheduler's "
        "own **distinct** node count, so it is lower than the sum of each job's node count "
        "whenever two jobs of the same QoS share a node. `Blocked` counts pending jobs held "
        "by `QOSGrpNodeLimit` - demand waiting on the cap rather than on free hardware. "
        "Sorted by QoS name."
    )
    L.append("")
    qos_run_by_name = group_usage(jobs, nodes, "qos", "RUNNING")
    qos_pend_by_name = group_usage(jobs, nodes, "qos", "PENDING")
    blocked_jobs, blocked_nodes = defaultdict(int), defaultdict(int)
    for j in jobs:
        if j["state"] == "PENDING" and pending_reasons.get(j["jobid"]) == "QOSGrpNodeLimit":
            k = j["qos"] or "(none)"
            blocked_jobs[k] += 1
            blocked_nodes[k] += j["nnodes"]

    limit_by_name = {q["name"]: q["node_limit"] for q in qos_rows}
    for name, u in qos_usage.items():
        limit_by_name.setdefault(name, u["limit"])
    all_qos = sorted(set(limit_by_name) | set(qos_run_by_name) | set(qos_pend_by_name))

    L.append(
        "| QoS | Run jobs | Run nodes | Limit | Used % | Pend jobs | Pend nodes | "
        "Blocked jobs | Blocked nodes | MaxSubmitPU |"
    )
    L.append("|-----|------|------|------|------|------|------|------|------|------|")
    limit_sum = 0
    for k in all_qos:
        r = qos_run_by_name.get(k, {"jobs": 0, "nodes": set()})
        p = qos_pend_by_name.get(k, {"jobs": 0, "node_slots": 0})
        limit = limit_by_name.get(k)
        # Prefer the scheduler's distinct-node count; fall back to our own set.
        used = qos_usage.get(k, {}).get("in_use", len(r["nodes"]))
        # A limit of None means unlimited; a limit of 0 is a real cap that pins
        # the QoS to zero nodes, so the two must not be collapsed.
        if limit is None:
            limit_disp, pct = "unlimited", "-"
        else:
            limit_sum += limit
            limit_disp = str(limit)
            pct = f"{used * 100 // limit}%" if limit else "n/a"
        msp = next((q["max_submit_pu"] for q in qos_rows if q["name"] == k), "") or "-"
        L.append(
            f"| {k} | {r['jobs']} | {used} | {limit_disp} | {pct} | {p['jobs']} | "
            f"{p['node_slots']} | {blocked_jobs[k]} | {blocked_nodes[k]} | {msp} |"
        )

    usable = sum(1 for i in nodes.values() if i["state"] not in DOWN_STATES)
    L.append("")
    L.append(f"- **Sum of QoS node limits**: {limit_sum} vs {usable} usable nodes")
    if usable:
        L.append(
            f"- **Oversubscription**: {limit_sum / usable:.1f}x - limits are deliberately "
            "overcommitted, so a QoS below its cap can still be starved of hardware."
        )
    at_cap = [
        k
        for k in all_qos
        if (limit_by_name.get(k) or 0) > 0 and qos_usage.get(k, {}).get("in_use", 0) >= limit_by_name[k]
    ]
    if at_cap:
        L.append(f"- **QoS at their node cap ({len(at_cap)})**: {', '.join(at_cap)}")
    zero_cap = [k for k in all_qos if limit_by_name.get(k) == 0]
    if zero_cap:
        L.append(f"- **QoS capped at 0 nodes (cannot run)**: {', '.join(zero_cap)}")
    tot_blocked = sum(blocked_jobs.values())
    if tot_blocked:
        L.append(
            f"- **Held by `QOSGrpNodeLimit`**: {tot_blocked} jobs / "
            f"{sum(blocked_nodes.values())} nodes of demand"
        )

    # 4. Per-QoS usage
    L.append(h("4. Node Usage by QoS (running jobs)"))
    L.append(
        "> Nodes can be shared (mix), so the sum of per-QoS distinct nodes may exceed the number of physically busy nodes."
    )
    L.append("")
    qos_run = group_usage(jobs, nodes, "qos", "RUNNING")
    qos_pend = group_usage(jobs, nodes, "qos", "PENDING")
    L.append(
        "| QoS | Running jobs | Distinct nodes | Node-job slots | GPUs | Pending jobs | Pending nodes req |"
    )
    L.append("|-----|------|------|------|------|------|------|")
    for k in sorted(
        set(qos_run) | set(qos_pend), key=lambda x: -len(qos_run.get(x, {"nodes": set()})["nodes"])
    ):
        r = qos_run.get(k, {"jobs": 0, "node_slots": 0, "nodes": set(), "gpus": 0})
        p = qos_pend.get(k, {"jobs": 0, "node_slots": 0})
        L.append(
            f"| {k} | {r['jobs']} | {len(r['nodes'])} | {r['node_slots']} | {r['gpus']} | {p['jobs']} | {p['node_slots']} |"
        )

    # 5. Per-account usage
    L.append(h("5. Node Usage by Account (running jobs)"))
    acc_run = group_usage(jobs, nodes, "account", "RUNNING")
    acc_pend = group_usage(jobs, nodes, "account", "PENDING")
    L.append(
        "| Account | Running jobs | Distinct nodes | Node-job slots | GPUs | Pending jobs | Pending nodes req |"
    )
    L.append("|-----|------|------|------|------|------|------|")
    for k in sorted(
        set(acc_run) | set(acc_pend), key=lambda x: -len(acc_run.get(x, {"nodes": set()})["nodes"])
    ):
        r = acc_run.get(k, {"jobs": 0, "node_slots": 0, "nodes": set(), "gpus": 0})
        p = acc_pend.get(k, {"jobs": 0, "node_slots": 0})
        L.append(
            f"| {k} | {r['jobs']} | {len(r['nodes'])} | {r['node_slots']} | {r['gpus']} | {p['jobs']} | {p['node_slots']} |"
        )

    # 6. Reservations
    L.append(h("6. Reservations"))
    if resvs:
        L.append("| Name | Nodes | Users | End time |")
        L.append("|-----|------|------|------|")
        for r in resvs:
            nlist = expand_hostlist(r.get("Nodes", ""))
            L.append(
                f"| {r.get('ReservationName','?')} | {len(nlist)} | {r.get('Users','-')} | {r.get('EndTime','-')} |"
            )
    else:
        L.append("No active reservations.")

    # 7. Queue pressure
    L.append(h("7. Queue Pressure"))
    pend = [j for j in jobs if j["state"] == "PENDING"]
    run_j = [j for j in jobs if j["state"] == "RUNNING"]
    L.append(f"- **Running jobs**: {len(run_j)}")
    L.append(f"- **Pending jobs**: {len(pend)} (requesting {sum(j['nnodes'] for j in pend)} nodes total)")

    # 8. My jobs
    L.append(h("8. My Jobs (user=%s)" % user))
    mine = [j for j in jobs if j["user"] == user]
    if mine:
        L.append("| JobID | QoS | Account | State | Nodes | Nodelist |")
        L.append("|-----|------|------|------|------|------|")
        for j in mine:
            L.append(
                f"| {j['jobid']} | {j['qos']} | {j['account']} | {j['state']} | {j['nnodes']} | {j['nodelist'] or '-'} |"
            )
    else:
        L.append("No running/pending jobs.")

    # 9. Jobs by user (all users)
    L.append(h("9. Jobs by User (all users)"))
    L.append(
        "> One row per user. `Jobs` = total (R running / P pending); `Nodes` = distinct nodes held by running jobs."
    )
    L.append("")
    by_user = group_by_user(jobs, nodes)
    L.append("| User | Jobs (R/P) | Nodes | #Accts | Accounts | QoS | Partitions |")
    L.append("|-----|------|------|------|------|------|------|")
    for u in sorted(by_user, key=lambda x: (-len(by_user[x]["nodes"]), -by_user[x]["total"], x)):
        a = by_user[u]
        acct_disp = ",".join(sorted(a["accounts"]))
        qos_disp = ",".join(sorted((q or "(none)") for q in a["qos"]))
        part_disp = ",".join(sorted(a["partitions"]))
        L.append(
            f"| {u} | {a['total']} ({a['running']}R/{a['pending']}P) | {len(a['nodes'])} | "
            f"{len(a['accounts'])} | {acct_disp} | {qos_disp} | {part_disp} |"
        )

    # Appendix: common commands
    L.append(h("Appendix: Common Commands"))
    L.append(COMMANDS_APPENDIX)

    return "\n".join(L) + "\n"


COMMANDS_APPENDIX = """```bash
# --- Account / QoS ---
spur accounts show user                 # all user->account rows (grep yourself)
spur accounts show qos                  # global QoS list (Prio/Preempt/GrpTRES)
spur accounts show qos -P                # same, pipe-delimited (blank columns keep position)
spur accounts show account              # all accounts
sshare -u $(whoami)                     # fair-share info

# --- QoS limits and live usage (limit + distinct nodes in use) ---
scontrol show assoc_mgr                 # QOS Records, then Association Records
# per-QoS node cap and current usage, as `node=LIMIT(IN_USE)`:
scontrol show assoc_mgr | awk '
  /^Association Records/ { exit }               # stop before the per-account block
  /^QOS=/ { q=$1; sub("QOS=","",q); next }
  /GrpTRES=/ { match($0, /node=[0-9N]+\\([0-9]+\\)/)
               if (RSTART) print q, substr($0, RSTART, RLENGTH) }' | column -t
# which pending jobs are held by a QoS cap (vs waiting on hardware):
squeue -h -t PENDING -o "%q %D %R" | grep QOSGrpNodeLimit

# --- Cluster / nodes ---
sinfo                                   # partitions + node counts per state
sinfo -N -o "%N %P %t %G"               # per node: name/partition/state/GRES
sinfo -N -h -o "%t" | sort | uniq -c    # node count per state
scontrol show node <node>               # single-node detail (CPUAlloc/GRES/State)
scontrol show reservation               # reservations

# --- Jobs / queue ---
squeue                                   # queue
squeue -o "%i %P %q %a %u %T %D %N"      # custom columns (NOTE: delimiters render as spaces)
squeue -u $(whoami)                      # only your jobs
squeue -t PENDING                        # only pending jobs
scontrol show job <jobid>                # single-job detail (QOS/Account/NodeList)

# --- Submit (whole-node exclusive) ---
sbatch -A amd-primus -q amd-primus-qos -p amd-spur --exclusive -N <N> -t <t> <script>
scancel <jobid>                          # cancel a job

# --- Other spur-native views ---
spur priority                            # job priority breakdown
spur stat                                # running-job statistics
spur report cluster -s now-7days         # cluster utilization report
spur health                              # node health
```

> Spur quirks: custom `-o` delimiters render as spaces; positional filters for `show user`/`show account` are ignored (use grep/awk); the `sacctmgr` binary is not on PATH (use `spur accounts ...`, which is the same tool); there is no `scontrol show hostnames`; QoS membership is not enforced per account (only the account is), but per-QoS `GrpTRES=node=` caps *are* enforced - see `QOSGrpNodeLimit` in the pending reasons."""


def main():
    ap = argparse.ArgumentParser(description="Spur cluster status report")
    ap.add_argument("--user", default=None, help="user to report on (default: current user)")
    args = ap.parse_args()
    user = args.user or getpass.getuser()
    print(build_report(user))


if __name__ == "__main__":
    main()
