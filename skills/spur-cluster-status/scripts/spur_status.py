#!/usr/bin/env python3
"""Collect Spur cluster node-allocation status and print a Markdown report.

Targets the AMD "Spur" scheduler (SLURM-compatible CLI: sinfo/squeue/scontrol/
spur accounts). Read-only. Handles Spur quirks: `-o` delimiters are rendered as
spaces, positional filters are ignored, there is no `scontrol show hostnames`,
and QoS is not enforced per account (only the account is).

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


def parse_qos():
    """Return list of dicts: {name, prio, preempt}."""
    text = run(["spur", "accounts", "show", "qos"])
    rows = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 2 or toks[0] in ("Name", "----"):
            continue
        rows.append(
            {
                "name": toks[0],
                "prio": toks[1] if len(toks) > 1 else "",
                "preempt": toks[2] if len(toks) > 2 else "",
            }
        )
    return rows


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
    L.append("| QoS | Prio | Preempt |")
    L.append("|-----|------|---------|")
    for q in sorted(qos_rows, key=lambda x: -int(x["prio"]) if x["prio"].isdigit() else 0):
        L.append(f"| {q['name']} | {q['prio']} | {q['preempt']} |")

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

    # 3. Per-QoS usage
    L.append(h("3. Node Usage by QoS (running jobs)"))
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

    # 4. Per-account usage
    L.append(h("4. Node Usage by Account (running jobs)"))
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

    # 5. Reservations
    L.append(h("5. Reservations"))
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

    # 6. Queue pressure
    L.append(h("6. Queue Pressure"))
    pend = [j for j in jobs if j["state"] == "PENDING"]
    run_j = [j for j in jobs if j["state"] == "RUNNING"]
    L.append(f"- **Running jobs**: {len(run_j)}")
    L.append(f"- **Pending jobs**: {len(pend)} (requesting {sum(j['nnodes'] for j in pend)} nodes total)")

    # 7. My jobs
    L.append(h("7. My Jobs (user=%s)" % user))
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

    # 8. Jobs by user (all users)
    L.append(h("8. Jobs by User (all users)"))
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
sacctmgr show user $(whoami)            # equivalent alias
spur accounts show qos                  # global QoS list (Prio/Preempt)
spur accounts show account              # all accounts
sshare -u $(whoami)                     # fair-share info

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

> Spur quirks: custom `-o` delimiters render as spaces; positional filters for `show user`/`show account` are ignored (use grep/awk); `sacctmgr` policy queries are blocked; there is no `scontrol show hostnames`; the association table is empty and QoS is not enforced per account (only the account is)."""


def main():
    ap = argparse.ArgumentParser(description="Spur cluster status report")
    ap.add_argument("--user", default=None, help="user to report on (default: current user)")
    args = ap.parse_args()
    user = args.user or getpass.getuser()
    print(build_report(user))


if __name__ == "__main__":
    main()
