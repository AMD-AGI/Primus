###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run convergence tests in the background and report on them.

A convergence run takes hours, longer than a terminal or an agent session is
worth tying up. `start` launches run_convergence_test.sh detached from the
calling shell, after checking that the GPUs are free (or, with --queue, as soon
as they are); `status` reads the run's log and says where it is: building the
dataset, compiling, or training (with iteration, loss, step time and ETA), and
once it has finished, the verdict and the result files.

A plan that asks to probe first (a generated config, or a time budget) runs as
one job of two stages: the 20-iteration probe, then -- if it trained cleanly --
the full run, sized to the budget the probe measured. Nobody has to come back
in between.

    python3 tools/convergence_test/jobs.py start --plan output/convergence/plans/<plan>.json
    python3 tools/convergence_test/jobs.py start -- --model megatron/llama2_7B --source c4
    python3 tools/convergence_test/jobs.py status            # the latest job
    python3 tools/convergence_test/jobs.py list
    python3 tools/convergence_test/jobs.py stop <job>

Jobs live in output/convergence/jobs/<job>/: job.json, driver.log, exit_code.
"""

import argparse
import datetime
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_loss  # noqa: E402
from resolve_config import PRIMUS_PATH  # noqa: E402

TOOL_DIR = Path(__file__).resolve().parent
DRIVER = TOOL_DIR / "run_convergence_test.sh"
JOBS_DIR = PRIMUS_PATH / "output" / "convergence" / "jobs"
MARKER = "convergence-job"

# GPUs count as busy above these, e.g. another user's job outside docker.
BUSY_VRAM_MB = 4096
BUSY_ACTIVITY_PCT = 10

EXIT_MEANING = {
    0: "PASS: trained every iteration (and matched the baseline, if one was given)",
    1: "FAILED: a lint error, a crash, or an unreadable baseline",
    3: "FAIL: the loss moved from the baseline by more than the tolerance",
    4: "FAIL: training stopped before the last iteration",
    6: "FAIL: the probe had nan or skipped iterations, so the full run was not started",
}
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\[ERROR\]\s+\[|^\s*ERROR\s|(?:^|\] )error:|\bFAIL\b|\b\w+Error: |"
    r"out of memory|RESOURCE_EXHAUSTED|Training stopped|exited with code [1-9]"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Jobs on disk
# ---------------------------------------------------------------------------


def _jobs():
    jobs = []
    for meta in sorted(JOBS_DIR.glob("*/job.json")):
        try:
            jobs.append(json.loads(meta.read_text()))
        except (OSError, ValueError):
            continue
    return jobs


def _find(job_id):
    jobs = _jobs()
    if not jobs:
        raise SystemExit(f"no convergence jobs under {JOBS_DIR}")
    if not job_id:
        return jobs[-1]
    matches = [j for j in jobs if j["id"] == job_id] or [j for j in jobs if job_id in j["id"]]
    if len(matches) != 1:
        names = ", ".join(j["id"] for j in (matches or jobs)[-10:])
        raise SystemExit(f"{'ambiguous' if matches else 'no'} job {job_id!r}; jobs: {names}")
    return matches[0]


def _alive(pid):
    try:
        return MARKER in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        return False


def state(job):
    job_dir = Path(job["dir"])
    exit_file = job_dir / "exit_code"
    if exit_file.exists():
        try:
            return "finished", int(exit_file.read_text().strip())
        except ValueError:
            return "finished", None
    if _alive(job["pid"]):
        return "running", None
    if (job_dir / "stopped").exists():
        return "stopped", None
    return "died", None


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def busy_gpus():
    """[(gpu, vram_mb, activity_pct)] for GPUs something else is using."""
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "--usage", "--mem-usage", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        data = json.loads(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    gpus = data if isinstance(data, list) else data.get("gpu_data", [])
    busy = []
    for gpu in gpus:
        try:
            vram = float(gpu["mem_usage"]["used_vram"]["value"])
            activity = float(gpu["usage"]["gfx_activity"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if vram > BUSY_VRAM_MB or activity > BUSY_ACTIVITY_PCT:
            busy.append((gpu.get("gpu"), vram, activity))
    return busy


def _containers():
    """{full container id: name} of running containers."""
    out = subprocess.run(
        ["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"], capture_output=True, text=True
    ).stdout
    return dict(line.split(" ", 1) for line in out.splitlines() if " " in line)


def gpu_holders():
    """{gpu: ["PID 123 (python) in container foo, 25.8 GB", ...]} for processes holding GPU memory."""
    try:
        out = subprocess.run(
            ["amd-smi", "process", "--json"], capture_output=True, text=True, timeout=30
        ).stdout
        data = json.loads(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    containers = _containers()
    holders = {}
    for gpu in data if isinstance(data, list) else []:
        for entry in gpu.get("process_list") or []:
            proc = entry.get("process_info") or {}
            try:
                vram = float(proc["memory_usage"]["vram_mem"]["value"]) / 2**20
                pid = int(proc["pid"])
            except (KeyError, TypeError, ValueError):
                continue
            if vram < 1:
                continue
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip()
                cgroup = Path(f"/proc/{pid}/cgroup").read_text()
            except OSError:
                comm, cgroup = proc.get("name", "?"), ""
            where = next((f" in container {name}" for cid, name in containers.items() if cid in cgroup), "")
            holders.setdefault(gpu.get("gpu"), []).append(f"PID {pid} ({comm}){where}, {vram / 1024:.1f} GB")
    return holders


def _queued(job):
    return (Path(job["dir"]) / "queued").exists()


def preflight(exclude=None):
    """(problems, only_ours): what keeps the node busy, and whether it is all convergence jobs.

    Queued jobs hold nothing yet, so they do not count.
    """
    jobs_running = [
        f"convergence job {j['id']} is still running"
        for j in _jobs()
        if j["id"] != exclude and not _queued(j) and state(j)[0] == "running"
    ]
    others = [
        f"training container {c} is running"
        for c in _containers().values()
        if c.startswith("primus-training")
    ]
    busy = busy_gpus()
    holders = gpu_holders() if busy else {}
    for gpu, vram, activity in busy:
        held = "; ".join(holders.get(gpu, [])) or "held by a process this user cannot see"
        others.append(f"GPU {gpu} is in use ({vram / 1024:.1f} GB, {activity:.0f}% busy): {held}")
    # A running job's own container and GPUs are not a second occupant.
    only_ours = bool(jobs_running) and all(
        "primus-training" in o or "in container primus-training" in o for o in others
    )
    return jobs_running + others, only_ours


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _job_name(driver_args):
    name = "run"
    for flag in ("--model", "--config"):
        if flag in driver_args:
            value = driver_args[driver_args.index(flag) + 1]
            name = Path(value).name.replace("-convergence.yaml", "").replace("/", "-")
    if "--probe" in driver_args:
        name += "-probe"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def _with_train_iters(driver_args, train_iters):
    """The driver arguments with the run length set, ahead of any training overrides after --."""
    cut = driver_args.index("--") if "--" in driver_args else len(driver_args)
    head, tail = list(driver_args[:cut]), list(driver_args[cut:])
    if "--train-iters" in head:
        head[head.index("--train-iters") + 1] = str(train_iters)
    else:
        head += ["--train-iters", str(train_iters)]
    return head + tail


def _plan_stages(plan_file, probe=False, train_iters=None, no_probe=False):
    """What a plan runs: its probe, its full run, or the probe and then the full run.

    A plan that asks to probe first (a generated config, a time budget) does both
    in one job, so the full run starts even if nobody is there when the probe ends;
    with a budget, the full run takes the iteration count the probe measured.
    """
    plan = json.loads(Path(plan_file).read_text())
    if plan.get("status") != "ready":
        raise SystemExit(f"{plan_file} is not a ready plan")
    probe_stage = {"name": "probe", "args": list(plan["probe_driver_args"])}
    full_args = list(plan["driver_args"])
    if train_iters:
        full_args = _with_train_iters(full_args, train_iters)
    if probe:
        return [probe_stage]
    if train_iters or no_probe or not plan.get("probe_first"):
        return [{"name": "run", "args": full_args}]
    budget = "--budget-hours" in probe_stage["args"]
    return [probe_stage, {"name": "full run", "args": full_args, "budget_from_probe": budget}]


def start(args):
    driver_args = args.driver_args
    if driver_args and driver_args[0] == "--":
        driver_args = driver_args[1:]
    if args.plan:
        if driver_args:
            raise SystemExit("give either --plan or driver arguments, not both")
        stages = _plan_stages(args.plan, args.probe, args.train_iters, args.no_probe)
    elif args.probe or args.train_iters or args.no_probe:
        raise SystemExit(
            "--probe, --no-probe and --train-iters go with --plan; otherwise pass them to the driver after --"
        )
    elif not driver_args:
        raise SystemExit("usage: jobs.py start --plan <plan.json> | -- <run_convergence_test.sh arguments>")
    else:
        stages = [{"name": "run", "args": driver_args}]
    driver_args = stages[-1]["args"]
    problems, only_ours = preflight()
    queue = bool(problems) and args.queue and not args.force
    if problems and not args.force and not queue:
        print("not starting; the node is busy:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if only_ours:
            print(
                "add --queue to start it as soon as that job is done, or stop it with: jobs.py stop <job>",
                file=sys.stderr,
            )
        else:
            print(
                "something other than a convergence job holds the GPUs; add --queue to start as soon "
                "as it is done (--force starts anyway, sharing the GPUs)",
                file=sys.stderr,
            )
        raise SystemExit(2)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{stamp}-{args.name or _job_name(driver_args)}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    log = job_dir / "driver.log"
    if queue:
        (job_dir / "queued").write_text("; ".join(problems) + "\n")
    driver = str(DRIVER.relative_to(PRIMUS_PATH))
    job = {
        "id": job_id,
        "pid": None,
        "dir": str(job_dir),
        "log": str(log),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "driver_args": driver_args,
        "command": "  then  ".join(shlex.join([driver] + s["args"]) for s in stages),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    # A new session: the job survives the shell, terminal or agent that started it.
    wrapper = 'cd "$1" && python3 "$2" run --dir "$3"; echo $? > "$3/exit_code"'
    with open(log, "wb") as out:
        proc = subprocess.Popen(
            ["bash", "-c", wrapper, MARKER, str(PRIMUS_PATH), str(Path(__file__).resolve()), str(job_dir)],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    job["pid"] = proc.pid
    (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    time.sleep(3)
    if state(job)[0] != "running":
        print(report(job))
        raise SystemExit(1)
    print(f"{'queued' if queue else 'started'} {job_id}")
    if queue:
        print("  it starts by itself once the node is free")
    print("  stages : " + ", then ".join(s["name"] for s in stages))
    print(f"  log    : {log}")
    print(f"  status : python3 {Path(__file__).relative_to(PRIMUS_PATH)} status {job_id}")


def wait_turn(job_id, job_dir, interval=60):
    """Block a queued job until no earlier queued job waits and the node is free."""
    marker = Path(job_dir) / "queued"
    reported = None
    while True:
        ahead = [j for j in _jobs() if j["id"] < job_id and _queued(j) and state(j)[0] == "running"]
        problems, _ = preflight(exclude=job_id)
        reasons = [f"job {j['id']} is ahead in the queue" for j in ahead] + problems
        if not reasons:
            marker.unlink(missing_ok=True)
            print(f"[jobs] the node is free; starting at {datetime.datetime.now():%H:%M:%S}", flush=True)
            return
        marker.write_text("; ".join(reasons) + "\n")
        # Log a new reason, not every change in its memory and activity figures.
        gist = re.sub(r"[\d.]+", "#", "; ".join(reasons))
        if gist != reported:
            print(f"[jobs] waiting: {'; '.join(reasons)}", flush=True)
            reported = gist
        time.sleep(interval)


STAGE_RE = re.compile(r"^\[jobs\] stage (\d+)/(\d+): (.+)$", re.MULTILINE)


def run_stages(args):
    """The job itself, in its own session: wait its turn, then run each stage."""
    job_dir = Path(args.dir)
    job = json.loads((job_dir / "job.json").read_text())
    if (job_dir / "queued").exists():
        wait_turn(job["id"], job_dir, args.interval)
    stages = job["stages"]
    code = 0
    for number, stage in enumerate(stages, 1):
        driver_args = list(stage["args"])
        if stage.get("budget_from_probe"):
            budget = progress(job).get("recommended_iterations")
            if budget:
                driver_args = _with_train_iters(driver_args, budget)
                print(f"[jobs] the probe fits {budget} iterations in the time budget", flush=True)
            else:
                print("[jobs] the probe reported no budget; running the planned length", flush=True)
        print(f"[jobs] stage {number}/{len(stages)}: {stage['name']}", flush=True)
        print(f"[jobs] {shlex.join([str(DRIVER.relative_to(PRIMUS_PATH))] + driver_args)}", flush=True)
        code = subprocess.call([str(DRIVER), *driver_args], cwd=PRIMUS_PATH)
        if code:
            if number < len(stages):
                print(f"[jobs] the {stage['name']} exited {code}; not starting the {stages[number]['name']}")
            return code
        if number < len(stages) and progress(job).get("nan_or_skipped"):
            print(
                f"[jobs] the {stage['name']} had nan or skipped iterations; not starting the {stages[number]['name']}"
            )
            return 6
    return code


def _duration(seconds):
    seconds = int(seconds)
    return (
        f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
        if seconds >= 3600
        else f"{seconds // 60}m{seconds % 60:02d}s"
    )


def progress(job):
    """What the job's current stage is doing, from its log."""
    log = Path(job["log"])
    full_text = ANSI_RE.sub("", log.read_text(errors="ignore")) if log.exists() else ""
    info = {"phase": "starting", "errors": [], "results": [], "summary": [], "plan": []}
    budgets = re.findall(r"recommended iters : (\d+)", full_text)
    if budgets:
        info["recommended_iterations"] = int(budgets[-1])
    # Each stage (probe, full run) is reported on its own.
    stages = list(STAGE_RE.finditer(full_text))
    text = full_text
    if stages:
        last = stages[-1]
        info["stage"] = f"{last.group(3)} (stage {last.group(1)}/{last.group(2)})"
        text = full_text[last.start() :]
    lines = text.splitlines()
    info["plan"] = [l for l in lines if l.startswith("[convergence] ") and " : " in l][:12]

    runs = []
    if "lm loss:" in text or "completed step:" in text:
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".log") as stage_log:
                stage_log.write(text)
                stage_log.flush()
                runs = plot_loss.parse_log(stage_log.name)
        except Exception:  # noqa: BLE001 - a log being written
            runs = []
    if runs:
        train, valid = runs[-1]
        last = train[-1]
        info["phase"] = "training"
        info["iteration"] = last["iteration"]
        info["total"] = last.get("total_iters")
        info["loss"] = last.get("loss")
        info["first_loss"] = train[0].get("loss")
        if valid:
            info["valid_loss"] = valid[-1]["loss"]
            info["valid_iteration"] = valid[-1]["iteration"]
        step_ms = plot_loss.steady_state_ms(train)
        if step_ms:
            info["seconds_per_iteration"] = step_ms / 1000
            if info["total"]:
                info["eta_seconds"] = (info["total"] - info["iteration"]) * step_ms / 1000
        info["nan_or_skipped"] = int(max((r.get("nan", 0) + r.get("skipped", 0) for r in train), default=0))
        info["nan_or_skipped"] += sum(
            1 for r in train if r.get("loss") is not None and r["loss"] != r["loss"]
        )
        if "peak_mem_pct" in last:
            info["peak_memory_pct"] = last["peak_mem_pct"]
    elif "[convergence] launching:" in text:
        info["phase"] = "starting the container and compiling"
    elif "convergence lint" in text:
        info["phase"] = "checking the config"
    elif "[prepare-dataset]" in text:
        info["phase"] = "building the dataset"
        building = [l for l in lines if l.startswith("[prepare-dataset]")]
        info["detail"] = building[-1].replace("[prepare-dataset]", "").strip() if building else ""
    if "[plot-loss]" in text:
        info["phase"] = "checking the result"

    for line in lines:
        match = re.search(r"\[plot-loss\] wrote (\S+)", line)
        if match:
            info["results"].append(match.group(1))
        match = re.search(r"\[convergence\] console log: (\S+)", line)
        if match:
            info["console_log"] = match.group(1)
    # What plot_loss prints after its "<label>: <log> (N points, ...)" line: the
    # summary, the budget, and the baseline verdict, all indented.
    headers = [i for i, l in enumerate(lines) if re.match(r"^\[plot-loss\] .+ \(\d+ points", l)]
    if headers:
        info["summary"] = [l for l in lines[headers[-1] + 1 :] if l.startswith("  ") and l.strip()]
    errors = []
    for line in lines:
        # Exception and FAIL/ERROR lines; traceback frames only bury the cause.
        if (
            not ERROR_RE.search(line)
            or "UserWarning" in line
            or ' File "' in line
            or "Traceback (most" in line
        ):
            continue
        line = line.strip()[:240]
        if line not in errors:
            errors.append(line)
    info["errors"] = errors[-8:]
    return info


def report(job):
    status, code = state(job)
    info = progress(job)
    started = datetime.datetime.fromisoformat(job["started"])
    elapsed = (datetime.datetime.now() - started).total_seconds()
    out = [f"job      : {job['id']}", f"command  : {job['command']}"]
    if status == "finished":
        meaning = EXIT_MEANING.get(code, "FAILED; see the log")
        if code == 0 and "--prepare-only" in job["driver_args"]:
            meaning = "the dataset is ready"
        elif code == 0 and "--plot-only" in job["driver_args"]:
            meaning = "re-plotted"
        out.append(f"state    : finished, exit {code} -- {meaning}")
    elif status == "running" and _queued(job):
        waiting = (Path(job["dir"]) / "queued").read_text().strip()
        out.append(f"state    : queued for {_duration(elapsed)}, waiting for: {waiting}")
    elif status == "running":
        stage = f"{info['stage']}: " if info.get("stage") else ""
        out.append(f"state    : running for {_duration(elapsed)}, {stage}{info['phase']}")
    elif status == "stopped":
        when = (Path(job["dir"]) / "stopped").read_text().strip()
        out.append(f"state    : stopped with jobs.py stop at {when}; no verdict")
    else:
        out.append("state    : died without an exit status (killed, or the node rebooted); no verdict")
    if status in ("stopped", "died") and "iteration" in info and info.get("console_log"):
        prefix = info["console_log"][: -len(".log")]
        out.append(
            f"partial  : python3 {Path(__file__).relative_to(PRIMUS_PATH).parent}/plot_loss.py "
            f"{info['console_log']} --out {prefix}"
        )
    if info.get("detail"):
        out.append(f"detail   : {info['detail']}")
    if "iteration" in info:
        total = f"/{info['total']}" if info.get("total") else ""
        out.append(
            f"progress : iteration {info['iteration']}{total}, loss {info['first_loss']:.4f} -> {info['loss']:.4f}"
        )
        if "valid_loss" in info:
            out.append(f"valid    : {info['valid_loss']:.4f} at iteration {info['valid_iteration']}")
        if "seconds_per_iteration" in info:
            eta = (
                f", about {_duration(info['eta_seconds'])} left"
                if status == "running" and "eta_seconds" in info
                else ""
            )
            out.append(f"speed    : {info['seconds_per_iteration']:.2f} s/iteration{eta}")
        health = f"nan/skipped {info['nan_or_skipped']}"
        if "peak_memory_pct" in info:
            health += f", peak memory {info['peak_memory_pct']:.0f}%"
        flag = "   <-- investigate" if info["nan_or_skipped"] else ""
        out.append(f"health   : {health}{flag}")
    if "recommended_iterations" in info:
        out.append(f"budget   : {info['recommended_iterations']} iterations fit the requested time")
    if status != "running" and info["summary"]:
        out.append("summary  :")
        out += [f"  {line.strip()}" for line in info["summary"]]
    if info["results"]:
        out.append("results  : " + ", ".join(info["results"]))
    if info.get("console_log"):
        out.append(f"console  : {info['console_log']}")
    passed = status == "finished" and code == 0
    if info["errors"] and not passed and (status != "running" or info["phase"] != "training"):
        out.append("errors   :")
        out += [f"  {line}" for line in info["errors"]]
    out.append(f"log      : {job['log']}")
    return "\n".join(out)


def status_cmd(args):
    job = _find(args.job)
    if args.json:
        status, code = state(job)
        print(json.dumps({**job, "state": status, "exit_code": code, **progress(job)}, indent=2, default=str))
    else:
        print(report(job))


def list_cmd(args):
    jobs = _jobs()
    if not jobs:
        print(f"no convergence jobs under {JOBS_DIR}")
        return
    for job in jobs[-args.last :]:
        status, code = state(job)
        info = progress(job)
        if status == "running" and _queued(job):
            status, detail = "queued", "waiting for the node"
        else:
            detail = (
                info["phase"] if status == "running" else f"exit {code}" if status == "finished" else status
            )
            if status == "running" and info.get("stage"):
                detail = f"{info['stage']}: {detail}"
        if "iteration" in info:
            detail += f", iteration {info['iteration']}/{info.get('total') or '?'} loss {info['loss']:.3f}"
        print(f"{job['id']:56} {status:9} {detail}")


def stop_cmd(args):
    job = _find(args.job)
    if state(job)[0] != "running":
        print(f"{job['id']} is not running")
        return
    text = Path(job["log"]).read_text(errors="ignore")
    containers = sorted(set(re.findall(r"--name (primus-training-\d+)", text)))
    try:
        os.killpg(job["pid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    for container in containers:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    (Path(job["dir"]) / "stopped").write_text(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
    print(f"stopped {job['id']}" + (f" and removed {', '.join(containers)}" if containers else ""))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start", help="Launch run_convergence_test.sh in the background")
    p.add_argument("--plan", help="A plan saved by plan_request.py")
    p.add_argument("--probe", action="store_true", help="With --plan: run only the plan's probe")
    p.add_argument("--train-iters", type=int, help="With --plan: run this many iterations, no probe")
    p.add_argument("--no-probe", action="store_true", help="With --plan: skip the probe the plan asks for")
    p.add_argument("--name", help="Job name (default: from the model)")
    p.add_argument("--queue", action="store_true", help="If the node is busy, start as soon as it is free")
    p.add_argument("--force", action="store_true", help="Start even if the GPUs look busy")
    p.add_argument("driver_args", nargs=argparse.REMAINDER, help="-- then run_convergence_test.sh arguments")
    p.set_defaults(func=start)
    p = sub.add_parser("status", help="Where a job is (default: the latest)")
    p.add_argument("job", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=status_cmd)
    p = sub.add_parser("list", help="Recent jobs")
    p.add_argument("--last", type=int, default=20)
    p.set_defaults(func=list_cmd)
    p = sub.add_parser("stop", help="Stop a running or queued job and its container")
    p.add_argument("job")
    p.set_defaults(func=stop_cmd)
    p = sub.add_parser("run", help=argparse.SUPPRESS)  # the detached job itself
    p.add_argument("--dir", required=True)
    p.add_argument("--interval", type=float, default=60, help="Seconds between checks while queued")
    p.set_defaults(func=run_stages)
    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
