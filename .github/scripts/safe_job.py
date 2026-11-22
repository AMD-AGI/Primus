"""Runs a job's GPU steps in order on one 8-GPU Primus-SaFE workload over the SaFE SSH gateway.

usage: safe_job.py start <script> (submits the workload and hands it $GITHUB_WORKSPACE) | <script> |
       run <command> | fetch <path>... (copies them back into $GITHUB_WORKSPACE) | delete
Each step returns its script's exit code. GPU_WORKSPACES is "workspace:priority ..."; a workload still
queued after QUEUE_MINUTES moves to the next one. PASS_ENV names the variables the workload gets and
STEP_ENV the ones each step gets, kept out of the workload's spec; lines a step appends to $SAFE_ENV are
sourced before the later steps. With IMAGE_PULL_USER and IMAGE_PULL_TOKEN, the workload pulls IMAGE
with them through a SaFE image secret.
"""

import base64
import io
import json
import os
import re
import shlex
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

import paramiko

API = "https://global.primus-safe.amd.com/api/v1"
CERTS = "https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/"
GATEWAY = ("global.primus-safe.amd.com", 2222)
TEMP = os.environ["RUNNER_TEMP"]
NAME = f"primus-ci-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
STATE = os.path.join(TEMP, "safe-workload.json")
WORKSPACES = [entry.split(":") for entry in os.environ["GPU_WORKSPACES"].split()]
# SSH sessions do not inherit the container's environment; take it from its main process. Workloads
# reach the shared caches as root or as the submitting user, depending on the workspace.
LAUNCH = """cd /ci/src && test ! -e /ci/{n}.log && SAFE_ENV=/ci/env setsid nohup bash -c '
  while IFS= read -r -d "" kv; do export "$kv"; done < /proc/1/environ
  umask 000; set -a; . /ci/env; set +a
  bash /ci/{n}.sh > /ci/{n}.log 2>&1; echo $? > /ci/{n}.exit' > /dev/null 2>&1 < /dev/null &"""
POLL = """tail -c +{start} /ci/{n}.log 2> /dev/null; if [ ! -d /ci/src ]; then echo restarted >&2;
  elif [ -s /ci/{n}.exit ]; then echo exit=$(cat /ci/{n}.exit) >&2; fi"""


def log(message):
    print(message, flush=True)


def fail(message):
    log(f"::error::{message}")
    sys.exit(1)


def clock():
    return time.strftime("%H:%M:%S", time.gmtime())


def tls_context():
    ca = os.path.join(TEMP, "amd-ca.pem")
    if not os.path.exists(ca):
        certs = (
            urllib.request.urlopen(CERTS + f, timeout=60).read()
            for f in ("amd-root-ca.crt", "amd-issuing-ca.crt")
        )
        with open(ca, "wb") as f:
            f.write(b"".join(certs))
    return ssl.create_default_context(cafile=ca)


def api(method, path="", body=None, resource="workloads"):
    key = os.environ.get("SAFE_API_KEY") or fail(
        "SAFE_API_KEY is empty: the secret is not available to this run"
    )
    request = urllib.request.Request(
        f"{API}/{resource}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60, context=tls_context()) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"{method} {path or '/'}: HTTP {e.code} {e.read().decode(errors='replace')[:500]}"
        ) from e


def delete_workloads():
    for workspace, _ in WORKSPACES:
        for item in api("GET", f"?workspaceId={workspace}&kind=Deployment&limit=100").get("items") or []:
            if item["workloadId"].startswith(f"{NAME}-"):
                log(f"🧹 Deleting the workload {item['workloadId']}")
                try:
                    api("DELETE", f"/{item['workloadId']}")
                except RuntimeError as e:
                    log(f"::warning::Could not delete the workload {item['workloadId']}: {e}")


def image_pull_secret():
    """Creates or refreshes the SaFE secret the workload pulls IMAGE with; returns its id."""
    registry = os.environ["IMAGE"].split("/")[0]
    params = {
        "server": "https://index.docker.io/v1/" if registry == "docker.io" else registry,
        "username": os.environ["IMAGE_PULL_USER"],
        "password": base64.b64encode(os.environ["IMAGE_PULL_TOKEN"].encode()).decode(),
    }
    body = {
        "params": [params],
        "workspaceIds": [workspace for workspace, _ in WORKSPACES],
    }
    for item in api("GET", "?type=image", resource="secrets").get("items") or []:
        if item["secretName"] == "primus-ci-image-pull":
            api("PATCH", f"/{item['secretId']}", body, resource="secrets")
            return item["secretId"]
    body.update(name="primus-ci-image-pull", type="image")
    return api("POST", "", body, resource="secrets")["secretId"]


def submit(index):
    workspace, priority = WORKSPACES[index]
    body = {
        "displayName": NAME,
        "workspaceId": workspace,
        "groupVersionKind": {"kind": "Deployment", "version": "v1"},
        "images": [os.environ["IMAGE"]],
        "entryPoints": [base64.b64encode(b"exec sleep infinity\n").decode()],
        "resources": [
            {
                "replica": 1,
                "gpu": "8",
                "cpu": "96",
                "memory": "512Gi",
                "sharedMemory": "64Gi",
                "ephemeralStorage": "300Gi",
            }
        ],
        "env": {
            name: os.environ[name] for name in os.environ.get("PASS_ENV", "").split() if name in os.environ
        },
        "priority": int(priority),
        "useWorkspaceStorage": True,
    }
    if os.environ.get("IMAGE_PULL_TOKEN"):
        body["secrets"] = [{"id": image_pull_secret(), "type": "image"}]
    workload = api("POST", "", body).get("workloadId") or fail("SaFE returned no workload id")
    log(
        f"🚀 {clock()} Workload {workload} in {workspace} at priority {priority}, image {os.environ['IMAGE']}"
    )
    return workload


class Watch:
    """Prints the workload's state changes and fails the step if the workload ends."""

    def __init__(self, workload):
        self.workload, self.last, self.lost = workload, None, 0

    def check(self):
        try:
            status = api("GET", f"/{self.workload}")
        except (RuntimeError, OSError):
            self.lost += 1
            if self.lost >= 8:
                fail(f"Workload {self.workload} is gone: deleted or stopped by the platform")
            return {}
        self.lost = 0
        state = f"{status.get('phase')}: {(status.get('message') or '')[:200]}"
        if state != self.last:
            log(f"⏳ {clock()} Workload {self.workload} {state}")
            self.last = state
        if status.get("phase") in ("Failed", "Stopped"):
            fail(f"Workload {self.workload} {status['phase']}")
        return status


class Pod:
    """Runs commands in the workload's pod over the SaFE SSH gateway, reconnecting as needed."""

    def __init__(self, user):
        self.user, self.client = user, None

    def run(self, command, stdin=None):
        """Returns (exit code, stdout, stderr); `stdin` is bytes or an iterable of byte chunks."""
        if (
            self.client is None
            or not self.client.get_transport()
            or not self.client.get_transport().is_active()
        ):
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(
                *GATEWAY,
                username=self.user,
                key_filename=os.path.join(TEMP, "safe-ssh-key"),
                look_for_keys=False,
                allow_agent=False,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
            self.client.get_transport().set_keepalive(30)
        channel = self.client.get_transport().open_session()
        channel.exec_command(command)
        if stdin is not None:
            for chunk in [stdin] if isinstance(stdin, bytes) else stdin:
                channel.sendall(chunk)
            channel.shutdown_write()
        out, err = channel.makefile("rb").read(), channel.makefile_stderr("rb").read()
        return channel.recv_exit_status(), out, err

    def try_run(self, command, stdin=None, attempts=1):
        """Like run(), retrying a dropped connection; returns None if every attempt fails."""
        for attempt in range(attempts):
            try:
                return self.run(command, stdin)
            except (paramiko.SSHException, OSError, EOFError):
                self.client = None
                if attempt + 1 < attempts:
                    time.sleep(15)
        return None


def start():
    key = os.environ.get("SAFE_SSH_KEY") or fail(
        "SAFE_SSH_KEY is empty: the secret is not available to this run"
    )
    with open(
        os.open(
            os.path.join(TEMP, "safe-ssh-key"),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        ),
        "w",
    ) as f:
        f.write(key.strip() + "\n")
    index = 0
    workload, submitted = submit(index), time.time()
    watch = Watch(workload)
    while watch.check().get("phase") != "Running":
        time.sleep(15)
        if (
            index + 1 < len(WORKSPACES)
            and time.time() - submitted > int(os.environ.get("QUEUE_MINUTES", 10)) * 60
        ):
            log(f"↪️ {clock()} Still queued in {WORKSPACES[index][0]}; moving to {WORKSPACES[index + 1][0]}")
            try:
                api("DELETE", f"/{workload}")
            except RuntimeError as e:
                log(f"::warning::Could not delete the workload {workload}: {e}")
            index += 1
            workload, submitted = submit(index), time.time()
            watch = Watch(workload)
    minutes = (time.time() - submitted) // 60
    log(f"✅ {clock()} Workload {workload} started in {WORKSPACES[index][0]} after {minutes:.0f} minutes")
    for _ in range(10):
        ssh = (watch.check().get("pods") or [{}])[0].get("sshCommand") or ""
        pod = Pod(m.group(1)) if (m := re.search(r"(\S+)@", ssh)) else None
        if pod and pod.try_run("true"):
            break
        time.sleep(15)
    else:
        fail(f"Cannot reach workload {workload} over SSH")
    archive = subprocess.Popen(
        [
            "tar",
            "-C",
            os.environ["GITHUB_WORKSPACE"],
            "--exclude=./.git",
            "--exclude=./build",
            "-cf",
            "-",
            ".",
        ],
        stdout=subprocess.PIPE,
    )
    code, _, err = pod.run(
        "mkdir -p /ci/src && touch /ci/env && tar -xf - -C /ci/src",
        iter(lambda: archive.stdout.read(1 << 20), b""),
    )
    if archive.wait() or code:
        fail(f"Could not hand the build to workload {workload}: {err.decode(errors='replace')[-500:]}")
    log(f"📦 {clock()} Handed the build to workload {workload}")
    with open(STATE, "w") as f:
        json.dump({"workload": workload, "user": pod.user}, f)
    return workload, pod


def main(args):
    if args[:1] == ["delete"]:
        return delete_workloads()
    if args[:1] == ["start"]:
        workload, pod = start()
        args = args[1:]
    elif os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
        workload, pod = state["workload"], Pod(state["user"])
    else:
        fail("No GPU workload: the step that starts it did not succeed")

    if args[:1] == ["fetch"]:
        paths = " ".join(shlex.quote(path) for path in args[1:])
        result = pod.try_run(f"cd /ci/src && tar -cf - $(ls -d {paths} 2> /dev/null)", attempts=5)
        if result is None or result[0]:
            fail(f"Could not fetch {' '.join(args[1:])} from workload {workload}")
        with tarfile.open(fileobj=io.BytesIO(result[1])) as archive:
            archive.extractall(os.environ["GITHUB_WORKSPACE"], filter="data")
        return 0
    if args[:1] == ["run"]:
        script = args[1].encode() + b"\n"
    else:
        with open(args[0], "rb") as f:
            script = f.read()
    names = [name for name in os.environ.get("STEP_ENV", "").split() if name in os.environ]
    script = "".join(f"export {name}={shlex.quote(os.environ[name])}\n" for name in names).encode() + script

    steps = os.path.join(TEMP, "safe-steps")
    n = 1
    if os.path.exists(steps):
        with open(steps) as f:
            n = int(f.read()) + 1
    with open(steps, "w") as f:
        f.write(str(n))
    result = pod.try_run(f"test -d /ci/src && cat > /ci/{n}.sh", script, attempts=5)
    if result is None or result[0]:
        fail(f"Cannot hand the step to workload {workload}: its pod restarted or SSH failed")
    if pod.try_run(LAUNCH.format(n=n), attempts=5) is None:
        fail(f"Cannot start the step on workload {workload}: SSH failed")

    # Read the step's log and status in short calls, so a dropped connection does not end the step.
    watch, offset, failures = Watch(workload), 0, 0
    while True:
        time.sleep(10)
        result = pod.try_run(POLL.format(start=offset + 1, n=n))
        if result is None:
            failures += 1
            if failures >= 30:
                fail(f"Lost the SSH connection to workload {workload}")
            watch.check()
            continue
        failures = 0
        _, out, err = result
        sys.stdout.buffer.write(out)
        sys.stdout.flush()
        offset += len(out)
        status = err.decode(errors="replace").strip()
        if status == "restarted":
            fail("The GPU pod restarted and lost its state")
        if status.startswith("exit="):
            sys.stdout.buffer.write(pod.run(f"tail -c +{offset + 1} /ci/{n}.log")[1])
            sys.stdout.flush()
            return int(status[len("exit=") :])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except RuntimeError as e:
        fail(str(e))
