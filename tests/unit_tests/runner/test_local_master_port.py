###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""runner/helpers/local_master_port.py: the single-node MASTER_PORT fallback (stdlib only)."""

import importlib.util
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "runner/helpers/local_master_port.py"
DIRECT = ROOT / "runner/primus-cli-direct.sh"


def _mod():
    spec = importlib.util.spec_from_file_location("local_master_port", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hold(family, host):
    sock = socket.socket(family, socket.SOCK_STREAM)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    sock.bind((host, 0))
    sock.listen(1)
    return sock


@pytest.fixture
def busy_port():
    sock = _hold(socket.AF_INET, "127.0.0.1")
    yield sock.getsockname()[1]
    sock.close()


@pytest.fixture
def busy_v6_port():
    # a port held by a v6-only listener only: free on IPv4, busy for torchrun on "::"
    mod = _mod()
    if not mod.has_ipv6():
        pytest.skip("no IPv6 on this host")
    for _ in range(20):
        sock = _hold(socket.AF_INET6, "::")
        port = sock.getsockname()[1]
        try:
            mod._listen(socket.AF_INET, port)
            break
        except OSError:
            sock.close()
    else:
        pytest.skip("no port free on IPv4 and held on IPv6")
    yield port
    sock.close()


def _free_port():
    port = _mod().find_unused_port()
    assert port
    return port


@pytest.mark.parametrize("addr", ["127.0.0.1", "localhost", "127.0.1.1", "::1"])
def test_a_busy_port_on_a_local_single_node_run_is_replaced(busy_port, addr):
    new = _mod().choose_port("1", addr, str(busy_port), environ={})
    assert new != str(busy_port) and 0 < int(new) < 65536


def test_a_port_busy_only_on_ipv6_is_replaced(busy_v6_port):
    mod = _mod()
    assert mod.is_port_in_use(busy_v6_port)
    assert mod.choose_port("1", "::1", str(busy_v6_port), environ={}) != str(busy_v6_port)


def test_a_free_port_is_kept():
    port = str(_free_port())
    assert _mod().choose_port("1", "127.0.0.1", port, environ={}) == port


@pytest.mark.parametrize(
    "nnodes, addr", [("2", "127.0.0.1"), ("10", "10.0.0.1"), ("1", "10.0.0.1"), ("x", "127.0.0.1")]
)
def test_multi_node_or_remote_runs_keep_the_port(busy_port, nnodes, addr):
    # the other nodes connect to this exact port: it is never changed
    assert _mod().choose_port(nnodes, addr, str(busy_port), environ={}) == str(busy_port)


@pytest.mark.parametrize("port", ["abc", "", "70000", "65536", "0", "-1"])
def test_a_value_that_is_not_a_port_is_kept_for_torchrun_to_report(port):
    assert _mod().choose_port("1", "127.0.0.1", port, environ={}) == port


@pytest.mark.parametrize("keep", ["1", "true", "YES", " on "])
def test_keep_env_keeps_an_explicit_busy_port(busy_port, keep):
    mod = _mod()
    assert mod.choose_port("1", "127.0.0.1", str(busy_port), environ={mod.KEEP_ENV: keep}) == str(busy_port)


@pytest.mark.parametrize("keep", ["0", "false", "", "no"])
def test_other_keep_values_do_not_keep(busy_port, keep):
    mod = _mod()
    assert mod.choose_port("1", "127.0.0.1", str(busy_port), environ={mod.KEEP_ENV: keep}) != str(busy_port)


@pytest.mark.parametrize("addr", ["LOCALHOST", "[::1]", " 127.0.0.1 ", "127.1.2.3", "::1"])
def test_loopback_spellings_are_local(addr):
    assert _mod().is_local(addr)


@pytest.mark.parametrize("addr", ["127.example.com", "127.0.0.1.nip.io", "10.0.0.1", "node01", "::2", "127."])
def test_hostnames_and_other_ips_are_not_local(addr):
    # parsed as an IP: a hostname that starts with "127." is not loopback
    assert not _mod().is_local(addr)


def test_a_non_loopback_hostname_keeps_a_busy_port(busy_port):
    assert _mod().choose_port("1", "127.example.com", str(busy_port)) == str(busy_port)


@pytest.mark.parametrize(
    "nnodes, single",
    [("1", True), ("1:1", True), ("1:2", False), ("2", False), ("2:1", False), ("0:1", False), ("0", False)],
)
def test_elastic_nnodes(nnodes, single):
    assert _mod().is_single_node(nnodes) is single


def test_an_elastic_single_node_run_is_handled(busy_port):
    assert _mod().choose_port("1:1", "localhost", str(busy_port), environ={}) != str(busy_port)


def test_a_bind_error_other_than_in_use_keeps_the_port(monkeypatch):
    import errno

    mod = _mod()

    def denied(family, port):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(mod, "_listen", denied)
    assert not mod.is_port_in_use(80, ipv6=False)
    monkeypatch.setattr(mod, "has_ipv6", lambda: False)
    assert mod.choose_port("1", "127.0.0.1", "80", environ={}) == "80"


def test_a_port_held_on_loopback_only_is_in_use():
    sock = _hold(socket.AF_INET, "127.0.0.1")
    try:
        assert _mod().is_port_in_use(sock.getsockname()[1])
    finally:
        sock.close()


def test_the_cli_prints_the_port(busy_port):
    def run(*argv):
        return subprocess.run(
            [sys.executable, str(HELPER), *argv], capture_output=True, text=True, check=True
        ).stdout.strip()

    out = run("1", "127.0.0.1", str(busy_port))
    assert out.isdigit() and out != str(busy_port)
    assert run("2", "127.0.0.1", str(busy_port)) == str(busy_port)
    assert run("1", "127.0.0.1", "70000") == "70000"  # no traceback: printed unchanged


def test_primus_cli_direct_uses_the_helper_before_torchrun():
    src = DIRECT.read_text()
    use = src.index("helpers/local_master_port.py")
    assert use < src.index('--master_port "${MASTER_PORT:-1234}"')
    assert 'export MASTER_PORT="$port"' in src
    assert '[[ "$DRY_RUN_MODE" != "1" ]] && ! grep -q' in src  # a dry run does not probe sockets


def _direct_block():
    """The MASTER_PORT block of primus-cli-direct.sh (from its comment to the ``fi`` before
    DISTRIBUTED_ARGS), run on its own in bash with stub loggers."""
    src = DIRECT.read_text()
    start = src.index("    # Single node, loopback rendezvous only")
    end = src.index("    DISTRIBUTED_ARGS=(", start)
    return src[start:end]


def _run_block(tmp_path, env, direct_env="", helper=None):
    script = tmp_path / "block.sh"
    script.write_text(
        'LOG_WARN() { echo "WARN $*" >&2; }\n'
        "declare -A direct_config=()\n"
        f"direct_config[env]={shlex.quote(direct_env)}\n"
        "DRY_RUN_MODE=0\n" + _direct_block() + 'echo "PORT=$MASTER_PORT"\n'
    )
    runner = tmp_path / "runner"
    (runner / "helpers").mkdir(parents=True, exist_ok=True)
    (runner / "helpers/local_master_port.py").write_text(helper if helper is not None else HELPER.read_text())
    full = {"PATH": os.environ["PATH"], "RUNNER_DIR": str(runner), **env}
    p = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=full)
    return p.stdout.strip().splitlines()[-1], p.stderr


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash + python3"
)
def test_direct_block_replaces_a_busy_port_and_keeps_a_pinned_one(tmp_path, busy_port):
    env = {"NNODES": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(busy_port)}
    out, err = _run_block(tmp_path, env)
    assert out != f"PORT={busy_port}" and "is in use on this host" in err
    # pinned with --env / direct.env (final precedence): kept, not probed
    out, err = _run_block(tmp_path, env, direct_env=f"NCCL_DEBUG=INFO\nMASTER_PORT={busy_port}")
    assert out == f"PORT={busy_port}" and "WARN" not in err


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash + python3"
)
def test_direct_block_logs_a_helper_failure_and_keeps_the_port(tmp_path):
    env = {"NNODES": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511"}
    out, err = _run_block(tmp_path, env, helper="raise SystemExit('boom')\n")
    assert out == "PORT=29511"
    assert "MASTER_PORT check failed, keeping 29511: boom" in err


def test_choose_port_probes_ipv6_once(monkeypatch, busy_port):
    mod = _mod()
    calls = []
    real = mod.has_ipv6
    monkeypatch.setattr(mod, "has_ipv6", lambda: calls.append(1) or real())
    assert mod.choose_port("1", "127.0.0.1", str(busy_port), environ={}) != str(busy_port)
    assert len(calls) == 1
