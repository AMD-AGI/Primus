###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import errno
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "runner" / "helpers" / "hooks" / "02_setup_nic_userspace_driver.sh"

_spec = importlib.util.spec_from_file_location(
    "nic_userspace_driver", ROOT / "runner" / "helpers" / "nic_userspace_driver.py"
)
nud = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = nud  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(nud)

HOST_VER = "237.1.137.0"


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _tar(path: Path, members: dict, mode: str = "w:gz") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _fake_sysfs(root: Path, devices: dict, bnxt_version: str = HOST_VER) -> Path:
    for i, (name, (vendor, fw)) in enumerate(devices.items()):
        _write(root / "class" / "infiniband" / name / "device" / "vendor", f"{vendor}\n".encode())
        _write(root / "class" / "infiniband" / name / "fw_ver", f"{fw}\n".encode())
        _write(root / "class" / "infiniband_verbs" / f"uverbs{i}" / "ibdev", f"{name}\n".encode())
    if bnxt_version:
        _write(root / "module" / "bnxt_re" / "version", f"{bnxt_version}\n".encode())
    return root


def _ctx(tmp_path, version=HOST_VER, devices=(), **settings):
    layout = nud.ProviderLayout("-rdmav34.so", tmp_path / "prov", tmp_path / "lib", tmp_path / "etc")
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return nud.Context(nud.Settings(**settings), layout, scratch, version, list(devices))


# ---------------------------------------------------------------------------
# Settings and detection
# ---------------------------------------------------------------------------


def test_settings_defaults_and_legacy_switches():
    s = nud.Settings.from_env({})
    assert s.mode == "auto" and not s.strict and not s.forced_providers
    assert [str(p) for p in s.search_path] == nud.DEFAULT_SEARCH_PATH.split(":")
    assert s.ainic_repo_url == nud.DEFAULT_AINIC_REPO_URL

    s = nud.Settings.from_env(
        {
            "PRIMUS_NIC_USERSPACE_DRIVER": "force",
            "PRIMUS_NIC_DRIVER_SEARCH_PATH": "/a::/b",
            "PRIMUS_NIC_DRIVER_STRICT": "1",
            "PRIMUS_AINIC_REPO_URL": "none",
            "REBUILD_BNXT": "1",
            "PATH_TO_BNXT_TAR_PACKAGE": "/x/libbnxt_re-1.2.3.tar.gz",
        }
    )
    assert s.mode == "force" and s.strict
    assert s.search_path == [Path("/a"), Path("/b")]
    assert s.ainic_repo_url == ""
    assert s.forced_providers == {"bnxt_re"}
    assert s.bnxt_package == Path("/x/libbnxt_re-1.2.3.tar.gz")
    assert nud.Settings.from_env({"PRIMUS_NIC_USERSPACE_DRIVER": "off"}).mode == "off"
    assert nud.Settings.from_env({"REBUILD_BNXT": "0"}).forced_providers == frozenset()
    override = nud.Settings.from_env({"PRIMUS_AINIC_BUNDLE_VERSION": "1.117.5-a-147"})
    assert override.version_overrides == {"ionic": "1.117.5-a-147"}
    assert nud.Settings.from_env({}).version_overrides == {}


def test_devices_are_grouped_by_pci_vendor(tmp_path):
    sysfs = _fake_sysfs(
        tmp_path,
        {
            "rdma0": ("0x14e4", "237.1.148.0"),
            "ionic_0": ("0x1dd8", "1.117.5-a-77"),
            "ionic_1": ("0x1dd8", "1.117.5-a-77"),
            "mlx5_0": ("0x15b3", "28.39.1002"),
        },
    )
    devices = nud.list_rdma_devices(sysfs)
    assert {d.name: d.uverbs for d in devices} == {
        "rdma0": "uverbs0",
        "ionic_0": "uverbs1",
        "ionic_1": "uverbs2",
        "mlx5_0": "uverbs3",
    }
    bnxt, ionic = nud.FAMILIES
    assert [d.name for d in devices if bnxt.matches(d)] == ["rdma0"]
    assert [d.name for d in devices if ionic.matches(d)] == ["ionic_0", "ionic_1"]
    assert bnxt.host_version(devices, sysfs) == HOST_VER
    assert ionic.host_version([d for d in devices if ionic.matches(d)], sysfs) == "1.117.5-a-77"
    assert nud.list_rdma_devices(tmp_path / "missing") == []


def test_assess_flags_listing_open_and_queue_pair_failures():
    devs = [nud.RdmaDevice("rdma0", "0x14e4", ""), nud.RdmaDevice("rdma1", "0x14e4", "")]
    warning = "Driver bnxt_re does not support the kernel ABI of 8 (supports 1 to 1) for device /sys/class/infiniband/rdma0"
    broken = nud.ProbeResult(abi_mismatch={"rdma0": warning})
    assert "0 of 2" in nud.assess(devs, broken)
    assert warning in nud.assess(devs, broken)

    both = ["rdma0", "rdma1"]
    assert "cannot open rdma1" in nud.assess(devs, nud.ProbeResult(enumerated=both, opened=["rdma0"]))

    abi_qp = nud.ProbeResult(
        enumerated=both, opened=both, qp_failures={"rdma1": ["ibv_create_qp", errno.EINVAL]}
    )
    assert "ibv_create_qp fails on rdma1" in nud.assess(devs, abi_qp)

    # Locked-memory limits are an environment problem, not a reason to replace the provider.
    memlock = nud.ProbeResult(
        enumerated=both, opened=both, qp_failures={"rdma0": ["ibv_create_cq", errno.EPERM]}
    )
    assert nud.assess(devs, memlock) is None
    assert list(memlock.limit_failures(both)) == ["rdma0"]

    assert nud.assess(devs, nud.ProbeResult(enumerated=both, opened=both)) is None


def test_device_access_is_judged_per_character_device(tmp_path, monkeypatch):
    monkeypatch.setattr(nud, "DEV_INFINIBAND", tmp_path)
    _write(tmp_path / "uverbs0")
    assert nud.device_accessible(nud.RdmaDevice("rdma0", "0x14e4", "", "uverbs0"))
    assert not nud.device_accessible(nud.RdmaDevice("rdma1", "0x14e4", "", "uverbs1"))
    assert not nud.device_accessible(nud.RdmaDevice("rdma2", "0x14e4", "", ""))


def test_provider_layout_is_read_from_libibverbs(tmp_path):
    lib = _write(
        tmp_path / "libibverbs.so.1",
        b"\x7fELF\x00junk\x00/etc/libibverbs.d\x00/usr/lib/x86_64-linux-gnu/libibverbs/lib%s-rdmav34.so\x00"
        b"lib%s-rdmav34.so\x00",
    )
    layout = nud.read_provider_layout(str(lib))
    assert layout.suffix == "-rdmav34.so"
    assert layout.provider_dir == Path("/usr/lib/x86_64-linux-gnu/libibverbs")
    assert layout.config_dir == Path("/etc/libibverbs.d")
    assert layout.lib_dir == tmp_path

    linker_only = nud.read_provider_layout(str(_write(tmp_path / "other.so", b"\x00lib%s-rdmav25.so\x00")))
    assert linker_only.suffix == "-rdmav25.so" and linker_only.provider_dir is None
    with pytest.raises(nud.SetupError):
        nud.read_provider_layout(str(_write(tmp_path / "bogus.so", b"nothing here")))


# ---------------------------------------------------------------------------
# Finding a provider
# ---------------------------------------------------------------------------


def test_bnxt_prefers_exact_source_then_deb_then_nested_then_host_then_closest(tmp_path, monkeypatch):
    bundles = tmp_path / "bundles"
    _write(bundles / "bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/bnxt-rocelib-237.1.137.0-1_all.deb")
    _write(bundles / "bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/libbnxt_re-237.1.137.0.tar.gz")
    _write(bundles / "bcm_233.1.135.7/drivers_linux/bnxt_rocelib/libbnxt_re-233.0.152.2.tar.gz")
    _write(bundles / "old/libbnxt_re-237.1.100.0.tar.gz")
    _tar(
        bundles / "bcm_237.1.150.0.tar.gz",
        {"bcm_237.1.150.0/drivers_linux/bnxt_rocelib/libbnxt_re-237.1.137.0.tar.gz": b"s"},
    )
    _tar(bundles / "drivers/bnxt_en-1.10.3-1.0.0.tar.gz", {"libbnxt_re-237.1.137.0.tar.gz": b"ignored"})
    host = tmp_path / "host"
    _write(host / "bnxt_re" / HOST_VER / "libbnxt_re-rdmav34.so")
    monkeypatch.setattr(nud, "HOST_PROVIDER_ROOT", host)

    ctx = _ctx(tmp_path, devices=[nud.RdmaDevice("rdma0", "0x14e4", "237.1.150.0")], search_path=[bundles])
    got = [(c.kind, c.version, c.member is not None) for c in nud.BnxtFamily().candidates(ctx)]
    assert got == [
        ("source", HOST_VER, False),
        ("deb", HOST_VER, False),
        ("source", HOST_VER, True),  # inside bcm_237.1.150.0.tar.gz (matches the device fw_ver)
        ("prebuilt", HOST_VER, False),
        ("source", "237.1.100.0", False),  # same major release first
        ("source", "233.0.152.2", False),
    ]


def test_bnxt_bundle_archive_yields_source_before_deb(tmp_path, monkeypatch):
    monkeypatch.setattr(nud, "HOST_PROVIDER_ROOT", tmp_path / "none")
    bundle = _tar(
        tmp_path / "bundles" / "bcm_237.1.148.0a.tar.gz",
        {
            "bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/bnxt-rocelib-237.1.137.0-1_all.deb": b"d",
            "bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/libbnxt_re-237.1.137.0.tar.gz": b"s",
        },
    )
    ctx = _ctx(tmp_path, search_path=[bundle.parent])
    cands = list(nud.BnxtFamily().candidates(ctx))
    assert [c.kind for c in cands] == ["source", "deb"]
    assert nud.materialize(cands[0], ctx.scratch).read_bytes() == b"s"


def test_bnxt_explicit_tarball_comes_first_even_with_another_version(tmp_path, monkeypatch):
    monkeypatch.setattr(nud, "HOST_PROVIDER_ROOT", tmp_path / "none")
    explicit = _write(tmp_path / "pkg" / "libbnxt_re-233.0.152.2.tar.gz")
    bundles = tmp_path / "bundles"
    _write(bundles / f"libbnxt_re-{HOST_VER}.tar.gz")
    ctx = _ctx(tmp_path, search_path=[bundles], bnxt_package=explicit)
    cands = list(nud.BnxtFamily().candidates(ctx))
    assert cands[0].explicit and cands[0].path == str(explicit) and cands[0].version == "233.0.152.2"
    assert cands[1].version == HOST_VER


def test_ionic_finds_codename_deb_in_nested_bundle_before_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(nud, "os_codename", lambda: "noble")
    rdma_core = io.BytesIO()
    with tarfile.open(fileobj=rdma_core, mode="w:gz") as tf:  # gzip stream despite the .tar.xz name
        for name in ("jammy/libionic1_54.0-197-1_amd64.deb", "noble/libionic1_54.0-197-1_amd64.deb"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"d"))
    host_sw = io.BytesIO()
    with tarfile.open(fileobj=host_sw, mode="w:xz") as tf:
        for name, data in (
            (
                "host_sw_pkg/ionic_driver/deb/libionic-debs.tar.xz",
                _tar_bytes({"noble/libionic1_50.0_amd64.deb": b"d"}),
            ),
            ("host_sw_pkg/ionic_driver/deb/rdma-core-debs.tar.xz", rdma_core.getvalue()),
            ("host_sw_pkg/ionic_driver/deb/linux-debs.tar.xz", _tar_bytes({"noble/ionic-dkms.deb": b"d"})),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    bundles = tmp_path / "bundles"
    _tar(
        bundles / "ainic_bundle_1.117.5-a-147.tar.gz",
        {"ainic_bundle_1.117.5-a-147/host_sw_pkg.tar.xz": host_sw.getvalue()},
    )
    _tar(bundles / "ainic_bundle_1.117.5-a-77.tar.gz", {"x/host_sw_pkg.tar.xz": b"not scanned"})

    ctx = _ctx(tmp_path, version="1.117.5-a-147", search_path=[bundles])
    cands = list(nud.IonicFamily().candidates(ctx))
    assert [c.kind for c in cands] == ["deb", "deb", "download"]
    assert "rdma-core-debs" in cands[0].origin and cands[0].member == "noble/libionic1_54.0-197-1_amd64.deb"
    assert "libionic-debs" in cands[1].origin
    assert cands[2].path == f"{nud.DEFAULT_AINIC_REPO_URL}/1.117.5-a-147"

    ctx.settings.ainic_repo_url = ""
    assert all(c.kind != "download" for c in nud.IonicFamily().candidates(ctx))


def _tar_bytes(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_parse_packages_index_picks_stanzas():
    text = (
        "Package: libionic-dev\nVersion: 1\nFilename: pool/a.deb\n\n"
        "Package: libionic1\nVersion: 54.0-197-1\nFilename: pool/main/r/rdma-core/libionic1.deb\n"
        "Description: x\n multi-line\n"
    )
    stanzas = nud.parse_packages_index(text)
    assert [s["Package"] for s in stanzas] == ["libionic-dev", "libionic1"]
    assert stanzas[1]["Filename"] == "pool/main/r/rdma-core/libionic1.deb"


def test_elf_soname_reads_dt_soname():
    libc = next(
        (line.split()[-1] for line in open("/proc/self/maps") if "/libc.so" in line or "/libc-" in line), None
    )
    if libc is None:
        pytest.skip("no glibc mapped")
    assert nud.elf_soname(Path(libc)) == "libc.so.6"


def test_provider_in_tree_resolves_package_symlink(tmp_path):
    lib = tmp_path / "usr/lib/x86_64-linux-gnu"
    _write(lib / "libionic.so.1.1.50")
    (lib / "libibverbs").mkdir()
    os.symlink("../libionic.so.1.1.50", lib / "libibverbs" / "libionic-rdmav34.so")
    assert nud.provider_in_tree(tmp_path, "ionic", "-rdmav34.so") == (lib / "libionic.so.1.1.50").resolve()
    with pytest.raises(nud.SetupError, match="loads libionic-rdmav25.so"):
        nud.provider_in_tree(tmp_path, "ionic", "-rdmav25.so")


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------


def _snapshot(*dirs):
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            out[str(p)] = (
                ("->" + os.readlink(p)) if p.is_symlink() else (p.read_bytes() if p.is_file() else None)
            )
    return out


def test_system_install_ionic_retires_old_lib_and_rolls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(nud, "_ldconfig", lambda: None)
    lib, prov, etc = tmp_path / "lib", tmp_path / "lib" / "libibverbs", tmp_path / "etc"
    _write(lib / "libionic.so.1.1.54.0-187", b"old")
    os.symlink("libionic.so.1.1.54.0-187", lib / "libionic.so.1")
    prov.mkdir()
    os.symlink("../libionic.so.1.1.54.0-187", prov / "libionic-rdmav34.so")
    _write(etc / "ionic.driver", b"driver ionic\n")
    before = _snapshot(lib, etc)

    layout = nud.ProviderLayout("-rdmav34.so", prov, lib, etc)
    staged = nud.Staged(
        "1.117.5-a-196", _write(tmp_path / "stage" / "libionic.so.1.1.50", b"new"), ["libionic.so.1"]
    )
    installer = nud.SystemInstaller(layout, tmp_path / "backup")
    inst = installer.install(nud.IonicFamily(), staged)

    assert inst.env == {} and inst.provider_path == prov / "libionic-rdmav34.so"
    assert inst.sonames == {"libionic.so.1": lib / "libionic.so.1.1.50"}
    assert (lib / "libionic.so.1").resolve() == (lib / "libionic.so.1.1.50").resolve()
    assert (prov / "libionic-rdmav34.so").resolve() == (lib / "libionic.so.1.1.50").resolve()
    assert not (lib / "libionic.so.1.1.54.0-187").exists()  # ldconfig would prefer it over 1.1.50

    installer.rollback(inst)
    assert _snapshot(lib, etc) == before

    installer.commit(installer.install(nud.IonicFamily(), staged))
    assert list((tmp_path / "backup").iterdir()) == []


def test_system_install_bnxt_replaces_provider_and_fixes_config(tmp_path):
    lib, prov, etc = tmp_path / "lib", tmp_path / "lib" / "libibverbs", tmp_path / "etc"
    _write(prov / "libbnxt_re-rdmav34.so", b"inbox")
    _write(etc / "bnxt_re.driver", b"driver /somewhere/else/libbnxt_re\n")
    before = _snapshot(lib, etc)
    installer = nud.SystemInstaller(nud.ProviderLayout("-rdmav34.so", prov, lib, etc), tmp_path / "backup")
    inst = installer.install(
        nud.BnxtFamily(), nud.Staged(HOST_VER, _write(tmp_path / "b" / "libbnxt_re-rdmav34.so", b"new"))
    )
    assert (prov / "libbnxt_re-rdmav34.so").read_bytes() == b"new"
    assert (etc / "bnxt_re.driver").read_text() == "driver bnxt_re\n"
    installer.rollback(inst)
    assert _snapshot(lib, etc) == before


def test_user_install_selects_provider_through_env(tmp_path):
    layout = nud.ProviderLayout("-rdmav34.so", None, tmp_path / "lib", tmp_path / "etc")
    real = _write(tmp_path / "stage" / "libionic.so.1.1.50", b"new")
    installer = nud.UserInstaller(layout, tmp_path / "prefixes")
    inst = installer.install(nud.IonicFamily(), nud.Staged("1.117.5-a-196", real, ["libionic.so.1"]))
    prefix = inst.provider_path.parent
    assert inst.env == {"RDMAV_DRIVERS": str(prefix / "libionic"), "LD_LIBRARY_PATH": str(prefix)}
    assert sorted(os.listdir(prefix)) == ["libionic-rdmav34.so", "libionic.so.1", "libionic.so.1.1.50"]
    assert (
        installer.install(nud.IonicFamily(), nud.Staged("1.117.5-a-196", real, ["libionic.so.1"])).env
        == inst.env
    )

    bnxt = installer.install(
        nud.BnxtFamily(), nud.Staged(HOST_VER, _write(tmp_path / "b" / "libbnxt_re-rdmav34.so"))
    )
    assert set(bnxt.env) == {"RDMAV_DRIVERS"}

    merged = {}
    nud.merge_env(merged, inst.env)
    nud.merge_env(merged, bnxt.env)
    assert merged["RDMAV_DRIVERS"] == f"{bnxt.env['RDMAV_DRIVERS']}:{inst.env['RDMAV_DRIVERS']}"


def test_user_install_never_reuses_a_directory_others_can_write(tmp_path):
    layout = nud.ProviderLayout("-rdmav34.so", None, tmp_path / "lib", tmp_path / "etc")
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    real = _write(tmp_path / "b" / "libbnxt_re-rdmav34.so", b"good")
    inst = nud.UserInstaller(layout, shared).install(nud.BnxtFamily(), nud.Staged(HOST_VER, real))
    assert shared not in inst.provider_path.parents
    assert inst.provider_path.read_bytes() == b"good"


def test_user_install_replaces_a_tampered_prefix(tmp_path):
    layout = nud.ProviderLayout("-rdmav34.so", None, tmp_path / "lib", tmp_path / "etc")
    real = _write(tmp_path / "b" / "libbnxt_re-rdmav34.so", b"good")
    installer = nud.UserInstaller(layout, tmp_path / "prefixes")
    first = installer.install(nud.BnxtFamily(), nud.Staged(HOST_VER, real))
    os.chmod(first.provider_path, 0o644)
    first.provider_path.write_bytes(b"evil")
    second = installer.install(nud.BnxtFamily(), nud.Staged(HOST_VER, real))
    assert second.provider_path == first.provider_path and second.provider_path.read_bytes() == b"good"


def test_verify_rejects_a_second_copy_of_the_shared_library(tmp_path, monkeypatch):
    real = _write(tmp_path / "lib" / "libionic.so.1.1.50")
    stray = _write(tmp_path / "usr-local" / "libionic.so.1.1.54")
    devs = [nud.RdmaDevice("ionic_0", "0x1dd8", "1.117.5-a-196", "uverbs0")]
    inst = nud.Installed(provider_path=real, sonames={"libionic.so.1": real})
    seen_env = {}

    def probe(extra_env=None, trace=False):
        seen_env.update(extra_env or {})
        return nud.ProbeResult(
            enumerated=["ionic_0"],
            opened=["ionic_0"],
            loaded=[str(real)],
            sonames={"libionic.so.1": str(stray)},
        )

    monkeypatch.setattr(nud, "run_probe", probe)
    assert "two copies" in nud.verify(devs, inst)
    assert seen_env["PRIMUS_NIC_PROBE_SONAMES"] == "libionic.so.1"

    monkeypatch.setattr(
        nud,
        "run_probe",
        lambda extra_env=None, trace=False: nud.ProbeResult(
            enumerated=["ionic_0"],
            opened=["ionic_0"],
            loaded=[str(real)],
            sonames={"libionic.so.1": str(real)},
        ),
    )
    assert nud.verify(devs, inst) is None


def test_unreachable_repository_fails_after_one_attempt(tmp_path, monkeypatch):
    calls = []

    def unreachable(url, timeout=None):
        calls.append(url)
        raise nud.urllib.error.URLError("timed out")

    monkeypatch.setattr(nud.urllib.request, "urlopen", unreachable)
    monkeypatch.setattr(nud, "os_codename", lambda: "noble")
    ctx = _ctx(tmp_path, version="1.117.5-a-196")
    cand = nud.Candidate("1.117.5-a-196", "download", "https://repo.invalid/x/1.117.5-a-196", "repo")
    with pytest.raises(nud.SetupError, match="cannot fetch the package index"):
        nud.IonicFamily()._download(cand, ctx)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# setup() orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def broadcom_node(tmp_path, monkeypatch):
    sysfs = _fake_sysfs(
        tmp_path / "sys", {"rdma0": ("0x14e4", "237.1.148.0"), "rdma1": ("0x14e4", "237.1.148.0")}
    )
    layout = nud.ProviderLayout("-rdmav34.so", None, tmp_path / "lib", tmp_path / "etc")
    monkeypatch.setattr(nud, "read_provider_layout", lambda _: layout)
    monkeypatch.setattr(nud, "DEV_INFINIBAND", tmp_path / "dev")
    for uverbs in ("uverbs0", "uverbs1"):
        _write(tmp_path / "dev" / uverbs)
    monkeypatch.setattr(nud, "HOST_PROVIDER_ROOT", tmp_path / "host")
    monkeypatch.delenv("RDMAV_DRIVERS", raising=False)
    monkeypatch.delenv("NNODES", raising=False)
    return sysfs


def _probe(enumerated):
    return lambda *a, **k: nud.ProbeResult(
        enumerated=enumerated, opened=enumerated, libibverbs="/x/libibverbs.so.1"
    )


def test_setup_is_a_noop_when_the_provider_works(broadcom_node, monkeypatch, capsys):
    monkeypatch.setattr(nud, "run_probe", _probe(["rdma0", "rdma1"]))
    monkeypatch.setattr(nud, "choose_installer", lambda _: pytest.fail("must not install"))
    assert nud.setup(nud.Settings(), broadcom_node) == 0
    out = capsys.readouterr().out
    assert "works with all 2 devices" in out and "env." not in out


def test_setup_installs_a_matching_provider_and_exports_env(broadcom_node, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nud, "run_probe", _probe([]))
    _write(tmp_path / "host" / "bnxt_re" / HOST_VER / "libbnxt_re-rdmav34.so", b"host build")
    installer = nud.UserInstaller(nud.read_provider_layout(""), tmp_path / "prefixes")
    monkeypatch.setattr(nud, "choose_installer", lambda _: installer)
    monkeypatch.setattr(nud, "verify", lambda devices, inst: None)

    assert nud.setup(nud.Settings(search_path=[]), broadcom_node) == 0
    env_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("env.")]
    assert len(env_lines) == 1 and env_lines[0].startswith(f"env.RDMAV_DRIVERS={tmp_path / 'prefixes'}")


def test_setup_rolls_back_and_reports_when_nothing_works(broadcom_node, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nud, "run_probe", _probe([]))
    _write(tmp_path / "host" / "bnxt_re" / HOST_VER / "libbnxt_re-rdmav34.so")
    rolled_back = []

    class Recorder(nud.UserInstaller):
        def rollback(self, inst):
            rolled_back.append(inst.provider_path)

    monkeypatch.setattr(nud, "choose_installer", lambda layout: Recorder(layout, tmp_path / "prefixes"))
    monkeypatch.setattr(nud, "verify", lambda devices, inst: "still broken")

    assert nud.setup(nud.Settings(search_path=[]), broadcom_node) == 0
    assert len(rolled_back) == 1
    err = capsys.readouterr().err
    assert "still broken" in err and "falls back to TCP" in err and "mixed transports" not in err
    monkeypatch.setenv("NNODES", "4")
    assert nud.setup(nud.Settings(search_path=[]), broadcom_node) == 0
    assert "mixed transports" in capsys.readouterr().err
    assert nud.setup(nud.Settings(search_path=[], strict=True), broadcom_node) == 1


def test_setup_only_judges_devices_passed_into_the_container(broadcom_node, tmp_path, monkeypatch, capsys):
    os.remove(tmp_path / "dev" / "uverbs1")  # rdma1 was not passed through
    monkeypatch.setattr(nud, "run_probe", _probe(["rdma0"]))  # libibverbs skips rdma1 for that reason
    monkeypatch.setattr(nud, "choose_installer", lambda _: pytest.fail("must not install"))
    assert nud.setup(nud.Settings(), broadcom_node) == 0
    out = capsys.readouterr().out
    assert "rdma1 not passed into this environment" in out and "works with all 1 devices" in out


def test_setup_without_device_access_warns_instead_of_installing(
    broadcom_node, tmp_path, monkeypatch, capsys
):
    for uverbs in ("uverbs0", "uverbs1"):
        os.remove(tmp_path / "dev" / uverbs)
    monkeypatch.setattr(nud, "run_probe", _probe([]))
    monkeypatch.setattr(nud, "choose_installer", lambda _: pytest.fail("must not install"))
    assert nud.setup(nud.Settings(), broadcom_node) == 0
    assert "--device /dev/infiniband" in capsys.readouterr().err


def test_setup_without_a_source_explains_where_to_put_the_bundle(broadcom_node, monkeypatch, capsys):
    monkeypatch.setattr(nud, "run_probe", _probe([]))
    monkeypatch.setattr(nud, "choose_installer", lambda layout: nud.UserInstaller(layout))
    assert nud.setup(nud.Settings(search_path=[]), broadcom_node) == 0
    assert f"drivers_linux/bnxt_rocelib/libbnxt_re-{HOST_VER}.tar.gz" in capsys.readouterr().err


def test_ainic_bundle_override_is_used_when_fw_ver_is_missing(tmp_path, monkeypatch, capsys):
    sysfs = _fake_sysfs(tmp_path / "sys", {"ionic_0": ("0x1dd8", "")}, bnxt_version="")
    _write(tmp_path / "dev" / "uverbs0")
    monkeypatch.setattr(nud, "DEV_INFINIBAND", tmp_path / "dev")
    layout = nud.ProviderLayout("-rdmav34.so", None, tmp_path / "lib", tmp_path / "etc")
    monkeypatch.setattr(nud, "read_provider_layout", lambda _: layout)
    monkeypatch.setattr(nud, "run_probe", _probe([]))
    monkeypatch.setattr(nud, "choose_installer", lambda layout: nud.UserInstaller(layout, tmp_path / "p"))
    asked = []
    monkeypatch.setattr(
        nud.IonicFamily, "candidates", lambda self, ctx: asked.append(ctx.host_version) or iter(())
    )

    nud.setup(nud.Settings(search_path=[]), sysfs)
    assert asked == [] and "PRIMUS_AINIC_BUNDLE_VERSION" in capsys.readouterr().err

    nud.setup(nud.Settings(search_path=[], version_overrides={"ionic": "1.117.5-a-147"}), sysfs)
    assert asked == ["1.117.5-a-147"]


def test_hook_off_switch_is_a_silent_noop():
    env = dict(os.environ, PRIMUS_NIC_USERSPACE_DRIVER="off")
    result = subprocess.run(["bash", str(HOOK)], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""


@pytest.mark.skipif(not list(Path("/sys/class/infiniband").glob("*")), reason="needs RDMA devices in /sys")
def test_hook_crash_does_not_block_the_launch_unless_strict(tmp_path):
    fake = _write(tmp_path / "python3", b'#!/bin/bash\n[[ "$1" == "-c" ]] && exit 0\nexit 3\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    env.pop("PRIMUS_NIC_USERSPACE_DRIVER", None)
    env.pop("PRIMUS_NIC_DRIVER_STRICT", None)
    result = subprocess.run(["bash", str(HOOK)], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and "[WARN]" in result.stderr
    strict = subprocess.run(
        ["bash", str(HOOK)], env=dict(env, PRIMUS_NIC_DRIVER_STRICT="1"), capture_output=True
    )
    assert strict.returncode == 3


def test_status_command_runs(capsys):
    assert nud.main(["status"]) == 0
    assert "mode:" in capsys.readouterr().out
