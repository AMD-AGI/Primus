# Multi-node networking guide

Multi-node training depends on **high-bandwidth, low-latency** communication between GPUs. On AMD systems, **RCCL** (ROCm Collective Communications Library) provides GPU collectives with an API aligned to **NCCL**, so most **NCCL-prefixed** environment variables apply to RCCL as well.

This guide summarizes how Primus configures networking, how **InfiniBand**, **RoCE**, and **AINIC (AMD AI NIC)** fit in, and how to validate and troubleshoot cluster connectivity.

**Primary sources in this repository**

| Topic | File |
|-------|------|
| Default NCCL/RCCL and socket setup | `runner/helpers/envs/base_env.sh` |
| IB HCA detection | `runner/helpers/envs/get_nccl_ib_hca.sh` |
| Socket / interface detection | `runner/helpers/envs/get_ip_interface.sh` |
| AINIC hook (container/CLI integration) | `runner/helpers/hooks/03_enable_ainic.sh` |
| AINIC CLI defaults | `runner/use_ainic.yaml` |
| ANP / `NCCL_NET_PLUGIN` selection | `runner/helpers/hooks/03_enable_ainic.sh` |
| RDMA userspace driver matching | `runner/helpers/hooks/02_setup_nic_userspace_driver.sh`, `runner/helpers/nic_userspace_driver.py` |

---

## 1. Overview

- **Goal:** Keep gradient and parameter exchanges from becoming the bottleneck when scaling across nodes.
- **Stack:** PyTorch distributed uses the ROCm **NCCL** backend name in many configs; the implementation is **RCCL** on AMD GPUs.
- **Transports:** Common fabrics include **InfiniBand (IB)**, **RoCE** (RDMA over Converged Ethernet), and **AINIC** on supported AMD platforms. Primus scripts set or detect **HCAs**, **socket interfaces**, and optional **AINIC** tuning.

---

## 2. InfiniBand configuration

These variables are standard in NCCL/RCCL deployments. Primus seeds several from `runner/helpers/envs/base_env.sh` when that script is sourced.

| Variable | Role |
|----------|------|
| `NCCL_IB_HCA` | Selects **InfiniBand Host Channel Adapters** (device:port list). |
| `NCCL_IB_GID_INDEX` | **GID index** for the active port (RoCE and IB differ; see vendor docs). |
| `NCCL_IB_TC` | **Traffic class** for InfiniBand. |
| `NCCL_IB_FIFO_TC` | Traffic class for FIFO traffic. |
| `NCCL_IB_RETRY_CNT` | Retry count for IB operations (tune with vendor guidance). |
| `NCCL_IB_TIMEOUT` | Timeout for IB operations. |
| `NCCL_IB_QPS_PER_CONNECTION` | Queue pairs per connection. |
| `NCCL_NET_GDR_LEVEL` | **GPUDirect RDMA** level for NIC/GPU transfers. |
| `NCCL_DMABUF_ENABLE` | Use **DMA-BUF** path where supported. |

### Auto-detection in Primus

If `NCCL_IB_HCA` is **unset**, `base_env.sh` runs `runner/helpers/envs/get_nccl_ib_hca.sh`, which enumerates `/sys/class/infiniband/`, skips bonded/storage-style devices, and builds a comma-separated `device:port` list for `NCCL_IB_HCA`.

Default in `base_env.sh`:

```bash
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
```

AINIC-oriented configs often override `NCCL_IB_GID_INDEX` to `1` (see `runner/use_ainic.yaml` and `03_enable_ainic.sh`).

---

## 3. RoCE (RDMA over converged ethernet)

RoCE reuses much of the **IB verb** stack; the same **`NCCL_IB_*`** knobs apply.

| Variable | Typical use |
|----------|-------------|
| `NCCL_IB_ROCE_VERSION_NUM` | RoCE version (commonly **2** for RoCE v2). |

GID selection (`NCCL_IB_GID_INDEX`) and traffic classes (`NCCL_IB_TC`, `NCCL_IB_FIFO_TC`) remain important on RoCE fabrics. Follow your network team’s mapping (often **GID index 1** for RoCE v2 vs **3** for some IB fabrics—your site might differ).

---

## 4. AINIC (AMD AI NIC)

**AINIC** refers to AMD’s AI-optimized NIC path (for example, the **AMD Pensando™ Pollara 400 AI NIC**) used in some clusters. Enabling it is a combination of **environment**, **container image**, and **device pass-through**.

### Enable AINIC

- Set **`USING_AINIC=1`**. The hook `runner/helpers/hooks/03_enable_ainic.sh` runs when this is set and exports AINIC-related variables back to the caller (`env.VAR=VALUE` lines).
- Use container images built for AINIC when required by your site. Examples in this repository use tags such as `docker.io/tasimage/primus:<version>-ainic` (see `examples/customer_package/` and `.github/workflows/ci.yaml`). Match the image to your ROCm and ANP bundle.
- If the AINIC bundle in a published image does not match your host driver, see [AINIC bundle versions](./ainic-bundle-versions.md) for how to rebuild the image against a different bundle. Note that the bundle named in an image's build arguments is **not** always the one installed; verify with `dpkg-query -W libionic1`.

### `runner/use_ainic.yaml`

Primus CLI system defaults for AINIC-oriented runs include:

- Container **`device`** mounts: `/dev/kfd`, `/dev/dri`, `/dev/infiniband` (required for GPU and IB access in the container).
- Environment entries such as `USING_AINIC=1`, `NCCL_PXN_DISABLE=0`, and `NCCL_IB_GID_INDEX=1`.

Adjust **`NCCL_IB_GID_INDEX`** and **`container.options.image`** to match your cluster; comments in `runner/use_ainic.yaml` call this out explicitly.

### AINIC hook

**`runner/helpers/hooks/03_enable_ainic.sh`** is the supported hook path: it sets ANP/RCCL/MPI home directories, IB QoS, RoCE version, P2P channel counts, GDR flush behavior, `LD_LIBRARY_PATH` (including `libibverbs` and RCCL/ANP/MPI build paths), and related flags. Default `NCCL_IB_FIFO_TC` in the hook is **192**; align this value with your fabric.

### RCCL network plugin (ANP)

For ANP-based networking, `NCCL_NET_PLUGIN` is set to **`librccl-anp.so`** when that library is present under `ANP_HOME_DIR`, falling back to `librccl-net.so` otherwise. This selection and the matching library paths both live in `runner/helpers/hooks/03_enable_ainic.sh`, so they apply to every launcher mode.

### Variables commonly set for AINIC

From `03_enable_ainic.sh` (non-exhaustive):

| Variable | Purpose |
|----------|---------|
| `ANP_HOME_DIR`, `RCCL_HOME_DIR`, `MPI_HOME_DIR` | Install roots for ANP, RCCL, and Open MPI. |
| `NCCL_IB_TC`, `NCCL_IB_FIFO_TC` | Traffic classes for IB/RoCE. |
| `NCCL_IB_GID_INDEX` | Often **1** for AINIC-oriented configs in Primus examples. |
| `NCCL_IB_ROCE_VERSION_NUM` | RoCE v2. |
| `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING` | Stricter GDR flush ordering (set to `0` in these scripts). |
| `LD_LIBRARY_PATH` | Prepends `libibverbs`, RCCL, ANP, and MPI library paths. |

---

## 5. RDMA userspace driver (libibverbs provider)

An RDMA NIC driver has two halves: the kernel driver on the host, and the userspace provider that libibverbs loads inside the container — `libbnxt_re` for Broadcom, `libionic` for AMD Pensando AINIC. Training images ship a fixed provider, so on a cluster that runs a different driver release the provider can reject every device:

```text
libibverbs: Warning: Driver bnxt_re does not support the kernel ABI of 8 (supports 1 to 1) for device /sys/class/infiniband/rdma0
NCCL INFO NET/IB : No device found.
NCCL INFO Using network Socket
```

RCCL then runs over TCP sockets. Training still completes, just much slower: on a node with eight Broadcom 400G NICs, a 256 MB all-reduce forced over the network ran at 3.3 GB/s this way, against 33 GB/s with a matching provider.

The system hook `runner/helpers/hooks/02_setup_nic_userspace_driver.sh` fixes this before every launch, in every launcher mode. In the common case it needs no configuration.

### What the hook does

1. Finds RDMA devices by PCI vendor and checks whether the current provider can list and open them and create a queue pair on each. If it can, the hook does nothing. Only devices whose `/dev/infiniband/uverbs*` character device is available are checked.
2. Reads the host driver version: the `bnxt_re` module version (`/sys/module/bnxt_re/version`), or, for AINIC, the bundle version that the ionic devices report in `fw_ver`.
3. Looks for a provider of that version:

   | NIC | Sources, in order |
   |-----|-------------------|
   | Broadcom | `PATH_TO_BNXT_TAR_PACKAGE`; `libbnxt_re-<version>.tar.gz`, then `bnxt-rocelib-<version>*.deb`, under `PRIMUS_NIC_DRIVER_SEARCH_PATH`, as plain files (an extracted `bcm_<release>` bundle) or inside bundle archives; the host's own `libbnxt_re` of that version. Without an exact match, the closest release found is tried. |
   | AINIC | `libionic1_*.deb` for the container's distribution, inside a bundle named after the version under `PRIMUS_NIC_DRIVER_SEARCH_PATH`; otherwise the package from `https://repo.radeon.com/amdainic/pensando/ubuntu/<bundle>/`, checked against the repository's SHA256. |

4. Builds the provider (Broadcom source tarball, about 10 seconds) or unpacks it (`.deb`), installs it, and probes again: every device must be listed and opened and accept a queue pair, libibverbs must have loaded the new file, and for AINIC `libionic.so.1` must resolve to that same file. A provider that fails is rolled back and the next source is tried.

If no working provider is found, the job continues exactly as it would have without the hook: RCCL uses TCP sockets on that node. The hook's warnings are printed on every node, not only node 0. On a multi-node job this deserves attention: nodes that did install a provider use RDMA, and RCCL can hang when nodes use different transports. Set `PRIMUS_NIC_DRIVER_STRICT=1` to stop the launch on such a node instead. See [When RDMA still does not work](#when-rdma-still-does-not-work) for what each warning means.

Where the provider is installed depends on where the hook runs:

- **As root inside a container** (the `primus-cli container` and `primus-cli slurm` default): over the image's provider, in the system library paths. Only the container is changed, and it is discarded after the run.
- **Anywhere else** (non-root containers, bare metal): into a private, user-owned prefix under `/tmp`, selected for this run through `RDMAV_DRIVERS` (plus `LD_LIBRARY_PATH` for AINIC). System files are never modified. In this mode the image's own Broadcom provider still prints the `does not support the kernel ABI` warning before the new provider claims the device; the warning is harmless.

The hook never fails a launch on its own (unless `PRIMUS_NIC_DRIVER_STRICT=1`): if it cannot run or crashes, it prints a warning and the launch proceeds.

### Making driver sources available

`primus-cli container`, and therefore `primus-cli slurm` on every node, mounts these into the container read-only:

- every existing entry of `PRIMUS_NIC_DRIVER_SEARCH_PATH` (default `/opt/broadcom:/opt/bnxt-bundles:/opt/ainic-bundles`), at the same path;
- `PATH_TO_BNXT_TAR_PACKAGE`, at the same path;
- the host's `libbnxt_re-rdmav*.so`, when the same directory has `libbnxt_re-<version>.so` for the loaded `bnxt_re` module, which is what Broadcom's installer leaves in `/usr/local/lib`.

On a Broadcom cluster it is therefore enough to keep the NetXtreme-E Linux bundle that was used for the host install on every node, extracted or not — for example `/opt/broadcom/bcm_237.1.148.0a.tar.gz` — or to have installed `libbnxt_re` on the hosts from that bundle. Broadcom's downloads require accepting a license on Broadcom's site, so the hook never downloads `libbnxt_re` itself. AINIC clusters with access to `repo.radeon.com` need nothing.

If you start the container yourself and run `primus-cli direct` inside it, mount the bundle directory and `/dev/infiniband` yourself, and set `PRIMUS_NIC_DRIVER_SEARCH_PATH` if the bundle is not under one of the default paths.

### Controls

| Variable | Default | Effect |
|----------|---------|--------|
| `PRIMUS_NIC_USERSPACE_DRIVER` | `auto` | `auto` installs a provider only when the current one cannot use the NICs. `force` always installs the version matching the host, which also catches an AINIC version skew that still enumerates devices. `off` disables the hook. |
| `PRIMUS_NIC_DRIVER_SEARCH_PATH` | `/opt/broadcom:/opt/bnxt-bundles:/opt/ainic-bundles` | Colon-separated directories or bundle files to search. |
| `PRIMUS_NIC_DRIVER_STRICT` | `0` | `1` aborts the launch when RDMA is still unusable, instead of continuing over TCP sockets. |
| `PRIMUS_AINIC_REPO_URL` | `https://repo.radeon.com/amdainic/pensando/ubuntu` | AINIC package repository, for example a mirror on air-gapped sites. `none` disables downloads. |
| `PRIMUS_AINIC_BUNDLE_VERSION` | unset | AINIC bundle to install (for example `1.117.5-a-147`) when the host's `fw_ver` does not name it. |
| `REBUILD_BNXT`, `PATH_TO_BNXT_TAR_PACKAGE` | unset | Still honored but no longer needed: `REBUILD_BNXT=1` forces a Broadcom reinstall, and the given tarball is tried first. |

`PRIMUS_*` variables are forwarded into the container automatically.

### Checking that it worked

The hook's lines are tagged `[nic-driver]` and appear near the top of the launch log, before training starts. Node 0 prints all of them; every other node prints its warnings. Either of these lines means the provider is good:

```text
[nic-driver] Broadcom bnxt_re: the bnxt_re provider in this environment works with all 8 devices
[nic-driver] Broadcom bnxt_re: installed bnxt_re 237.1.137.0 from /opt/broadcom/bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/libbnxt_re-237.1.137.0.tar.gz (system install, 8 devices verified)
```

To inspect a node without changing anything, run the `status` command in the same environment the job uses. For a container, start it with the same image, devices and mounts:

```bash
docker run --rm --privileged --network host --device /dev/infiniband \
  -v "$PWD:$PWD" -w "$PWD" -v /opt/broadcom:/opt/broadcom:ro \
  rocm/primus:v26.7 python3 runner/helpers/nic_userspace_driver.py status
```

```text
Broadcom bnxt_re: 8 devices, host driver 237.1.137.0
  rdma0: fw 237.1.148.0, /dev/infiniband/uverbs1
  ...
  provider loaded from: /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so
  status: libibverbs lists 0 of 8 devices ("Driver bnxt_re does not support the kernel ABI of 8 (supports 1 to 1) for device /sys/class/infiniband/rdma0")
```

`status: OK` means the provider works; anything else is what the hook would fix. Finally, confirm that RCCL uses the NICs: run with `NCCL_DEBUG=INFO` and check for `Using network IB` and the absence of `NET/Socket`, as described in [AINIC bundle versions](./ainic-bundle-versions.md#confirm-the-fabric-is-actually-used).

### When RDMA still does not work

If the hook cannot install a working provider, it prints `RDMA is unusable on this node` with the reason in the `[nic-driver]` lines just above, and the job continues over TCP sockets on that node. Find the reason below.

| Message in the `[nic-driver]` lines | Cause | What to do |
|-------------------------------------|-------|------------|
| `no libbnxt_re <version> found (searched ...)` | No Broadcom source of the host's driver version is visible. | Put the NetXtreme-E Linux bundle used for the host install — the one that contains `drivers_linux/bnxt_rocelib/libbnxt_re-<version>.tar.gz`, where `<version>` is `cat /sys/module/bnxt_re/version` on the host — under `/opt/broadcom` on every node. Alternatively set `PRIMUS_NIC_DRIVER_SEARCH_PATH` to where it is, or `PATH_TO_BNXT_TAR_PACKAGE` to the tarball. Your cluster administrator has this bundle. |
| `trying bnxt_re <X> (host runs <Y>)` | Only a different Broadcom release was found. It is kept only if it passes every check. | Usually fine. For a supported setup, provide the bundle that matches the host driver. |
| `cannot build libbnxt_re here, missing ...` | The image lacks the build tools (autotools, a C compiler, `libibverbs-dev`). The Primus images have them. | Use a Primus image, or make the `bnxt-rocelib` `.deb` from the same bundle available, or install `libbnxt_re` on the hosts from the bundle. |
| `` `make` failed with exit code N (full log: /tmp/primus-libbnxt_re-….log) `` | The source tarball does not build in this image. | Read the log. Use the `.deb` from the bundle or a host-installed `libbnxt_re` instead. |
| `package provides lib…-rdmavN.so but this libibverbs loads …` or `does not match this libibverbs` | A prebuilt provider was built against a different rdma-core than the image's. | Provide the source tarball, which is built against the image's own rdma-core. |
| `<source> did not work (<reason>); rolled back` | A provider installed but failed a check, and the image's provider was restored. The next source is tried automatically. | If every source fails, the reasons say why. `kernel ABI` means that release cannot drive the host driver; provide the matching bundle. |
| `N devices are in /sys but none of their /dev/infiniband/uverbs* character devices is accessible here` | The container was started without the RDMA devices or, on bare metal, your user cannot open them. | Add `--device /dev/infiniband` (`primus-cli container` does by default). On bare metal, ask your administrator for read/write access to `/dev/infiniband/uverbs*`. |
| `<devices> not passed into this environment; checking the other N` | Only some RDMA devices were passed into the container. | Informational. Pass all of `/dev/infiniband` to use every NIC. |
| `ibv_create_cq fails on <device>: Operation not permitted; RCCL needs locked memory` | The locked-memory limit is too low for RDMA. The provider itself is fine. | Run the container with `--ulimit memlock=-1` or `--privileged` (the Primus default). On bare metal, raise `ulimit -l`. |
| `the ionic devices do not report fw_ver ...` | The AINIC bundle version cannot be read from the host. | Set `PRIMUS_AINIC_BUNDLE_VERSION` to the bundle installed on the hosts. |
| `cannot fetch the package index: ...` | `repo.radeon.com` is unreachable (air-gapped site, proxy), or `fw_ver` is not a published bundle name. | Set `https_proxy`, or point `PRIMUS_AINIC_REPO_URL` at a mirror, or put the AINIC bundle tarball under `/opt/ainic-bundles` on every node. If the bundle name is the problem, set `PRIMUS_AINIC_BUNDLE_VERSION`. |
| `.../Packages has no libionic1 package` | The bundle directory on the repository is an empty placeholder. | Use a published bundle; see [AINIC bundle versions](./ainic-bundle-versions.md#4-list-published-bundles). |
| `dpkg-deb is required ...` or `cannot determine the OS codename ...` | The image is not Debian/Ubuntu-based, and AINIC packages are Ubuntu `.deb` files. | Use an Ubuntu-based image, or install `libionic` into the image yourself. |
| `cannot check the RDMA userspace driver: cannot load libibverbs.so.1` | The image has no RDMA userspace at all. | Use an image with `libibverbs1` and `ibverbs-providers` (the Primus images have them). |
| `PATH_TO_BNXT_TAR_PACKAGE=... is not readable here` | The path does not exist inside the container. | Check the path on the host. `primus-cli container` mounts it automatically; a container you start yourself needs `--volume`. |
| `nodes that did get a matching provider will use RDMA, and RCCL can hang on mixed transports` | A multi-node job where this node has no working provider. | Fix this node as above, or set `PRIMUS_NIC_DRIVER_STRICT=1` so the job stops instead of hanging. |
| `python3 >= 3.7 not found`, `the NIC userspace driver check exited with N`, or `setup crashed` | The hook itself could not run. The launch continued without it. | Run the `status` command above and report the problem (see below). |
| `could not restore ... (backup kept in ...)` | A rollback failed, which should not happen. | Restart the container; the backup directory holds the original files. |

If the table does not get you there, these always work:

- **Name the source explicitly.** Set `PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-<version>.tar.gz` (Broadcom) or `PRIMUS_AINIC_BUNDLE_VERSION=<bundle>` (AINIC).
- **Turn the hook off.** `PRIMUS_NIC_USERSPACE_DRIVER=off` restores the old behavior: the image's provider is used as is.
- **Install the provider yourself**, in a running container or in a derived image. For Broadcom, in an image with autotools, a C compiler and `libibverbs-dev`:

  ```bash
  tar xzf libbnxt_re-<version>.tar.gz && cd libbnxt_re-<version>
  sh autogen.sh && ./configure && make -j && make install
  # Replace the image's provider; keep the rdmavNN suffix the image already uses.
  cp /usr/local/lib/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so
  ```

  For AINIC, rebuild the image as described in [AINIC bundle versions](./ainic-bundle-versions.md). An image built this way only matches hosts that run that driver release.

**Reporting a problem.** Include, from the failing node: the `[nic-driver]` lines of the launch log; the output of the `status` command run in the job's environment; the host driver version (`cat /sys/module/bnxt_re/version` for Broadcom, `cat /sys/class/infiniband/*/fw_ver` for AINIC); the image tag; and the build log if one is named.

---

## 6. Socket configuration

CPU-side and fallback socket traffic uses interface selection:

| Variable | Role |
|----------|------|
| `NCCL_SOCKET_IFNAME` | Interface name or pattern for NCCL socket transport (e.g. `eth0`, or `^docker0,lo` to **exclude** virtual interfaces). |
| `GLOO_SOCKET_IFNAME` | Interface for **Gloo** process groups (CPU barriers and related). |

**Primus behavior:** `base_env.sh` sets `IP_INTERFACE` via `runner/helpers/envs/get_ip_interface.sh` (fallback: first address from `hostname -I`). Both `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` default to **`IP_INTERFACE`** when unset.

**Requirement:** All nodes must agree on a **reachable** address family and interface choice; mismatched bindings are a frequent source of hangs.

---

## 7. PCIe cross-NIC (PXN)

| Variable | Default in `base_env.sh` | Meaning |
|----------|--------------------------|---------|
| `NCCL_PXN_DISABLE` | `1` | **PXN disabled** by default (saves GPU memory per comment in `base_env.sh`). |

When **`NCCL_PXN_DISABLE=0`**, **PCIe cross-NIC** is enabled: GPUs might use NICs attached to **other** PCIe switches, which can improve **multi-rail** bandwidth at the cost of **higher GPU memory** use. `runner/use_ainic.yaml` sets `NCCL_PXN_DISABLE=0` for AINIC-oriented runs.

---

## 8. Network diagnostics

### Preflight

```bash
primus-cli direct -- preflight --network
```

For multi-node (Slurm example):

```bash
primus-cli slurm srun -N 4 -- preflight --host --gpu --network
```

See `docs/02-user-guide/preflight.md` for flags, output locations (`output/preflight` by default), and interpretation.

Set **`PRIMUS_EXPECT_IB=1`** when InfiniBand is **required** for validation; preflight uses this in `primus/tools/preflight/network/network_standard.py`.

### RCCL benchmark

```bash
primus-cli slurm srun -N 4 -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M
```

This exercises collective bandwidth and latency across a message-size sweep. See `docs/02-user-guide/micro-benchmarking.md` and `primus/tools/benchmark/rccl_bench_args.py` for options (dtypes, operations, output files).

### Verbose RCCL logs

```bash
export NCCL_DEBUG=INFO
```

Use for short, controlled runs; **TRACE** can be extremely verbose.

---

## 9. Multi-node setup checklist

- **ROCm version** matches across all nodes (driver and container image).
- **`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`** (or auto-detected `IP_INTERFACE`) identify the **same logical network** on every node.
- **InfiniBand or RoCE** is up (`ibstat`, `/dev/infiniband`, kernel modules such as `ib_core` / `mlx5_core` as appropriate).
- **Firewall** allows ports required by your launcher and collective tests (`MASTER_ADDR` / `MASTER_PORT` reachable).
- **`MASTER_ADDR`** resolves and is reachable from **all** nodes.
- **`GPUS_PER_NODE`** matches physical GPUs per node.
- **Containers** mount `/dev/kfd`, `/dev/dri`, and `/dev/infiniband` when using IB/RoCE/AINIC (see `runner/use_ainic.yaml`).
- **Broadcom NICs:** the NetXtreme-E bundle used for the host driver is on every node under `PRIMUS_NIC_DRIVER_SEARCH_PATH`, or `libbnxt_re` is installed on the hosts (see [section 5](#5-rdma-userspace-driver-libibverbs-provider)).

---

## 10. Troubleshooting network issues

| Symptom | What to check |
|---------|----------------|
| **Timeout or hang** at init | `MASTER_ADDR` / `MASTER_PORT`, firewall, VPN, wrong `NCCL_SOCKET_IFNAME`, or inconsistent interface across nodes. |
| **Slow** collectives | IB vs Ethernet path, `NCCL_NET_GDR_LEVEL`, fabric errors, or contention; compare **`benchmark rccl`** to baseline. |
| **IB not detected** | `/dev/infiniband` missing, modules not loaded, or wrong container devices. |
| **`does not support the kernel ABI`**, `NET/IB : No device found`, `Using network Socket` | The container's RDMA provider does not match the host driver, and the `[nic-driver]` lines at the top of the log say why the hook could not install one. See [When RDMA still does not work](#when-rdma-still-does-not-work). |
| **`ibv_create_cq fails ... Operation not permitted`** in the `[nic-driver]` output | Locked memory is limited. Run the container with `--ulimit memlock=-1` or `--privileged` (the Primus default). |
| **Wrong interface** | Restrict with `NCCL_SOCKET_IFNAME=^docker0,lo` (exclude loopback and Docker bridges). |
| **GID / RoCE issues** | `NCCL_IB_GID_INDEX` vs site documentation; RoCE v2 settings (`NCCL_IB_ROCE_VERSION_NUM`). |

For a consolidated list of `NCCL_*` / `RCCL_*` variables, see `docs/03-configuration-reference/environment-variables.md` and the upstream [RCCL environment variables](https://rocm.docs.amd.com/projects/rccl/en/develop/api-reference/env-variables.html) documentation.

---

## Related documentation

- [Preflight diagnostics](../02-user-guide/preflight.md)
- [Micro-benchmarking suite](../02-user-guide/micro-benchmarking.md)
- [NCCL/RCCL collective operations](./collective-operations.md)
- [Environment variables](../03-configuration-reference/environment-variables.md)
