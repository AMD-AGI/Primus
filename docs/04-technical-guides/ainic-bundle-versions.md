# AINIC bundle versions

AMD published training images ship a **fixed AINIC bundle**. The bundle version is
baked in at build time through the `AINIC_BUNDLE_VERSION` build argument, and the
image installs the userspace half of the AINIC stack from
`https://repo.radeon.com/amdainic/pensando/ubuntu/<bundle>/`.

Sometimes the bundle in the image is not the one your cluster needs. This guide
shows how to rebuild a published image against an arbitrary bundle, and how to
confirm the change landed. Both directions are supported; a downgrade is a normal
and fully supported operation here, not a workaround.

> **Scope.** This guide covers **making the change** to the AINIC **hostlib**
> inside the container, and verifying it took effect.
>
> **Deciding which bundle you need is out of scope and is your responsibility.**
> The hostlib in the container has to be compatible with the `ionic` driver on the
> host, and only your cluster's operators know what that is. See
> [section 3](#3-choosing-a-bundle) for what to check before you pick a version.
>
> This guide also does not change the host driver, and does not change how AINIC
> is *enabled* at runtime — for that, see
> [Multi-node networking](./multi-node-networking.md#4-ainic-amd-ai-nic).

---

## 1. What is actually in the image

An AINIC-enabled training image contains only the **userspace** half of the pair:

| Package | Role |
|---------|------|
| `libionic1` | User space ionic provider driver for `libibverbs` |
| `libionic-dev` | Development headers for the ionic provider |

The driver-side packages of the bundle — `ionic-dkms`, `pds-dkms`, `nicctl`,
`ainic-monitor` — are **not** installed in the container. They belong to the host.
This is why bundle mismatches show up as a host/container skew rather than as a
missing package.

Note that the bundle name and the hostlib version are **different numbering
schemes**, and they are easy to confuse when discussing versions. The table below
is ordered by bundle version, newest first; the hostlib column does not follow,
because a higher bundle version does not imply a higher hostlib version:

| Bundle | `libionic` version |
|--------|--------------------|
| `1.125.0-a-187` | `54.0-192-1` |
| `1.117.5-a-147` | `54.0-197-1` |
| `1.117.5-a-77` | `54.0-187-1` |
| `1.117.5-a-56` | `54.0-184` |
| `1.117.1-a-63` | `54.0-149.g3304be71` |

## 2. Check what you have

```bash
# hostlib version actually installed in the image
docker run --rm <image> dpkg-query -W -f='${Package} ${Version}\n' libionic1 libionic-dev

# which bundle repositories the image was built against
docker run --rm <image> grep -rh amdainic /etc/apt/sources.list.d/
```

**The repository list is not a reliable indicator of the installed version.** An
image can carry several AINIC repositories, and `apt` resolves `libionic` to the
highest version across *all* enabled repositories. An image whose Dockerfile names
one bundle can therefore ship the hostlib of a different one. Always trust
`dpkg-query`, not the `sources.list` entries or the build argument.

On the **host**, check the other half of the pair:

```bash
lspci | grep -i pensando
modinfo ionic | grep -E '^version|^srcversion'
ls /sys/class/infiniband/          # expect ionic_* entries
```

## 3. Choosing a bundle

**Which bundle you need is determined by your cluster, not by this guide.** The
hostlib in the container must be compatible with the `ionic` driver on the host,
so the version to install is whatever your cluster's operators tell you it is.

Two things are worth knowing before you pick:

- **Newer is not automatically better.** Compatibility is not monotonic in the
  bundle version, so the correct target may be older than what the image already
  ships. That is why this guide supports downgrades as a first-class operation.
- **The two halves must agree on the uverbs ABI.** The kernel side is readable on
  the host:

  ```bash
  cat /sys/class/infiniband_verbs/uverbs*/abi_version
  ```

If you install a bundle the host cannot work with, the build still succeeds and
every package-level check in section 7 still passes — the mismatch only shows up
as an unreachable fabric at runtime. Section 7 covers how to confirm that,
which is why it is worth doing even when the build looks clean.

## 4. List published bundles

```bash
curl -s https://repo.radeon.com/amdainic/pensando/ubuntu/ | grep -oE 'href="[^"]+/"'
```

To see the exact package versions a bundle ships before committing to it:

```bash
BUNDLE=1.117.5-a-147
curl -s "https://repo.radeon.com/amdainic/pensando/ubuntu/${BUNDLE}/dists/noble/main/binary-amd64/Packages" \
  | awk '/^Package: /{p=$2} /^Version: /{print p"="$2}'
```

Some published directories are **empty placeholders** — they contain only `conf/`
and `db/`, with no `dists/` or `pool/`, and `apt` cannot install from them. The
`Packages` query above returns nothing for these, which is the quickest way to
spot one before a build fails.

## 5. Rebuild the image

The block below is a **complete Dockerfile**, both `RUN` stanzas included. Save
it as `Dockerfile` in an otherwise empty directory and run the build from there —
the whole directory is sent to the Docker daemon as build context, so keeping it
empty keeps the build fast.

Both arguments are placeholders, supplied at build time: set `BASE_IMAGE` to the
published image you already run, and `AINIC_BUNDLE_VERSION` to the bundle your
host driver needs. Neither has a default, so omitting one fails the build rather
than silently producing an unintended image.

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AINIC_BUNDLE_VERSION

RUN set -eux; \
    rm -f /etc/apt/sources.list.d/*amdainic*; \
    add-apt-repository -y "deb https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_BUNDLE_VERSION} noble main"; \
    apt update --allow-insecure-repositories; \
    ver="$(apt-cache madison libionic1 | awk -F'|' 'NR==1{gsub(/ /,"",$2); print $2}')"; \
    test -n "$ver"; \
    echo "installing libionic $ver from ${AINIC_BUNDLE_VERSION}"; \
    apt install -y --allow-unauthenticated --allow-downgrades \
        "libionic-dev=$ver" "libionic1=$ver"; \
    rm -rf /var/lib/apt/lists/*

# Fail the build rather than ship an image whose hostlib silently did not move.
RUN set -eux; \
    dpkg-query -W -f='${Package} ${Version}\n' libionic1 libionic-dev; \
    dpkg -C; \
    test -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so; \
    readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
```

Then, from that same directory:

```bash
BASE=rocm/jax-training:maxtext-v26.6
BUNDLE=1.117.5-a-147

docker build --network host \
  --build-arg BASE_IMAGE=$BASE \
  --build-arg AINIC_BUNDLE_VERSION=$BUNDLE \
  -t ${BASE##*/}-ainic-$BUNDLE .
```

Tagging the image after the bundle it contains is worth the small effort: the
bundle is otherwise invisible without running `dpkg-query` inside the image.

### Why the two extra steps

Both of the lines that look like boilerplate are load-bearing, and **both failure
modes are silent** — apt reports success and the build produces an image with the
wrong hostlib:

1. **`rm -f /etc/apt/sources.list.d/*amdainic*`** — without this, the bundle
   already configured in the base image stays enabled, apt picks the highest
   version across all of them, and `AINIC_BUNDLE_VERSION` becomes advisory rather
   than binding.

2. **Pinning `libionic-dev=$ver libionic1=$ver`** — an unpinned `apt install` is a
   no-op when the requested bundle is *older* than what the base image carries;
   apt reports `already the newest version` and exits 0. `--allow-downgrades` only
   *permits* a downgrade, it does not request one.

The pinned version must be read with `apt-cache madison`, which reports only what
the repository offers. `apt-cache policy ... Candidate` cannot be used here: apt
refuses to nominate a downgrade as the candidate at any pin priority, so for an
older bundle it reports the *installed* version and the pin silently becomes a
no-op again.

Because of the verification step at the end, a build that hits either of these
now fails instead of shipping.

## 6. No uninstall is required

Upgrades and downgrades both apply in place. `dpkg` unpacks the new version over
the old one with `0 to remove`, and `dpkg -C` stays clean afterwards.

This is worth stating explicitly because the hostlib filename embeds its version
(`libionic.so.1.1.54.0-187`), which normally suggests old files would accumulate.
They do not: `dpkg` removes the previous versioned object and repoints both
`libionic.so.1` and the `libibverbs` provider symlink `libionic-rdmav34.so` at the
new one.

## 7. Verify

```bash
docker run --rm --privileged --network host --cap-add=IPC_LOCK \
  -v /dev/infiniband:/dev/infiniband <image> bash -c '
    dpkg-query -W -f="libionic1 \${Version}\n" libionic1
    readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
    readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
    dpkg -C && echo "dpkg consistent"
    ibv_devices
  '
```

Check that the package version and the provider symlink agree, and that
`ibv_devices` lists `ionic` devices.

Two traps when automating this check:

- **Treat "no `ionic` devices" as a failure, not a warning.** An image whose
  hostlib cannot attach to the fabric passes every package-level check above. If
  the node does have AINIC hardware, an empty enumeration means the hostlib is
  incompatible — see section 3.
- **Do not compare versions by parsing the `.so` filename.** The soname scheme is
  not stable across bundles, so a literal prefix match rejects images that are in
  fact correct. Check package ownership instead, which is what you actually want
  to know and also catches a stale file:

  ```bash
  so=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1)
  dpkg -L libionic1 | grep -qxF "$so" && echo "owned by installed libionic1"
  ```

### Confirm the fabric is actually used

The checks above prove the *hostlib you asked for is installed*. They cannot prove
it works with your host. Do not skip this step on the grounds that the build was
clean and training runs — when the two halves are incompatible, RCCL falls back to
TCP and the job completes normally. In one measured 2-node, 16-GPU MaxText
comparison the fallback cost **3.2x throughput (245.6 vs 791.7 TFLOP/s/device)**
while completing every step and converging to an identical loss. Nothing in the
training output distinguished the two runs.

```bash
export NCCL_NET_PLUGIN=librccl-anp.so
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

In the resulting log, confirm the ANP plugin was selected and an `ionic` device
was chosen. A fallback to `NET/Socket` will still complete the job and report
plausible numbers while never touching the NIC. See
[Multi-node networking](./multi-node-networking.md#rccl-network-plugin-anp) for
how Primus sets `NCCL_NET_PLUGIN` through `runner/helpers/hooks/03_enable_ainic.sh`.

## 8. Installing from local `.deb` files

For a bundle that is not published — an internal or pre-release build — place the
two `.deb` files next to the Dockerfile and replace the install step:

```dockerfile
COPY libionic1_*.deb libionic-dev_*.deb /tmp/
RUN dpkg -i /tmp/libionic1_*.deb /tmp/libionic-dev_*.deb && rm /tmp/*.deb
```

`dpkg -i` applies the given version in either direction with no extra flags, so
neither pitfall in section 5 applies.

## 9. Troubleshooting

| Symptom | Cause |
|---------|-------|
| `apt install` reports `already the newest version` and nothing changes | Target bundle is older than the installed hostlib and the version was not pinned. See section 5. |
| Built image still has the old hostlib, build reported success | Old bundle repositories left enabled in the base image. See section 5. |
| `E: Packages were downgraded and -y was used without --allow-downgrades` | Expected for a downgrade; add `--allow-downgrades`. |
| `apt update` cannot find the repository | Bundle directory is an empty placeholder, or the name is wrong. See section 4. |
| `ibv_devices` lists no `ionic` device | Not AINIC hardware; the host `ionic` driver is not loaded; or the hostlib is ABI-incompatible with the host driver. See sections 2 and 3. |
| Training runs fine but throughput is far below expectation | RCCL fell back to `NET/Socket`. The job still completes and converges normally, so check the transport rather than the loss. See section 7. |
| The requested hostlib is installed but the fabric is unreachable | The installed bundle is not compatible with the host. Bundle selection is a cluster question — confirm the required version with your operators, and note that it may be older than what the image shipped. See section 3. |

---

## Related documentation

- [Multi-node networking](./multi-node-networking.md): enabling AINIC, ANP plugin, and NCCL/RCCL variables
- [Preflight diagnostics](../02-user-guide/preflight.md)
- [Deployment](../05-operations/deployment.md)
- [Troubleshooting](../05-operations/troubleshooting.md)
