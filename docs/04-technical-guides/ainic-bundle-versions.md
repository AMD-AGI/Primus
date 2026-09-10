# AINIC bundle versions

AMD published training images ship a **fixed AINIC bundle**. The bundle version is
baked in at build time through the `AINIC_BUNDLE_VERSION` build argument, and the
image installs the userspace half of the AINIC stack — the `libionic` library —
from `https://repo.radeon.com/amdainic/pensando/ubuntu/<bundle>/`.

Sometimes the bundle in the image is not the one your cluster needs. This guide
shows how to rebuild a published image against an arbitrary bundle, and how to
confirm the change landed. Upgrades and downgrades are both normal operations
here; the correct bundle for your cluster is often an older one.

> **Scope.** This guide covers **making the change** to `libionic` inside the
> container, and verifying it took effect. **Deciding which bundle you need is
> your responsibility** — `libionic` in the container has to be compatible with
> the `ionic` driver on the host. See [section 3](#3-choosing-a-bundle), which
> includes host commands that often answer the question directly.
>
> This guide does not change the host driver, and does not change how AINIC is
> *enabled* at runtime — for that, see
> [Multi-node networking](./multi-node-networking.md#4-ainic-amd-ai-nic).
>
> This guide assumes a **published** bundle throughout. If you have been given a
> bundle as a tarball rather than a repository, see
> [section 7](#7-installing-from-a-pre-downloaded-bundle).
>
> **What this was tested on.** The MaxText JAX training images —
> `rocm/jax-training:maxtext-v26.3.2`, `-v26.4`, and `-v26.6`. The rebuild was
> exercised on all three; the runtime check in [section 6](#6-verify) on
> `v26.3.2`. It has **not** been tested on Primus's own `rocm/primus:*` images.
> Nothing here is specific to MaxText — it operates on `libionic` and the apt
> configuration, not on the framework — but a different image may enable
> different bundle repositories or ship a different `libionic` to begin with, so
> confirm with section 6 rather than assuming the versions quoted here.

---

## Quickstart

If you already know which bundle you need, this is the whole procedure. Save the
block below as `Dockerfile` in an otherwise empty directory — the whole directory
is sent to the Docker daemon as build context, so keeping it empty keeps the
build fast.

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AINIC_BUNDLE_VERSION

RUN set -eux; \
    codename="$(. /etc/os-release; echo "$VERSION_CODENAME")"; \
    rm -f /etc/apt/sources.list.d/*amdainic*; \
    add-apt-repository -y "deb https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_BUNDLE_VERSION} $codename main"; \
    apt update --allow-insecure-repositories; \
    ver="$(apt-cache madison libionic1 \
            | awk -F'|' -v b="/${AINIC_BUNDLE_VERSION} " 'index($3,b){gsub(/ /,"",$2); print $2; exit}')"; \
    test -n "$ver"; \
    echo "installing libionic $ver from ${AINIC_BUNDLE_VERSION}"; \
    apt install -y --allow-unauthenticated --allow-downgrades \
        "libionic-dev=$ver" "libionic1=$ver"; \
    test "$(dpkg-query -W -f='${Version}' libionic1)" = "$ver"; \
    test "$(dpkg-query -W -f='${Version}' libionic-dev)" = "$ver"; \
    rm -rf /var/lib/apt/lists/*

# Structural check: the package is installed consistently and the provider
# symlink the fabric actually loads points at the installed object.
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

`--network host` is needed when the daemon's default bridge network cannot
resolve `repo.radeon.com`; it is harmless otherwise, so it is included by
default.

Neither build argument has a default, so omitting one fails the build rather than
silently producing an unintended image. Tagging the image after the bundle is
worth the small effort: the bundle is otherwise invisible without running
`dpkg-query` inside the image.

**A clean build does not mean the image works on your host.** It means the
version you asked for was installed. See [section 6](#6-verify) before trusting
it, and [section 5](#5-how-the-rebuild-works) for why the apparently boilerplate
lines are load-bearing.

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

Note that the bundle name and the `libionic` version are **different numbering
schemes**, and they are easy to confuse when discussing versions. The excerpt
below is ordered by bundle version, newest first; the `libionic` column does not
follow, because a higher bundle version does not imply a higher `libionic`
version. See [section 4](#4-list-published-bundles) to list every published
bundle.

| Bundle | `libionic` version |
|--------|--------------------|
| `1.125.0-a-187` | `54.0-192-1` |
| `1.117.5-a-147` | `54.0-197-1` |
| `1.117.5-a-77` | `54.0-187-1` |
| `1.117.5-a-56` | `54.0-184` |
| `1.117.1-a-63` | `54.0-149.g3304be71` |

Every published bundle uses this `54.0-NNN` scheme and ships one `libionic`
version for all distributions. A bundle delivered as a tarball may not — read the
version out of the installed package (or the `.deb` itself) rather than assuming
the scheme ([section 7](#7-installing-from-a-pre-downloaded-bundle)).

## 2. Check what you have

```bash
# libionic version actually installed in the image
docker run --rm <image> dpkg-query -W -f='${Package} ${Version}\n' libionic1 libionic-dev

# which bundle repositories the image was built against
docker run --rm <image> grep -rh amdainic /etc/apt/sources.list.d/
```

**The repository list is not a reliable indicator of the installed version.** An
image can carry several AINIC repositories, and `apt` resolves `libionic` to the
highest version across *all* enabled repositories. An image whose Dockerfile names
one bundle can therefore ship the library of a different one. Always trust
`dpkg-query`, not the `sources.list` entries or the build argument.

On the **host**, check the other half of the pair:

```bash
lspci | grep -i pensando
modinfo ionic | grep -E '^version|^srcversion'
ls /sys/class/infiniband/          # expect ionic_* entries
```

## 3. Choosing a bundle

`libionic` in the container must be compatible with the `ionic` driver on the
host. Start by asking the host, which frequently answers the question outright:

```bash
# Often reports the AINIC bundle version directly, e.g. 1.117.5-a-77
cat /sys/class/infiniband/*/fw_ver

# Same information from the network device side
ethtool -i <ionic_netdev> | grep -i firmware

# The host driver build, which tracks the bundle's linux-ionic component
modinfo ionic | grep '^version'
```

If `fw_ver` reports a bundle version, that is the bundle the host is running and
the natural target for the container. Where it does not, or where the host is
managed by someone else, bring your cluster's operators the output above together
with:

```bash
cat /sys/class/infiniband_verbs/uverbs*/abi_version
```

This is the kernel side of the uverbs ABI, the interface `libionic` has to speak.
There is no matching number you can read out of the container to compare it
against — a provider library declares the ABI range it supports internally, not
as a queryable version. Note also that this value is **coarse**: hosts running
different bundles can report the same ABI, so it will not distinguish two
candidate bundles from each other. Treat it as an input to the conversation with
your operators, not as a selection test. The authoritative test is the runtime one
in [section 6](#6-verify).

One thing worth knowing before you pick: **newer is not automatically better.**
Compatibility is not monotonic in the bundle version, so the correct target may
well be older than what the image already ships.

If you install a bundle the host cannot work with, the build still succeeds and
every package-level check in section 6 still passes. Package-level checks cannot
see a host/container skew at all, which is why section 6 ends with a runtime
check. Note that the converse also holds: a skewed pair is **not** guaranteed to
fail visibly — userspace/firmware compatibility spans a range of versions rather
than requiring an exact match, so a mismatched pair may enumerate devices and
reach the fabric normally. Matching the bundle to the host remains the supported
configuration, and the runtime check tells you which situation you are in.

## 4. List published bundles

```bash
curl -s https://repo.radeon.com/amdainic/pensando/ubuntu/ | grep -oE 'href="[0-9][^"]+/"'
```

The `[0-9]` anchor keeps the page header and parent-directory links out of the
result, so every line is a bundle.

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

Check for this even if you are deliberately picking the newest bundle: **the
newest entries in the listing are currently placeholders**, so choosing the
highest version is exactly the case that hits it. A build against a placeholder
fails at `apt update` with `404 Not Found` on the `Packages` file, before the
bundle version is ever evaluated.

This query also only works for published bundles. To install from a
pre-downloaded tarball instead, see
[section 7](#7-installing-from-a-pre-downloaded-bundle).

## 5. How the rebuild works

Four lines in the quickstart Dockerfile look like boilerplate and are not.

1. **Deriving `$codename` from `/etc/os-release`** — the repository path is
   per-distribution. Hardcoding `noble` works only for a noble base image and
   silently produces a `404` on any other, so it is read from the base image
   instead.

2. **`rm -f /etc/apt/sources.list.d/*amdainic*`** — the base image already has one
   or more AINIC repositories enabled. This matters less for the build itself than
   for what happens afterwards: **any later `apt install` or `apt upgrade` in a
   downstream layer can pull `libionic` back to whichever bundle offers the
   highest version**, silently undoing the rebuild. Removing the repositories
   makes the change durable.

3. **Selecting `$ver` from the requested bundle specifically** — the
   `index($3, "/${AINIC_BUNDLE_VERSION} ")` filter takes the version offered by
   *that* repository rather than the highest across all of them. Without the
   filter, a stale repository offering a higher version turns the pin into a
   request for the version you already have.

4. **Pinning `libionic-dev=$ver libionic1=$ver`** — an unpinned `apt install` is a
   no-op when the requested bundle is *older* than what the base image carries;
   apt reports `already the newest version` and exits 0. `--allow-downgrades` only
   *permits* a downgrade, it does not request one.

The pinned version must be read with `apt-cache madison`, which reports only what
the repository offers. `apt-cache policy ... Candidate` cannot be used here: apt
refuses to nominate a downgrade as the candidate at any pin priority, so for an
older bundle it reports the *installed* version and the pin silently becomes a
no-op again.

### The version assertion is what makes a silent failure loud

```dockerfile
    test "$(dpkg-query -W -f='${Version}' libionic1)" = "$ver"; \
    test "$(dpkg-query -W -f='${Version}' libionic-dev)" = "$ver";
```

These two lines are the only thing in the Dockerfile that compares what was
*installed* against what was *requested*. The structural checks in the second
`RUN` — `dpkg -C`, the provider symlink, `readlink` — all pass happily on an image
whose `libionic` never moved, because that image is perfectly self-consistent; it
is just built against the wrong bundle. Without the assertion, an unpinned install
that apt reported as `already the newest version` produces a successful build and
a wrong image.

### No uninstall is required

Upgrades and downgrades both apply in place. `dpkg` unpacks the new version over
the old one with `0 to remove`, and `dpkg -C` stays clean afterwards.

This is worth stating explicitly because the library filename embeds its version
(`libionic.so.1.1.54.0-187`), which normally suggests old files would accumulate.
They do not: `dpkg` removes the previous versioned object and repoints both
`libionic.so.1` and the `libibverbs` provider symlink `libionic-rdmav34.so` at the
new one. This holds across repeated changes, including moving out to another
bundle and back again.

## 6. Verify

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

Three traps when automating this check:

- **Treat "no `ionic` devices" as a failure, not a warning.** An image whose
  `libionic` cannot attach to the fabric passes every package-level check above.
  If the node does have AINIC hardware, an empty enumeration means the library is
  incompatible — see section 3.
- **A successful enumeration does not mean the bundle matches the host.** A
  container whose `libionic` comes from a different bundle than the host firmware
  will still enumerate every device normally. Enumeration detects an absent or
  grossly incompatible provider; it has no power to detect a version skew.
- **Do not compare versions by parsing the `.so` filename.** The soname scheme is
  not stable across bundles, so a literal prefix match rejects images that are in
  fact correct. Check package ownership instead, which is what you actually want
  to know and also catches a stale file:

  ```bash
  so=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1)
  dpkg -L libionic1 | grep -qxF "$so" && echo "owned by installed libionic1"
  ```

### Confirm the fabric is actually used

The checks above prove the *library you asked for is installed*. They cannot prove
it works with your host. Do not skip this step on the grounds that the build was
clean and training runs — when RCCL cannot use the NIC it falls back to TCP, and
the job still completes every step and converges to the same loss. The throughput
cost of that fallback is severe, but nothing else in the training output
distinguishes the two runs, so the transport has to be checked directly.

Run the job with RCCL logging enabled:

```bash
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

Then confirm in the log that an `ionic` device was selected and that the transport
is not `NET/Socket`:

```bash
grep -oE 'NET/IB|NET/Socket' rank0.log | sort | uniq -c
grep -oiE 'using network [A-Za-z-]+' rank0.log | sort -u
grep -oE 'ionic_[0-9]+' rank0.log | sort -u
```

**Any `NET/Socket` in the selected path is a failure**, regardless of whether the
job completes.

`NCCL_NET_PLUGIN` is set for you and the expected success signature depends on
which value is in effect, so check the one that applies:

| `NCCL_NET_PLUGIN` | Expected log line | Notes |
|---|---|---|
| `none` | `Using network IB` | RCCL's built-in verbs path. Also emits a `NET/Plugin: Could not find: none librccl-net-none.so.` line, which is **expected and harmless**. |
| `librccl-anp.so` | `Using network RCCL` | The external ANP plugin. May emit `Failed to find ncclCollNetPlugin_vN symbol` warnings, which are benign — the library provides a net plugin, not a collnet plugin. |

Primus selects between these in
`runner/helpers/hooks/03_enable_ainic.sh`, which sets `NCCL_NET_PLUGIN=none` on
ROCm builds where ANP is compiled into RCCL. **Let the hook decide** — an
explicitly exported `NCCL_NET_PLUGIN` always wins and will override it. See
[Multi-node networking](./multi-node-networking.md#rccl-network-plugin-anp) for
the full picture.

## 7. Installing from a pre-downloaded bundle

Occasionally a bundle is delivered as a tarball rather than published to
`repo.radeon.com`. **Prefer the apt path in [section 5](#5-how-the-rebuild-works)
whenever the bundle you need is published** — it handles either direction with no
extra flags and none of the care below.

Do not unpack the tarball on the host. Copy the bundle into the image and let
the build unpack it: that picks the distribution-matching `.deb` files from
inside the container, so you do not have to probe the base image's codename or
copy individual packages into the build context.

The same files live in `tools/ainic-bundle-rebuild/` if you would rather not
copy them out. From the Primus repository root:

```bash
./tools/ainic-bundle-rebuild/build.sh \
  rocm/jax-training:maxtext-v26.6 \
  ainic_bundle_1.117.5-a-147.tar.gz

./tools/ainic-bundle-rebuild/test.sh \
  jax-training:maxtext-v26.6-ainic-1.117.5-a-147
```

If the packaging of the bundle you were given differs from a published one, the
specific differences observed in one internal build — and the error each
produces — are recorded in
[Appendix A](#appendix-a-packaging-differences-observed-in-an-internal-build).
You should not need it to follow this section, only to recognise a symptom.

### Dockerfile

Save this as `Dockerfile` next to the bundle, or use the copy in
`tools/ainic-bundle-rebuild/Dockerfile`. Use `tar -xf` rather than `tar -xJf`
throughout: an inner archive is sometimes gzip regardless of its `.tar.xz` name,
and `-xf` detects the format either way.

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AINIC_BUNDLE

RUN mkdir /tmp/ainic
COPY ${AINIC_BUNDLE} /tmp/ainic/bundle.tar.gz
RUN set -ex; \
    rm -f /etc/apt/sources.list.d/*amdainic*; \
    cd /tmp/ainic; \
    tar -xf bundle.tar.gz; \
    cd */; \
    tar -xf host_sw_pkg.tar*; \
    cd host_sw_pkg/ionic_driver/deb; \
    tar -xf libionic-debs.tar* 2>/dev/null || tar -xf rdma-core-debs.tar*; \
    REL_NAME="$(. /etc/os-release; echo "$VERSION_CODENAME")"; \
    dpkg -i --force-overwrite "$REL_NAME"/libionic1_*.deb; \
    dpkg -i --force-overwrite "$REL_NAME"/libionic-dev_*.deb; \
    cd /tmp; \
    rm -rf /tmp/ainic

# Structural check: the package is installed consistently and the provider
# symlink the fabric actually loads points at the installed object.
RUN set -eux; \
    dpkg-query -W -f='${Package} ${Version}\n' libionic1 libionic-dev; \
    dpkg -C; \
    ls /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav*.so; \
    readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
```

The install `RUN` is doing four things that are easy to flatten incorrectly:

1. **Unpack inside the image, then install from `$VERSION_CODENAME/`.** The
   bundle extracts to one directory per distribution. Installing from the
   codename that matches *this* image avoids handing `dpkg` conflicting
   versions from `jammy/` and `noble/` at once. The inner archive carrying
   `libionic` has been seen named both `libionic-debs.tar.xz` and
   `rdma-core-debs.tar.xz`; `linux-debs.tar.xz` and `perftest-debs.tar.xz` in
   the same directory are the host driver and benchmark packages and must not
   be installed in the container.

2. **Two separate `dpkg -i` invocations, `libionic1` first.** Where
   `libionic-dev` **`Pre-Depends`** on an exact `libionic1` version, a single
   `dpkg -i` given both files cannot satisfy it: a pre-dependency must be met
   by an already-*configured* package, which one pass over both archives cannot
   arrange. It fails with `pre-dependency problem - not installing libionic-dev`
   *after* having already replaced `libionic1`, leaving the layer half-converted.
   Installing in two steps costs nothing when there is no pre-dependency.

3. **`--force-overwrite`.** The bare `libionic.so` symlink is not always owned by
   the same package — it has been seen shipped by `libionic1` and by
   `libionic-dev` — and the packages declare no `Replaces` for each other. Where
   the owning package changes, the install fails with
   `trying to overwrite '/usr/lib/x86_64-linux-gnu/libionic.so', which is also in package ...`.
   The flag is harmless when ownership does not change.

4. **`rm -f /etc/apt/sources.list.d/*amdainic*`.** More important here than on the
   apt path. A bundle installed from local files exists in no repository, and its
   version may well sort *below* what the base image's repositories offer, in
   which case apt regards the bundle you just replaced as an upgrade. Any later
   `apt install` or `apt upgrade` in a downstream layer then quietly reverts your
   bundle, and `dpkg -C` still reports a consistent system afterwards. There is no
   diagnostic signal at all. Removing the repositories is the only thing that
   makes a local install durable.

`dpkg -i` resolves nothing. If the `.deb` `Depends`/`Pre-Depends` are not already
satisfied in the base image — some builds pin a narrow `ibverbs-providers` range
against a newer `rdma-core` — the install fails rather than pulling
dependencies. Inspect with `dpkg-deb -f` on the extracted packages if you need
to see why.

There is no `$ver` from a repository on this path, so the version assertion from
section 5 does not carry over. The structural `RUN` still applies: it prints
whatever `libionic` was in the tarball. Trust that output, not the bundle
filename.

### Build script

`AINIC_BUNDLE` is a path relative to the Docker build context, so the bundle
file has to be visible to the daemon. The helper in
`tools/ainic-bundle-rebuild/build.sh` creates a temporary context containing
only that bundle. This avoids sending unrelated (and potentially unreadable)
files from the bundle's directory to Docker, and means the tarball does not
have to sit next to the Dockerfile:

```bash
#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Syntax: $0 <base docker image> <ainic bundle file>" >&2
    exit 1
fi

BASE=$1
BUNDLE=$2
if [ ! -f "$BUNDLE" ]; then
    echo "error: bundle file not found: $BUNDLE" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUNDLE_NAME=$(basename "$BUNDLE")
BUNDLE_VER=$(echo "$BUNDLE_NAME" | sed -E \
    's/^ainic_bundle_//; s/\.tar(\.(gz|xz|bz2|zst))?$//')

# Send only the requested bundle to Docker. Using the bundle's parent directory
# as the context can be both very large and unreadable because of unrelated
# files. Prefer a hard link to avoid copying a large bundle when /tmp shares the
# same filesystem, and fall back to a regular copy otherwise.
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT
ln "$BUNDLE" "$CONTEXT/$BUNDLE_NAME" 2>/dev/null || \
    cp --reflink=auto "$BUNDLE" "$CONTEXT/$BUNDLE_NAME"

set -x
docker build --network host \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE" \
  --build-arg AINIC_BUNDLE="$BUNDLE_NAME" \
  -t "${BASE##*/}-ainic-${BUNDLE_VER}" \
  "$CONTEXT"
```

```bash
./tools/ainic-bundle-rebuild/build.sh \
  rocm/jax-training:maxtext-v26.6 \
  ainic_bundle_1.117.5-a-147.tar.gz
```

### Test script

This is a convenience wrapper around the package-level half of
[section 6](#6-verify). It does **not** replace the runtime `ibv_devices` /
RCCL log check on a node with AINIC hardware; `ibv_devices` here will be empty
if you are not on such a node.

```bash
#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Syntax: $0 <docker image>" >&2
    exit 1
fi

IMAGE=$1
MOUNTS=()
if [ -d /dev/infiniband ]; then
    MOUNTS+=(-v /dev/infiniband:/dev/infiniband)
fi

docker run --rm --privileged --network host --cap-add=IPC_LOCK \
  "${MOUNTS[@]}" "$IMAGE" bash -c 'set -e
    dpkg-query -W -f="libionic1 \${Version}\n" libionic1
    readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
    readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
    dpkg -C && echo "dpkg consistent"
    ibv_devices
  '
```

```bash
./tools/ainic-bundle-rebuild/test.sh \
  jax-training:maxtext-v26.6-ainic-1.117.5-a-147
```

## 8. Troubleshooting

| Symptom | Cause |
|---------|-------|
| `apt install` reports `already the newest version` and nothing changes | Target bundle is older than the installed `libionic` and the version was not pinned. See section 5. |
| Build fails on `test "$(dpkg-query ...)" = "$ver"` | Working as intended — the install did not move `libionic` to the requested version. Check the pin and the repository filter. See section 5. |
| Built image still has the old `libionic`, build reported success | Version assertion missing from the install step. See section 5. |
| `libionic` reverts to another version after a later `apt` command | Old bundle repositories left enabled. See section 5, and section 7 for why this is worse after a tarball install. |
| `E: Packages were downgraded and -y was used without --allow-downgrades` | Expected for a downgrade; add `--allow-downgrades`. |
| `apt update` fails with `404 Not Found` on `Packages` | Bundle directory is an empty placeholder, the name is wrong, or the distribution codename in the repository line does not match the base image. See sections 4 and 5. |
| `xz: (stdin): File format not recognized` while extracting a bundle | Inner archive is gzip despite its `.tar.xz` name. Use `tar -xf`. See section 7. |
| `dpkg: pre-dependency problem - not installing libionic-dev` | Both `.deb` files given to one `dpkg -i`. Install `libionic1` first, in its own invocation. See section 7. |
| `dpkg: trying to overwrite '.../libionic.so', which is also in package ...` | The two bundles disagree about which package owns that symlink. Add `--force-overwrite`. See section 7. |
| `ibv_devices` lists no `ionic` device | Not AINIC hardware; the host `ionic` driver is not loaded; or `libionic` is ABI-incompatible with the host driver. See sections 2 and 3. |
| Training runs fine but throughput is far below expectation | RCCL fell back to `NET/Socket`. The job still completes and converges normally, so check the transport rather than the loss. See section 6. |
| The requested `libionic` is installed but the fabric is unreachable | The installed bundle is not compatible with the host. Bundle selection is a cluster question — check `fw_ver` on the host, and note that the required version may be older than what the image shipped. See section 3. |

---

## Appendix A: packaging differences observed in an internal build

**Not required reading.** Section 7's commands already handle everything here, and
they are written so that none of it has to be known in advance. This appendix
exists so that if you hit one of these errors, you can recognise it rather than
debug it from scratch.

**Sample size is one.** These are observations from a single internal build,
compared against the published bundles available at the time of writing. They are
**not** a specification of how internal builds are packaged, and there is no reason
to assume the next one will look the same. Verify against your own files with
`dpkg-deb -f` and `dpkg-deb -c`.

| | Published bundles (all inspected) | The internal build inspected |
|---|---|---|
| `libionic` version scheme | `54.0-NNN` | dated, `<maj>.<min>.YY.MM.DD.<build>-1~<distro>` |
| Per-distro directories | yes — `bookworm`, `buster`, `jammy`, `noble`, all shipping the *same* version | yes — six, each shipping a *different* version string |
| `libionic-dev` → `libionic1` | `Depends` | `Pre-Depends` |
| `libionic1` `Replaces:` | none | `ibverbs-providers, libibverbs1` |
| Owner of bare `libionic.so` | `libionic1` | `libionic-dev` |
| Archive carrying `libionic` | `rdma-core-debs.tar.xz`, real xz | `libionic-debs.tar.xz`, gzip despite the name |

### What each one does to you

- **Dated version scheme.** The consequence is not cosmetic: a dated version sorts
  *below* the `54.0-NNN` scheme, so apt regards whatever the base image's
  repositories offer as an **upgrade**. This is the mechanism behind the silent
  revert described in section 7 — and it is why removing the repositories matters
  more for a local install than for an apt one. Do not expect the version to
  resemble the bundle name; read it with `dpkg-deb -f`.
- **A different version per distribution.** Both kinds of bundle extract to one
  directory per codename, so that layout alone is unremarkable — but where the
  published bundle puts the *same* version in every directory, this one puts a
  different version string in each (`50.0.…-1~ubu24.04` under `noble`,
  `39.0.…-1~ubu22.04` under `jammy`, and so on). Installing every `libionic1_*.deb`
  therefore hands `dpkg` genuinely conflicting versions rather than duplicates, and
  all but one will fail the base image's `ibverbs-providers` constraint — `noble`'s
  pins `ibverbs-providers (>= 50.0), (<< 51)`. Section 7 installs from
  `$VERSION_CODENAME/` inside the image so only the matching pair is used.
- **`Pre-Depends` rather than `Depends`.** Produces
  `dpkg: error processing archive ... pre-dependency problem - not installing libionic-dev`
  from a single `dpkg -i` given both files, *after* `libionic1` has already been
  replaced. Fixed by the two-invocation form in section 7.
- **`libionic.so` owned by a different package.** Produces
  `dpkg: trying to overwrite '/usr/lib/x86_64-linux-gnu/libionic.so', which is also in package libionic-dev`.
  Neither package declares a `Replaces` for the other, so `dpkg` is right to
  refuse. Fixed by `--force-overwrite` in section 7.
- **Archive misnamed `.tar.xz` while being gzip.** `tar -xJf` fails with
  `xz: (stdin): File format not recognized`. Use `tar -xf`, which detects either.

### One hint that the dated scheme is a build-pipeline artifact

The dated scheme is visible in the published repositories too, but only on a
debug-symbol package rather than a shipped one:

```bash
curl -s "https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.5-a-147/dists/noble/main/binary-amd64/Packages" \
  | awk '/^Package: /{p=$2} /^Version: /{print p"="$2}' | grep libionic
```

```
libionic-dev=54.0-197-1
libionic1=54.0-197-1
libionic1-dbgsym=22.1.26.07.10.001-1~deb10
```

The deliverables carry release versions; the `dbgsym` carries a dated one of the
same shape. So the dated form appears to be what the build pipeline emits before
release versioning is applied, which is consistent with seeing it on a build that
was never published — and is a reason to treat it as a signal that you are holding
a pre-release artifact rather than a released bundle.

---

## Related documentation

- [Multi-node networking](./multi-node-networking.md): enabling AINIC, ANP plugin, and NCCL/RCCL variables
- [Preflight diagnostics](../02-user-guide/preflight.md)
- [Deployment](../05-operations/deployment.md)
- [Troubleshooting](../05-operations/troubleshooting.md)
