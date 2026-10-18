#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Shared helpers for the Primus perf batch runners: yq bootstrap, config
# selection, EXTRA_ENV and provenance capture. Per-run helpers (config fields,
# banner, footer) live in lib/run.sh.
#
# Callers must set PRIMUS_PATH (repo root) before sourcing, and also source
# lib/control.sh before taking the batch snapshots: those use its perf_log
# and perf_on_nodes.
###############################################################################

# ---------------------------------------------------------------------------
# yq bootstrap
#
# yq is a dependency-free static Go binary, so when it is missing we fetch it
# into a user-owned cache rather than asking for root. Nothing is installed
# system-wide and the user's PATH is not permanently modified: the cache dir is
# prepended to PATH for the lifetime of this run only.
#
# Pinned rather than tracking "latest", because a benchmark tool whose own
# dependency silently changes version between runs undermines the provenance
# the rest of this directory exists to record. Override with YQ_VERSION.
# ---------------------------------------------------------------------------

YQ_VERSION="${YQ_VERSION:-v4.53.2}"
PERF_CACHE_DIR="${PERF_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/primus-perf}"

# True only for mikefarah's yq v4+. Two unrelated tools are called `yq`: this
# one (Go), and kislyuk's Python jq wrapper installed by `pip install yq`.
# They take different expressions, so finding *a* `yq` on PATH is not enough.
perf_yq_is_usable() {
    local bin="${1:-yq}" version major
    command -v "$bin" >/dev/null 2>&1 || return 1

    version=$("$bin" --version 2>&1) || return 1
    case "$version" in
        *mikefarah*) ;;
        *) return 1 ;;
    esac

    major=$(printf '%s\n' "$version" | sed -n 's/.*version v\([0-9]\+\).*/\1/p')
    [ -n "$major" ] && [ "$major" -ge 4 ]
}

perf_install_yq() {
    local dest="$PERF_CACHE_DIR/bin" arch platform url tmp

    case "$(uname -s)" in
        Linux) ;;
        *)
            echo "[ERROR] automatic yq install only covers Linux; got $(uname -s)." >&2
            return 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        i386|i686) arch="386" ;;
        *)
            echo "[ERROR] no yq build mapped for architecture $(uname -m)." >&2
            return 1
            ;;
    esac

    platform="linux_${arch}"
    url="https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_${platform}"

    echo "[INFO] yq not found; fetching ${YQ_VERSION} (${platform}) into $dest" >&2
    mkdir -p "$dest" || return 1
    tmp="$dest/.yq.download.$$"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$tmp"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$tmp"
    else
        echo "[ERROR] neither curl nor wget is available to download yq." >&2
        rm -f "$tmp"
        return 1
    fi || {
        echo "[ERROR] failed to download yq from $url" >&2
        echo "        Install it manually and re-run, or set YQ_VERSION to an available release." >&2
        rm -f "$tmp"
        return 1
    }

    chmod +x "$tmp" || { rm -f "$tmp"; return 1; }

    # Confirm the download is a working binary of the right flavour before
    # putting it in place: a truncated file or an HTML error page would
    # otherwise fail much later with a confusing yq syntax error.
    if ! perf_yq_is_usable "$tmp"; then
        echo "[ERROR] downloaded yq is not a usable mikefarah v4 binary:" >&2
        echo "        $("$tmp" --version 2>&1 | head -1)" >&2
        rm -f "$tmp"
        return 1
    fi

    mv -f "$tmp" "$dest/yq" || { rm -f "$tmp"; return 1; }
    echo "[INFO] installed $("$dest/yq" --version 2>&1)" >&2
}

# Make sure a usable yq is on PATH, installing one if needed. Prepends the
# cache dir so a previously bootstrapped copy is reused without re-downloading.
perf_ensure_yq() {
    local cached="$PERF_CACHE_DIR/bin"

    perf_yq_is_usable && return 0

    if perf_yq_is_usable "$cached/yq"; then
        PATH="$cached:$PATH"
        export PATH
        return 0
    fi

    if command -v yq >/dev/null 2>&1; then
        echo "[WARN] the yq on PATH is not mikefarah's v4+ ($(yq --version 2>&1 | head -1))." >&2
        echo "       These scripts use v4 expressions; bootstrapping a private copy." >&2
    fi

    perf_install_yq || return 1

    PATH="$cached:$PATH"
    export PATH
    perf_yq_is_usable
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve a config path. Absolute and ~-prefixed paths are taken as-is,
# everything else is relative to the repo root -- because every catalog entry
# is an `examples/...` path.
perf_resolve_path() {
    local p="$1"
    if [ "${p#\~/}" != "$p" ]; then
        # Leading literal "~/" in the catalog entry; expand it ourselves.
        printf '%s\n' "$HOME/${p#\~/}"
    elif [ "${p#/}" != "$p" ]; then
        printf '%s\n' "$p"
    else
        printf '%s\n' "$PRIMUS_PATH/$p"
    fi
}

# ---------------------------------------------------------------------------
# Config selection
#
# Two modes, one default, no other implicit locations -- so "why didn't my
# config run" always has a single answer:
#
#   CONFIG_DIR set  -> directory mode: recursive find, sorted. Files renamed
#                      to *.yaml.done drop out, which is the resume trick.
#   otherwise       -> catalog mode: read CONFIG_FILE (default
#                      tools/perf/configs.yaml), filtered by GPU / BACKEND.
#
# Populates the CONFIG_FILES array with absolute paths and sets
# CONFIG_SOURCE to a human-readable description of where they came from.
# ---------------------------------------------------------------------------

perf_select_configs() {
    CONFIG_FILES=()
    CONFIG_SOURCE=""
    perf_split_backends
    # BACKEND only filters the catalog; directory mode has always ignored it.
    if [ -n "${CONFIG_DIR:-}" ] && [ "$PERF_RUN_DIFFUSION_DLRM" = 1 ]; then
        echo "[WARN] CONFIG_DIR is set, so BACKEND is ignored and $PERF_DIFFUSION_DLRM_BACKEND does not run." >&2
        PERF_RUN_DIFFUSION_DLRM=0
    fi

    if [ -n "${CONFIG_DIR:-}" ] && [ -n "${CONFIG_FILE:-}" ]; then
        echo "[ERROR] set either CONFIG_DIR or CONFIG_FILE, not both." >&2
        echo "        CONFIG_DIR=$CONFIG_DIR" >&2
        echo "        CONFIG_FILE=$CONFIG_FILE" >&2
        return 1
    fi

    if [ -n "${CONFIG_DIR:-}" ]; then
        perf_select_from_dir || return 1
    else
        perf_select_from_catalog || return 1
    fi

    perf_validate_configs
}

# diffusion_dlrm is not a catalog backend: run_batch.sh runs diffusion_bench
# for it after the catalog configs (lib/diffusion_dlrm.sh). Sets
# PERF_RUN_DIFFUSION_DLRM and PERF_CATALOG_BACKENDS (BACKEND without it).
PERF_DIFFUSION_DLRM_BACKEND="diffusion_dlrm"
PERF_RUN_DIFFUSION_DLRM=0
PERF_CATALOG_BACKENDS=()

perf_split_backends() {
    local backend backends=()
    PERF_RUN_DIFFUSION_DLRM=0
    PERF_CATALOG_BACKENDS=()
    IFS=',' read -r -a backends <<< "${BACKEND:-}"
    for backend in ${backends[@]+"${backends[@]}"}; do
        backend="${backend// /}"
        [ -n "$backend" ] || continue
        if [ "$backend" = "$PERF_DIFFUSION_DLRM_BACKEND" ]; then
            PERF_RUN_DIFFUSION_DLRM=1
        else
            PERF_CATALOG_BACKENDS+=("$backend")
        fi
    done
}

perf_select_from_dir() {
    local dir
    dir=$(perf_resolve_path "$CONFIG_DIR")

    if [ ! -d "$dir" ]; then
        echo "[ERROR] CONFIG_DIR does not exist: $dir" >&2
        return 1
    fi

    # Recursive so the directory can be flat or organised by framework.
    # `-name '*.yaml'` matches the basename, so `foo.yaml.done` is skipped:
    # that is the resume convention -- rename a config once it has passed and
    # the next batch leaves it alone.
    local f
    while IFS= read -r -d '' f; do
        CONFIG_FILES+=("$f")
    done < <(find "$dir" -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z)

    CONFIG_SOURCE="directory $dir"

    if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
        echo "[ERROR] no *.yaml files found under: $dir (searched recursively)" >&2
        echo "        Files renamed to *.yaml.done are skipped by design." >&2
        return 1
    fi
}

perf_select_from_catalog() {
    local catalog
    catalog=$(perf_resolve_path "${CONFIG_FILE:-tools/perf/configs.yaml}")

    if [ ! -f "$catalog" ]; then
        echo "[ERROR] config catalog not found: $catalog" >&2
        echo "        Set CONFIG_FILE=<catalog.yaml> or CONFIG_DIR=<dir of yamls>." >&2
        return 1
    fi

    # GPU: explicit wins, otherwise read it off the hardware. You already know
    # which node you are on; the script can too. Only the catalog backends
    # need it to be a catalog key; diffusion_dlrm alone runs on any GPU that
    # diffusion_bench knows (lib/diffusion_dlrm.sh).
    local need_gpu_key=0
    [ ${#PERF_CATALOG_BACKENDS[@]} -gt 0 ] && need_gpu_key=1
    if [ -z "${GPU:-}" ]; then
        GPU=$(perf_detect_gpu)
        if [ -z "$GPU" ] && [ "$need_gpu_key" = 1 ]; then
            echo "[ERROR] GPU is not set and could not be detected from amd-smi or rocm-smi." >&2
            echo "        Set GPU=<key>, one of: $(yq -r 'keys | .[]' "$catalog" 2>/dev/null | tr '\n' ' ')" >&2
            return 1
        fi
        [ -n "$GPU" ] && echo "[INFO] GPU not set; detected $GPU ($(perf_gpu_model))." >&2
    fi

    if [ "$need_gpu_key" = 1 ] && [ "$(yq -r "has(\"$GPU\")" "$catalog" 2>/dev/null)" != "true" ]; then
        echo "[ERROR] GPU '$GPU' is not a key in $catalog" >&2
        echo "        Available: $(yq -r 'keys | .[]' "$catalog" 2>/dev/null | tr '\n' ' ')" >&2
        return 1
    fi

    # BACKEND is required rather than defaulting to every backend, because each
    # training image carries one framework stack only:
    #   primus_the_rock_ci_* (PyTorch) -> megatron, torchtitan
    #   jax_the_rock_ci_*    (JAX)     -> maxtext, maxdiffusion
    # There is no image for which "all" is correct, so a default would only
    # ever queue runs with no framework to execute them.
    # diffusion_dlrm is offered by the single-node runner only.
    local extra_backend=""
    [ "${PERF_MULTINODE:-0}" = 1 ] || extra_backend="$PERF_DIFFUSION_DLRM_BACKEND"

    if [ -z "${BACKEND:-}" ]; then
        echo "[ERROR] BACKEND is required; each training image supports one framework stack." >&2
        echo "        Available for $GPU: $(yq -r ".[\"$GPU\"] | keys | .[]" "$catalog" 2>/dev/null | tr '\n' ' ')$extra_backend" >&2
        echo "        PyTorch image: BACKEND=megatron,torchtitan,megatron_bridge${extra_backend:+,$extra_backend}" >&2
        echo "        JAX image    : BACKEND=maxtext,maxdiffusion" >&2
        return 1
    fi

    local backend entry

    for backend in ${PERF_CATALOG_BACKENDS[@]+"${PERF_CATALOG_BACKENDS[@]}"}; do
        if [ "$(yq -r ".[\"$GPU\"] | has(\"$backend\")" "$catalog" 2>/dev/null)" != "true" ]; then
            echo "[ERROR] backend '$backend' is not listed under $GPU in $catalog" >&2
            echo "        Available for $GPU: $(yq -r ".[\"$GPU\"] | keys | .[]" "$catalog" 2>/dev/null | tr '\n' ' ')$extra_backend" >&2
            return 1
        fi
        while IFS= read -r entry; do
            [ -n "$entry" ] || continue
            CONFIG_FILES+=("$(perf_resolve_path "$entry")")
        done < <(yq -r ".[\"$GPU\"][\"$backend\"][]" "$catalog" 2>/dev/null)
    done

    # Consumed by the runner for the batch summary.
    # shellcheck disable=SC2034
    CONFIG_SOURCE="catalog $catalog (GPU=$GPU BACKEND=$BACKEND)"

    if [ ${#CONFIG_FILES[@]} -eq 0 ] && [ "$PERF_RUN_DIFFUSION_DLRM" != 1 ]; then
        # yq returns empty with exit 0 for an all-commented list, a missing
        # key and a typo'd key alike, so say exactly what was looked for.
        echo "[ERROR] no configs selected from: $catalog" >&2
        echo "        GPU=$GPU BACKEND=$BACKEND" >&2
        echo "        Every entry under those keys may be commented out." >&2
        return 1
    fi
}

# Reject missing paths and duplicates before anything launches. A duplicate
# matters because two identical entries produce the same log filename (same
# config hash, same rep) and the second would silently overwrite the first.
perf_validate_configs() {
    local bad=0 f seen_dupes

    # Guarded expansions: with only diffusion_dlrm selected the list is empty,
    # which bash < 4.4 treats as unset under set -u.
    for f in ${CONFIG_FILES[@]+"${CONFIG_FILES[@]}"}; do
        if [ ! -f "$f" ]; then
            echo "[ERROR] config does not exist: $f" >&2
            bad=$((bad + 1))
        fi
    done

    seen_dupes=$(printf '%s\n' ${CONFIG_FILES[@]+"${CONFIG_FILES[@]}"} | sort | uniq -d)
    if [ -n "$seen_dupes" ]; then
        echo "[ERROR] duplicate config entries (each would overwrite the other's log):" >&2
        while IFS= read -r f; do
            [ -n "$f" ] && echo "          $f" >&2
        done <<< "$seen_dupes"
        bad=$((bad + 1))
    fi

    if [ "$bad" -ne 0 ]; then
        echo "[ERROR] $bad problem(s) in the selected config set; nothing was launched." >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Extra container environment
#
# Exporting a variable in this shell does NOT necessarily reach the training
# container. primus-cli forwards only:
#   * names listed in runner/.primus.yaml under container.options.env, and
#   * anything matching PRIMUS_ NCCL_ RCCL_ GLOO_ IONIC_ HIPBLASLT_.
# Everything else is dropped silently -- an exported DEBUG_HIP_DYNAMIC_QUEUES=0
# never arrives, and nothing warns you.
#
# EXTRA_ENV is the reliable channel: each entry becomes an explicit
# `--env KEY=VALUE` on the launcher, which primus-cli always honours.
#
#   EXTRA_ENV="DEBUG_HIP_DYNAMIC_QUEUES=0 GPU_MAX_HW_QUEUES=2"
#
# Space-separated, so values themselves cannot contain spaces.
# ---------------------------------------------------------------------------

PERF_EXTRA_ENV_ARGS=()

perf_parse_extra_env() {
    PERF_EXTRA_ENV_ARGS=()
    local kv kvs=()

    # read -a splits on whitespace without glob expansion, so a value such as
    # `FOO=a*` is kept as typed rather than matched against files.
    read -r -a kvs <<< "${EXTRA_ENV:-}"
    for kv in ${kvs[@]+"${kvs[@]}"}; do
        if [[ "$kv" != *=* ]] || [[ "$kv" != [A-Za-z_]* ]]; then
            echo "[ERROR] EXTRA_ENV entries must look like KEY=VALUE, got: '$kv'" >&2
            echo "        Example: EXTRA_ENV=\"DEBUG_HIP_DYNAMIC_QUEUES=0 GPU_MAX_HW_QUEUES=2\"" >&2
            return 1
        fi
        # Each entry is also exported here, so PERF_* would rewrite the
        # runner's own state (e.g. PERF_MULTINODE) rather than the container's.
        if [[ "$kv" == PERF_* ]]; then
            echo "[ERROR] EXTRA_ENV must not set the runner's internal PERF_* variables, got: '$kv'" >&2
            return 1
        fi
        PERF_EXTRA_ENV_ARGS+=(--env "$kv")
        # Also set it in this shell, so the banner and env snapshot record what
        # was requested and direct (non-container) mode picks it up too.
        export "${kv?}"
    done
}

# ---------------------------------------------------------------------------
# Provenance
#
# Everything here answers "which settings produced this number". The values
# go into the per-run banner (which the extractor already parses as
# `# key : value`) and into batch-level snapshot files.
# ---------------------------------------------------------------------------

# Env prefixes that can move performance. Captured wholesale rather than as a
# hand-picked list, because the hand-picked list is always out of date.
PERF_ENV_PREFIXES=(
    HSA_ HIP_ ROCM_ ROCR_ GPU_ AMD_ NCCL_ RCCL_ XLA_ JAX_ LIBTPU_
    TORCH_ PYTORCH_ NVTE_ TE_ MIOPEN_ TRITON_ OMP_ PRIMUS_
)

# Anything whose name matches this is never written to a snapshot.
PERF_SECRET_RE='(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_KEY|APIKEY|API_KEY|AWS_)'

perf_env_regex() {
    local IFS='|'
    printf '^(%s)' "${PERF_ENV_PREFIXES[*]}"
}

# Perf-relevant env, one KEY=VALUE per line, secrets redacted.
perf_env_lines() {
    env | grep -E "$(perf_env_regex)" 2>/dev/null | sort |
        awk -F= -v re="$PERF_SECRET_RE" '{ if ($1 ~ re) print $1 "=<redacted>"; else print }'
}

# Full environment with secrets redacted, for the batch env snapshot.
perf_env_redacted() {
    env | sort | awk -F= -v re="$PERF_SECRET_RE" '{ if ($1 ~ re) print $1 "=<redacted>"; else print }'
}

# Filter for the launcher's own stdout/stderr, masking secret *values*.
#
# Redacting our env snapshot is not enough: primus-cli logs the container
# command it is about to run, which includes `--env HF_TOKEN=<value>`, so
# without this every per-run log ends up holding a live credential.
#
# Values shorter than 8 characters are skipped -- masking those would riddle
# the log with false positives.
# A command line for display (banner, summary, DRY_RUN): the value of any
# NAME=VALUE argument whose NAME looks like a secret is masked, as in the env
# snapshot. EXTRA_ENV="WANDB_API_KEY=..." would otherwise be logged in clear.
perf_display_argv() {
    local arg shown=""
    for arg in "$@"; do
        if [[ "$arg" == *=* && "${arg%%=*}" =~ $PERF_SECRET_RE ]]; then
            arg="${arg%%=*}=<redacted>"
        fi
        shown+="${shown:+ }$arg"
    done
    printf '%s\n' "$shown"
}

perf_redact_stream() {
    # awk reads the values straight out of ENVIRON, so no secret is ever passed
    # on a command line where `ps` could see it. Matching is literal
    # (index/substr rather than a regex) because secret values contain
    # characters that regex escaping gets wrong -- notably `+`, which BRE turns
    # into a metacharacter when backslash-escaped rather than escaping it.
    awk '
        BEGIN {
            n = 0
            for (name in ENVIRON) {
                if (name ~ /(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|APIKEY|API_KEY)/ ||
                    name ~ /_KEY$/ || name ~ /^AWS_/) {
                    v = ENVIRON[name]
                    # Short values would riddle the log with false positives.
                    if (length(v) >= 8) { n++; names[n] = name; vals[n] = v }
                }
            }
        }
        {
            line = $0
            for (i = 1; i <= n; i++) {
                out = ""
                while ((p = index(line, vals[i])) > 0) {
                    out = out substr(line, 1, p - 1) "<redacted:" names[i] ">"
                    line = substr(line, p + length(vals[i]))
                }
                line = out line
            }
            print line
            # Keep the stream live so progress still scrolls during a run.
            fflush()
        }
    '
}

# Immutable image identity. A tag can be re-pushed; the digest cannot.
perf_image_digest() {
    local image="${1:-}"
    [ -n "$image" ] || { echo "unknown (DOCKER_IMAGE unset)"; return 0; }
    command -v docker >/dev/null 2>&1 || { echo "unknown (docker not on PATH)"; return 0; }

    local digest
    digest=$(docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' \
        "$image" 2>/dev/null)
    if [ -n "$digest" ]; then
        echo "$digest"
    else
        echo "unknown (image not present locally)"
    fi
}

# Local image ID. Unlike the repo digest it does not depend on which registry
# the image was pulled from, so two runs (or two nodes) used the same image
# exactly when their IDs match -- also for an image that was re-tagged locally.
perf_image_id() {
    local image="${1:-}" id
    [ -n "$image" ] || { echo "unknown (DOCKER_IMAGE unset)"; return 0; }
    command -v docker >/dev/null 2>&1 || { echo "unknown (docker not on PATH)"; return 0; }
    id=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)
    echo "${id:-unknown (image not present locally)}"
}

# Primus HEAD, flagged when tracked files are modified. Untracked files are
# ignored on purpose: a bench checkout is usually full of result directories,
# and counting those would mark every run dirty.
perf_primus_commit() {
    local commit dirty
    commit=$(git -C "$PRIMUS_PATH" rev-parse --short=12 HEAD 2>/dev/null) || {
        echo "unknown (not a git checkout)"
        return 0
    }
    dirty=$(git -C "$PRIMUS_PATH" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
    if [ "$dirty" -gt 0 ]; then
        echo "${commit}-dirty"
    else
        echo "$commit"
    fi
}

perf_submodule_status() {
    git -C "$PRIMUS_PATH" submodule status 2>/dev/null || true
}

# Short hash over the submodule pins. The Primus commit alone does not
# identify the code that ran: a submodule bump changes performance while
# leaving HEAD untouched.
perf_submodule_pins() {
    local status
    status=$(perf_submodule_status)
    [ -n "$status" ] || { echo "none"; return 0; }
    printf '%s' "$status" | sha256sum | cut -c1-8
}

# Marketing name of the first GPU, e.g. "AMD Instinct MI355X". amd-smi first:
# some nodes ship only amd-smi. Its PRODUCT_NAME drops the suffix
# ("AMD Instinct MI355 OAM"), so MARKET_NAME is the field to read.
perf_gpu_model() {
    local model=""
    if command -v amd-smi >/dev/null 2>&1; then
        model=$(amd-smi static --asic 2>/dev/null |
            sed -n 's/.*MARKET_NAME:[[:space:]]*//p' | head -1 |
            sed 's/[[:space:]]*$//')
    fi
    if { [ -z "$model" ] || [ "$model" = "N/A" ]; } && command -v rocm-smi >/dev/null 2>&1; then
        model=$(rocm-smi --showproductname 2>/dev/null |
            sed -n 's/.*Card Series:[[:space:]]*//p' | head -1 |
            sed 's/[[:space:]]*$//')
    fi
    if [ -n "$model" ] && [ "$model" != "N/A" ]; then
        echo "$model"
    else
        echo "unknown"
    fi
}

perf_gpu_count() {
    local n=0
    if command -v amd-smi >/dev/null 2>&1; then
        n=$(amd-smi list 2>/dev/null | grep -c '^GPU:')
    fi
    if [ "$n" -eq 0 ] && command -v rocm-smi >/dev/null 2>&1; then
        n=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[')
    fi
    if [ "$n" -gt 0 ]; then
        echo "$n"
    else
        echo "unknown"
    fi
}

# Catalog key for the hardware this node actually has, e.g. "MI325X".
#
# Derived from the rocm-smi product name ("AMD Instinct MI325X") rather than
# asked of the user, since the node already knows. Returns empty when there is
# no rocm-smi or the name does not carry a recognisable part number, and the
# caller turns that into a "set GPU=" error.
perf_detect_gpu() {
    local model key
    model=$(perf_gpu_model)
    if [ -z "$model" ] || [ "$model" = "unknown" ]; then
        echo ""
        return 0
    fi

    key=$(printf '%s\n' "$model" | grep -oiE 'MI[0-9]{3}X?' | head -1 | tr '[:lower:]' '[:upper:]')
    echo "$key"
}

# Host ROCm version. ROCm is not always under /opt/rocm (some clusters load it
# as a module and set ROCM_PATH), and recent rocminfo no longer prints it.
perf_rocm_version() {
    local v="" f
    for f in "${ROCM_PATH:+$ROCM_PATH/.info/version}" /opt/rocm/.info/version; do
        if [ -n "$f" ] && [ -r "$f" ]; then
            v=$(head -1 "$f")
            [ -n "$v" ] && break
        fi
    done
    if [ -z "$v" ] && command -v amd-smi >/dev/null 2>&1; then
        v=$(amd-smi version 2>/dev/null | sed -n 's/.*ROCm version: *\([^ |]*\).*/\1/p' | head -1)
    fi
    if [ -z "$v" ] && command -v rocminfo >/dev/null 2>&1; then
        v=$(rocminfo 2>/dev/null | awk '/ROCm Version/ {print $3; exit}')
    fi
    echo "${v:-unknown}"
}

perf_gpu_snapshot() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showproductname --showclocks --showpower --showmaxpower 2>&1 || true
        echo
        echo "== topology =="
        rocm-smi --showtopo 2>&1 || true
    elif command -v amd-smi >/dev/null 2>&1; then
        amd-smi static 2>&1 || true
        echo
        echo "== power / clocks / temperature at batch start =="
        amd-smi metric --power --clock --temperature 2>&1 || amd-smi metric 2>&1 || true
    else
        echo "no rocm-smi / amd-smi on PATH"
    fi
}

# ---------------------------------------------------------------------------
# Host platform
#
# Everything about a node that can move a number without showing up in the
# image or the code: what the server is (vendor, model, BIOS), the CPUs'
# topology, frequency and idle states, memory, the GPUs as the driver reports
# them, the host GPU driver, RDMA NICs, OS, kernel command line and the host
# settings. All of it is read without root.
#
# The functions in PERF_HOST_FUNCS are shipped to remote nodes as source text
# for the multi-node snapshot, so they must stay self-contained: they call
# only each other and standard tools.
# ---------------------------------------------------------------------------

# First line of a sysfs or procfs file, trailing blanks removed; nothing when
# it is missing or unreadable (DMI serial numbers, for one, are root-only).
perf_host_read() {
    [ -r "$1" ] && head -n 1 "$1" 2>/dev/null | sed 's/[[:space:]]*$//'
}

# The distinct lines of stdin, most common first: "a" when they all agree,
# "a x6, b x2" when they do not.
perf_host_tally() {
    sed '/^$/d' | sort | uniq -c | sort -s -k1,1nr | awk '
        { n[NR] = $1; sub(/^[[:space:]]*[0-9]+[[:space:]]/, ""); v[NR] = $0 }
        END {
            if (NR == 1) { print v[1]; exit }
            for (i = 1; i <= NR; i++) printf "%s%s x%s", (i > 1 ? ", " : ""), v[i], n[i]
            if (NR) print ""
        }'
}

# "<key>: <tally of the newline-separated values>", "unknown" without any.
perf_host_field() {
    local value
    value=$(printf '%s\n' "$2" | perf_host_tally)
    printf '%s: %s\n' "$1" "${value:-unknown}"
}

# A DMI field, unless it holds one of the placeholders vendors leave in the
# fields they do not fill.
perf_host_dmi() {
    perf_host_read "/sys/class/dmi/id/$1" |
        grep -vixE '(to be filled by o\.e\.m\.|default string|system product name|system manufacturer|not specified|not applicable|none|n/a|unknown|0123456789)?'
}

# The host GPU driver from its package version, e.g. "1:6.14.14.30100100-
# 2212064.22.04", whose last upstream field is the driver release: 30100100
# is amdgpu 30.10.1. Before that numbering it was the ROCm release (60401 for
# ROCm 6.4.1). <rocm> is the ROCm release of the host's package repository,
# which amdgpu-install points at the release the driver ships with. Without a
# package that decodes, the loaded module's version.
perf_host_gpu_driver() {
    local pkg="$1" rocm="$2" release
    release=${pkg#*:}
    release=${release%%-*}
    release=${release##*.}
    case "$release" in
        [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9])
            printf 'amdgpu %d.%d.%d' "$((10#${release:0:2}))" "$((10#${release:2:2}))" "$((10#${release:4:2}))"
            [ "${release:6:2}" = 00 ] || printf '.%d' "$((10#${release:6:2}))"
            echo "${rocm:+ (ROCm $rocm)}"
            ;;
        [0-9][0-9][0-9][0-9][0-9])
            echo "ROCm ${release:0:1}.$((10#${release:1:2})).$((10#${release:3:2}))"
            ;;
        *)
            if [ -n "$pkg" ]; then
                echo "amdgpu-dkms $pkg"
            elif [ -r /sys/module/amdgpu/version ]; then
                echo "amdgpu $(perf_host_read /sys/module/amdgpu/version)"
            elif [ -d /sys/module/amdgpu ]; then
                echo "amdgpu inbox (kernel $(uname -r))"
            else
                echo "not loaded"
            fi
            ;;
    esac
}

# Host settings that change performance without showing up anywhere else,
# one "key: value" per line, "unknown" for a value that cannot be read.
perf_host_settings() {
    local f governors cmdline iommu_type thp defrag driver
    # A loop rather than `cat <glob>`: under the runners' nullglob an
    # unmatched glob would leave cat reading stdin, and the batch would hang.
    governors=$(for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -r "$f" ] && cat "$f"
    done | sort | uniq -c | awk '{ printf "%s%s x%s", sep, $2, $1; sep = ", " }')
    cmdline=$(grep -oE '(^| )(amd_|intel_)?iommu=[^ ]+' /proc/cmdline 2>/dev/null | sed 's/^ //' | paste -sd ' ')
    iommu_type=$(cat /sys/kernel/iommu_groups/0/type 2>/dev/null </dev/null)
    thp=$(sed -n 's/.*\[\(.*\)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
    defrag=$(sed -n 's/.*\[\(.*\)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)
    # The DKMS driver reports a version; the in-kernel one does not.
    if [ -r /sys/module/amdgpu/version ]; then
        driver=$(cat /sys/module/amdgpu/version </dev/null)
    elif [ -d /sys/module/amdgpu ]; then
        driver="inbox (kernel $(uname -r))"
    else
        driver="not loaded"
    fi
    echo "cpu_governor: ${governors:-unknown}"
    echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null </dev/null || echo unknown)"
    echo "transparent_hugepages: ${thp:-unknown}"
    echo "thp_defrag: ${defrag:-unknown}"
    echo "iommu: ${cmdline:-not set on the kernel command line}${iommu_type:+ (default domain: $iommu_type)}"
    echo "amdgpu_driver: $driver"
    echo "kernel: $(uname -r)"
    echo "cpu_model: $(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
    # --all: inside an srun step plain nproc counts only the step's CPUs.
    echo "cpus: $(nproc --all 2>/dev/null)"
    echo "memory: $(awk '/^MemTotal/ { printf "%.0f GiB", $2 / 1048576 }' /proc/meminfo 2>/dev/null)"
    perf_host_platform_settings
    perf_host_gpu_settings
}

# What the server is: OS and kernel command line, DMI vendor, model and BIOS,
# CPU topology, frequency and idle states, memory and the active RDMA ports.
perf_host_platform_settings() {
    local f d os cmdline vendor product board bios extra lscpu_out max_mhz freq status boost epp
    local idle idle_gov name enabled="" disabled="" installed dimms nics

    os=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | tr -d '"')
    echo "os: ${os:-unknown}"
    # BOOT_IMAGE, root and initrd name boot files rather than settings.
    cmdline=$(tr ' ' '\n' 2>/dev/null </proc/cmdline | grep -vE '^$|^(BOOT_IMAGE|root|initrd)=' | paste -sd ' ')
    echo "kernel_cmdline: ${cmdline:-unknown}"

    vendor=$(perf_host_dmi sys_vendor)
    product=$(perf_host_dmi product_name)
    # Some vendors repeat their name in the product name.
    [ -n "$product" ] && [ "${product#"$vendor"}" != "$product" ] && vendor=""
    product="${vendor:+$vendor }$product"
    echo "server: ${product:-unknown}"
    board=$(printf '%s %s %s' "$(perf_host_dmi board_vendor)" "$(perf_host_dmi board_name)" \
        "$(perf_host_dmi board_version)" | tr -s ' ' | sed 's/^ //; s/ $//')
    echo "board: ${board:-unknown}"
    bios=$(perf_host_dmi bios_version)
    extra=$(printf '%s, %s' "$(perf_host_dmi bios_date)" "$(perf_host_dmi bios_vendor)" | sed 's/^, //; s/, $//')
    echo "bios: ${bios:-unknown}${extra:+ ($extra)}"

    lscpu_out=$(LC_ALL=C lscpu 2>/dev/null </dev/null)
    echo "cpu_sockets: $(sed -n 's/^Socket(s):[[:space:]]*//p' <<< "$lscpu_out" | grep . || echo unknown)"
    echo "cpu_cores_per_socket: $(sed -n 's/^Core(s) per socket:[[:space:]]*//p' <<< "$lscpu_out" | grep . || echo unknown)"
    echo "cpu_threads_per_core: $(sed -n 's/^Thread(s) per core:[[:space:]]*//p' <<< "$lscpu_out" | grep . || echo unknown)"
    echo "numa_nodes: $(sed -n 's/^NUMA node(s):[[:space:]]*//p' <<< "$lscpu_out" | grep . || echo unknown)"

    freq=$(perf_host_read /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver)
    # amd-pstate's mode: active, passive or guided.
    status=$(perf_host_read /sys/devices/system/cpu/amd_pstate/status)
    boost=$(perf_host_read /sys/devices/system/cpu/cpufreq/boost)
    [ -n "$boost" ] || boost=$(perf_host_read /sys/devices/system/cpu/cpufreq/policy0/boost)
    case "$boost" in
        1) boost=on ;;
        0) boost=off ;;
        *) boost=$(sed -n 's/^Frequency boost:[[:space:]]*//p' <<< "$lscpu_out" | sed 's/^enabled$/on/; s/^disabled$/off/') ;;
    esac
    max_mhz=$(sed -n 's/^CPU max MHz:[[:space:]]*//p' <<< "$lscpu_out")
    max_mhz=${max_mhz%%.*}
    epp=$(for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
        [ -r "$f" ] && cat "$f"
    done | perf_host_tally)
    echo "cpu_frequency: ${freq:-no cpufreq driver}${status:+ ($status)}, boost ${boost:-unknown}${max_mhz:+, max $max_mhz MHz}${epp:+, EPP $epp}"

    # cpu0's idle states; disabling one is a global setting in practice.
    idle=$(perf_host_read /sys/devices/system/cpu/cpuidle/current_driver)
    idle_gov=$(perf_host_read /sys/devices/system/cpu/cpuidle/current_governor)
    [ -n "$idle_gov" ] || idle_gov=$(perf_host_read /sys/devices/system/cpu/cpuidle/current_governor_ro)
    for d in /sys/devices/system/cpu/cpu0/cpuidle/state[0-9]*; do
        name=$(perf_host_read "$d/name") || continue
        if [ "$(perf_host_read "$d/disable")" = 1 ]; then
            disabled+=" $name"
        else
            enabled+=" $name"
        fi
    done
    echo "cpu_idle: ${idle:-none}${idle_gov:+ ($idle_gov)}${enabled:+:$enabled}${disabled:+; disabled:$disabled}"

    # EDAC lists every DIMM rank, so its sizes add up to the installed memory;
    # amd64_edac's labels (mc#<n>csrow#<rank>channel#<n>) also give the
    # populated channels.
    installed=$(for f in /sys/devices/system/edac/mc/mc*/rank*/size /sys/devices/system/edac/mc/mc*/dimm*/size; do
        [ -r "$f" ] || continue
        d=${f%/size}
        printf '%s\t%s\t%s\n' "$(perf_host_read "$f")" "$(perf_host_read "$d/dimm_mem_type")" "$(perf_host_read "$d/dimm_label")"
    done | awk -F '\t' '
        $1 > 0 {
            mib += $1
            if ($2 != "") type[$2] = 1
            if ($3 ~ /^mc#[0-9]+csrow#[0-9]+channel#[0-9]+$/) { c = $3; sub(/csrow#[0-9]+/, "", c); channels[c] = 1 }
        }
        END {
            if (!mib) exit
            for (t in type) types = types (types == "" ? " " : "/") t
            for (c in channels) n++
            printf "%.0f GiB%s%s\n", mib / 1024, types, (n ? ", " n " channels" : "")
        }')
    echo "memory_installed: ${installed:-unknown}"
    # The DMI memory records udev keeps (no root needed). On some hosts that
    # list is incomplete, so it names the modules but does not count them.
    dimms=$(udevadm info --query=property --path=/sys/devices/virtual/dmi/id 2>/dev/null </dev/null | awk '
        match($0, /^MEMORY_DEVICE_[0-9]+_/) {
            id = substr($0, 15, RLENGTH - 15); rest = substr($0, RLENGTH + 1); eq = index(rest, "=")
            value = substr(rest, eq + 1); sub(/[[:space:]]+$/, "", value)
            if (tolower(value) ~ /^(not specified|unknown|undefined|none)$/) value = ""
            v[id, substr(rest, 1, eq - 1)] = value; ids[id] = 1
        }
        END {
            for (id in ids) {
                if (v[id, "SIZE"] + 0 == 0) continue
                s = sprintf("%.0f GiB", v[id, "SIZE"] / 1073741824)
                if (v[id, "MANUFACTURER"] != "") s = s " " v[id, "MANUFACTURER"]
                if (v[id, "PART_NUMBER"] != "") s = s " " v[id, "PART_NUMBER"]
                if (v[id, "SPEED_MTS"] != "") s = s ", " v[id, "SPEED_MTS"] " MT/s"
                if (v[id, "CONFIGURED_SPEED_MTS"] != "" && v[id, "CONFIGURED_SPEED_MTS"] != v[id, "SPEED_MTS"])
                    s = s ", configured " v[id, "CONFIGURED_SPEED_MTS"] " MT/s"
                print s
            }
        }' | sort -u | paste -sd ';' | sed 's/;/; /g')
    echo "memory_dimms: ${dimms:-unknown}"

    nics=$(for d in /sys/class/infiniband/*; do
        for f in "$d"/ports/*/state; do
            [ "$(perf_host_read "$f")" = "4: ACTIVE" ] || continue
            printf '%s %s (fw %s)\n' "$(basename "$(readlink -f "$d/device/driver")")" \
                "$(perf_host_read "${f%/state}/rate" | sed 's/ (.*//')" "$(perf_host_read "$d/fw_ver")"
        done
    done | sort | uniq -c | sort -s -k1,1nr |
        awk '{ n = $1; sub(/^[[:space:]]*[0-9]+[[:space:]]/, ""); printf "%s%sx %s", (NR > 1 ? ", " : ""), n, $0 }
             END { if (NR) print "" }')
    echo "rdma_nics: ${nics:-none}"
}

# The GPUs as the amdgpu driver reports them in sysfs, which any job step can
# read (amd-smi only sees the GPUs a step was given), and the host GPU driver.
# A field the GPUs disagree on is a tally, e.g. "1400 W x7, 1000 W x1".
perf_host_gpu_settings() {
    local d f dev seen=" " n=0 value cap def cp mp sclk cur max pkg="" rocm=""
    local product="" vram="" caps="" partitions="" clocks="" vbios="" links=""
    for d in /sys/class/drm/card*/device; do
        [ -r "$d/mem_info_vram_total" ] || continue
        dev=$(readlink -f "$d")
        # A partitioned GPU (CPX, NPS4) adds cards on the same device.
        case "$seen" in *" $dev "*) continue ;; esac
        seen+="$dev "
        n=$((n + 1))
        product+="$(perf_host_read "$d/product_name")"$'\n'
        vram+="$(awk '{ printf "%.0f GiB", $1 / 1073741824 }' "$d/mem_info_vram_total" 2>/dev/null)"$'\n'
        cap="" def=""
        for f in "$d"/hwmon/hwmon*/power1_cap; do
            cap=$(perf_host_read "$f")
            def=$(perf_host_read "${f}_default")
        done
        if [[ "$cap" =~ ^[0-9]+$ ]]; then
            value="$((cap / 1000000)) W"
            if [[ "$def" =~ ^[0-9]+$ ]] && [ "$def" != "$cap" ]; then
                value+=" (default $((def / 1000000)) W)"
            fi
            caps+="$value"$'\n'
        fi
        cp=$(perf_host_read "$d/current_compute_partition")
        mp=$(perf_host_read "$d/current_memory_partition")
        [ -n "$cp$mp" ] && partitions+="${cp:-?}/${mp:-?}"$'\n'
        # A lowered top level is how a clock limit (e.g. perf determinism) shows.
        sclk=$(grep -oE '[0-9]+ ?[Mm][Hh]z' "$d/pp_dpm_sclk" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -n 1)
        clocks+="perf level $(perf_host_read "$d/power_dpm_force_performance_level" | grep . || echo unknown)${sclk:+, max sclk $sclk MHz}"$'\n'
        vbios+="$(perf_host_read "$d/vbios_version")"$'\n'
        if [ -r "$d/current_link_speed" ]; then
            cur="$(perf_host_read "$d/current_link_speed") x$(perf_host_read "$d/current_link_width")"
            max="$(perf_host_read "$d/max_link_speed") x$(perf_host_read "$d/max_link_width")"
            [ "$cur" = "$max" ] || cur+=" (max $max)"
            links+="$cur"$'\n'
        fi
    done
    echo "gpu_count: $n"
    perf_host_field gpu_product "$product"
    perf_host_field gpu_vram "$vram"
    perf_host_field gpu_power_cap "$caps"
    perf_host_field gpu_partition "$partitions"
    perf_host_field gpu_clocks "$clocks"
    perf_host_field gpu_vbios "$vbios"
    perf_host_field gpu_pcie "$links"

    # ${...} here is dpkg-query's syntax, not the shell's.
    # shellcheck disable=SC2016
    if command -v dpkg-query >/dev/null 2>&1; then
        pkg=$(dpkg-query -W -f='${db:Status-Abbrev} ${Version}\n' amdgpu-dkms 2>/dev/null | awk '$1 == "ii" { print $2 }')
    fi
    if [ -z "$pkg" ] && command -v rpm >/dev/null 2>&1; then
        pkg=$(rpm -q --qf '%{VERSION}-%{RELEASE}\n' amdgpu-dkms 2>/dev/null | grep -v 'not installed')
    fi
    rocm=$(for f in /etc/apt/sources.list /etc/apt/sources.list.d/* /etc/yum.repos.d/* /etc/zypp/repos.d/*; do
        [ -f "$f" ] && grep -vE '^[[:space:]]*#' "$f" 2>/dev/null
    done | grep -oE 'repo\.radeon\.com/rocm/[^[:space:]]+' | grep -oE '/[0-9]+(\.[0-9]+)+' | head -n 1 | tr -d /)
    echo "gpu_driver: $(perf_host_gpu_driver "$pkg" "$rocm")"
    echo "gpu_driver_package: ${pkg:+amdgpu-dkms }${pkg:-unknown}"
}

# The raw sources behind perf_host_settings, for the questions its one-line
# values do not answer. Every section starts with a "-- <name> --" line.
perf_host_details() {
    local d f dev net seen=" "
    echo "-- lscpu --"
    LC_ALL=C lscpu 2>&1 </dev/null
    echo "-- numa --"
    if command -v numactl >/dev/null 2>&1; then
        numactl --hardware 2>&1 </dev/null
    else
        for d in /sys/devices/system/node/node[0-9]*; do
            [ -r "$d/cpulist" ] || continue
            echo "${d##*/}: cpus $(perf_host_read "$d/cpulist"), $(awk '/MemTotal/ { printf "%.0f GiB", $4 / 1048576 }' "$d/meminfo" 2>/dev/null)"
        done
    fi
    echo "-- cpu idle states (cpu0) --"
    for d in /sys/devices/system/cpu/cpu0/cpuidle/state[0-9]*; do
        [ -r "$d/name" ] || continue
        echo "$(perf_host_read "$d/name"): latency $(perf_host_read "$d/latency") us, disabled $(perf_host_read "$d/disable"), $(perf_host_read "$d/desc")"
    done
    echo "-- memory --"
    grep -E '^(MemTotal|SwapTotal|HugePages_Total|Hugepagesize):' /proc/meminfo 2>/dev/null
    echo "vm.zone_reclaim_mode = $(perf_host_read /proc/sys/vm/zone_reclaim_mode)"
    echo "vm.swappiness = $(perf_host_read /proc/sys/vm/swappiness)"
    echo "-- dimms (udev's DMI records, which can miss some; serial numbers left out) --"
    udevadm info --query=property --path=/sys/devices/virtual/dmi/id 2>/dev/null </dev/null | awk '
        match($0, /^MEMORY_DEVICE_[0-9]+_/) {
            id = substr($0, 15, RLENGTH - 15); rest = substr($0, RLENGTH + 1); eq = index(rest, "=")
            v[id, substr(rest, 1, eq - 1)] = substr(rest, eq + 1); ids[id] = 1
        }
        END {
            for (id in ids) {
                if (v[id, "SIZE"] + 0 == 0) continue
                printf "%d\t%s: %.0f GiB, %s, %s MT/s (configured %s), %s %s, rank %s\n", id, v[id, "LOCATOR"],
                    v[id, "SIZE"] / 1073741824, v[id, "TYPE_DETAIL"], v[id, "SPEED_MTS"],
                    v[id, "CONFIGURED_SPEED_MTS"], v[id, "MANUFACTURER"], v[id, "PART_NUMBER"], v[id, "RANK"]
            }
        }' | sort -n | cut -f 2-
    echo "-- gpus (sysfs) --"
    for d in /sys/class/drm/card*/device; do
        [ -r "$d/mem_info_vram_total" ] || continue
        dev=$(readlink -f "$d")
        case "$seen" in *" $dev "*) continue ;; esac
        seen+="$dev "
        echo "${dev##*/}: $(perf_host_read "$d/product_name"), device $(perf_host_read "$d/device"), vbios $(perf_host_read "$d/vbios_version"), numa node $(perf_host_read "$d/numa_node"), vram $(perf_host_read "$d/mem_info_vram_total") B"
        echo "    partition $(perf_host_read "$d/current_compute_partition")/$(perf_host_read "$d/current_memory_partition"), available $(perf_host_read "$d/available_compute_partition") / $(perf_host_read "$d/available_memory_partition")"
        echo "    pcie $(perf_host_read "$d/current_link_speed") x$(perf_host_read "$d/current_link_width"), max $(perf_host_read "$d/max_link_speed") x$(perf_host_read "$d/max_link_width")"
        for f in "$d"/hwmon/hwmon*/power1_cap; do
            echo "    power cap $(perf_host_read "$f") uW, default $(perf_host_read "${f}_default"), max $(perf_host_read "${f}_max")"
        done
        echo "    perf level $(perf_host_read "$d/power_dpm_force_performance_level")"
        for f in pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk; do
            [ -r "$d/$f" ] && echo "    $f: $(tr '\n' ' ' 2>/dev/null <"$d/$f")"
        done
    done
    echo "-- rdma --"
    for d in /sys/class/infiniband/*; do
        [ -d "$d" ] || continue
        net=""
        for f in "$d"/device/net/*; do
            [ -e "$f" ] && net=${f##*/} && break
        done
        for f in "$d"/ports/*/state; do
            [ -r "$f" ] || continue
            echo "${d##*/} port $(basename "${f%/state}"): $(basename "$(readlink -f "$d/device/driver")") ${net:-?}, $(perf_host_read "${f%/state}/rate"), $(perf_host_read "$f"), $(perf_host_read "${f%/state}/link_layer"), fw $(perf_host_read "$d/fw_ver")"
        done
    done
    echo "-- amdgpu module parameters --"
    for f in /sys/module/amdgpu/parameters/*; do
        [ -r "$f" ] && printf '%s=%s\n' "${f##*/}" "$(perf_host_read "$f")"
    done | paste -sd ' '
    echo "-- driver packages and repositories --"
    # shellcheck disable=SC2016
    if command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W -f='${db:Status-Abbrev} ${Package} ${Version}\n' 'amdgpu*' 2>/dev/null | awk '$1 == "ii" { print $2, $3 }'
    elif command -v rpm >/dev/null 2>&1; then
        rpm -qa 'amdgpu*' 2>/dev/null
    fi
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/* /etc/yum.repos.d/* /etc/zypp/repos.d/*; do
        [ -f "$f" ] && grep -H 'repo\.radeon\.com' "$f" 2>/dev/null | grep -vE '^[^:]*:[[:space:]]*#'
    done
    echo "-- docker --"
    if command -v docker >/dev/null 2>&1; then
        timeout 10 docker version --format 'client {{.Client.Version}}, server {{.Server.Version}}' 2>&1 </dev/null
        timeout 10 docker info --format 'storage driver {{.Driver}}, cgroup v{{.CgroupVersion}}, default runtime {{.DefaultRuntime}}' 2>&1 </dev/null
    else
        echo "(no docker command)"
    fi
    echo "-- interfaces --"
    ip -br addr 2>/dev/null || echo "(no ip command)"
}

PERF_HOST_FUNCS=(
    perf_host_read perf_host_tally perf_host_field perf_host_dmi perf_host_gpu_driver
    perf_host_settings perf_host_platform_settings perf_host_gpu_settings perf_host_details
)

# One section per node: its settings, then the raw details. srun interleaves
# the nodes' output line by line, so every line carries its host and the
# sections are put back together here.
perf_host_snapshot() {
    # shellcheck disable=SC2016
    perf_on_nodes bash -c "$(declare -f "${PERF_HOST_FUNCS[@]}")"'
        h=$(hostname)
        { perf_host_settings; perf_host_details; } | sed "s/^/$h\t/"
    ' 2>&1 | sort -s -t $'\t' -k1,1 |
        awk -F '\t' '
            # Untagged lines are messages from srun itself, not node settings.
            NF < 2 { other = other $0 "\n"; next }
            $1 != host { if (host != "") print ""; print "== " $1 " =="; host = $1 }
            { sub(/^[^\t]*\t/, ""); print }
            END { if (other != "") printf "\n== other output ==\n%s", other }'
}

# The settings lines of one node's section of a host snapshot: <host>'s, or
# the first node's when <host> has none.
perf_host_section() {
    awk -v want="$2" '
        /^== [^ ]+ ==$/ { host = $2; if (first == "") first = host; kv = 1; next }
        /^(== |-- )/ { kv = 0 }
        kv && NF { lines[host] = lines[host] $0 "\n" }
        END { printf "%s", lines[(want in lines) ? want : first] }' "$1"
}

# The settings that differ between the nodes of a host snapshot, one per
# line: "<key>: <value> (<n> nodes); <value> (<n> nodes)".
perf_host_differences() {
    awk '
        /^== [^ ]+ ==$/ { kv = 1; next }
        /^(== |-- )/ { kv = 0 }
        kv && index($0, ": ") {
            key = substr($0, 1, index($0, ": ") - 1); value = substr($0, index($0, ": ") + 2)
            if (!(key in nvalues)) keys[++nkeys] = key
            if (!((key, value) in count)) values[key, ++nvalues[key]] = value
            count[key, value]++
        }
        END {
            for (i = 1; i <= nkeys; i++) {
                key = keys[i]
                if (nvalues[key] < 2) continue
                line = key ":"
                for (j = 1; j <= nvalues[key]; j++) {
                    value = values[key, j]
                    line = line (j > 1 ? ";" : "") " " value " (" count[key, value] " node" (count[key, value] > 1 ? "s" : "") ")"
                }
                print line
            }
        }' "$1"
}

# The image's build manifest (build recipe, dpkg and pip lists), copied once
# per batch so the result directory records how its image was built. Prints
# what happened.
perf_copy_image_manifest() {
    local image="$1" dest="$2" cid err
    if [ -z "$image" ] || ! command -v docker >/dev/null 2>&1; then
        echo "not copied (no image or no docker)"
        return 0
    fi
    # --pull=never: never start an unbounded pull just for the manifest.
    # Docker before 20.10 lacks the flag; with the image present locally a
    # plain create cannot pull either.
    cid=$(docker create --pull=never "$image" /bin/true 2>/dev/null)
    if [ -z "$cid" ] && docker image inspect "$image" >/dev/null 2>&1; then
        cid=$(docker create "$image" /bin/true 2>/dev/null)
    fi
    if [ -z "$cid" ]; then
        echo "not copied (cannot create a container from $image; is it present locally?)"
        return 0
    fi
    # "/." copies the contents, so an existing directory is not nested into.
    if err=$(docker cp "$cid:/workspace/.manifest/." "$dest" 2>&1) && [ -d "$dest" ]; then
        echo "$dest"
    elif [[ "$err" == *"Could not find"* || "$err" == *"No such"* ]]; then
        echo "none in the image (/workspace/.manifest)"
    else
        echo "copy failed: ${err:0:200}"
    fi
    docker rm -f "$cid" >/dev/null 2>&1
}

# Library versions from inside the container. The image digest does not cover
# these, because runtime pip installs happen during setup.
perf_stack_snapshot() {
    local image="${1:-}"
    [ -n "$image" ] || { echo "DOCKER_IMAGE unset"; return 0; }
    command -v docker >/dev/null 2>&1 || { echo "docker not on PATH"; return 0; }

    echo "== image =="
    echo "tag    : $image"
    echo "digest : $(perf_image_digest "$image")"
    echo "id     : $(perf_image_id "$image")"
    echo
    echo "== container libraries =="
    # Single quotes are deliberate: this script runs inside the container, so
    # nothing may expand in the host shell. ROCm comes from /opt/rocm when the
    # image has a classic install, otherwise from the pip-packaged ROCm SDK
    # (TheRock images); hipconfig reports the HIP version, which differs.
    # shellcheck disable=SC2016
    timeout 180 docker run --rm --entrypoint /bin/bash "$image" -lc '
        rocm=$(cat /opt/rocm/.info/version 2>/dev/null)
        [ -n "$rocm" ] || rocm=$(python3 -m pip show rocm 2>/dev/null | awk "/^Version:/ { print \$2 }")
        [ -n "$rocm" ] || rocm=$(python3 -m pip show rocm-sdk-core 2>/dev/null | awk "/^Version:/ { print \$2 }")
        [ -n "$rocm" ] || rocm=$(basename "$(readlink -f /opt/rocm 2>/dev/null)" | sed -n "s/^rocm-//p")
        echo "rocm: ${rocm:-unknown}"
        echo "hip: $(hipconfig --version 2>/dev/null || echo unknown)"
        python3 -m pip list 2>/dev/null |
            grep -iE "^(torch|torchvision|torchaudio|jax|jaxlib|libtpu|transformer|transformers|primus|triton|flash|numpy|nvidia-|megatron|rocm|hip-python|aiter|amd-aiter|apex)" ||
            echo "(pip list unavailable)"
    ' 2>&1 || echo "(container probe failed or timed out; image may need a runtime this host lacks)"
}

# Framework versions lifted out of the stack snapshot so the per-run banner
# can carry them, which is what puts them in the extractor's CSV. Probing the
# container once per batch rather than once per run.
PERF_TORCH_VERSION=""
PERF_JAX_VERSION=""
PERF_CONTAINER_ROCM=""
PERF_CONTAINER_HIP=""
PERF_HOST_SETTINGS=""
PERF_SYSTEM_DESCRIPTION=""
PERF_HOST_CHECK=""

perf_load_stack_versions() {
    local stack_file="$1"
    [ -r "$stack_file" ] || return 0
    PERF_TORCH_VERSION=$(awk '$1 == "torch" { print $2; exit }' "$stack_file")
    PERF_JAX_VERSION=$(awk '$1 == "jax" { print $2; exit }' "$stack_file")
    PERF_CONTAINER_ROCM=$(awk '$1 == "rocm:" { print $2; exit }' "$stack_file")
    PERF_CONTAINER_HIP=$(awk '$1 == "hip:" { print $2; exit }' "$stack_file")
}

# Value of one perf_host_settings key on this node, for the banner.
perf_host_setting() {
    sed -n "s/^$1: //p" <<< "$PERF_HOST_SETTINGS" | head -1
}

# The same, but nothing for "unknown".
perf_host_known() {
    local value
    value=$(perf_host_setting "$1")
    [ "$value" = unknown ] || printf '%s\n' "$value"
}

# GPU memory type, e.g. HBM3E. sysfs does not have it; amd-smi does.
perf_gpu_vram_type() {
    command -v amd-smi >/dev/null 2>&1 || return 0
    timeout 60 amd-smi static --vram -g 0 2>/dev/null </dev/null |
        awk -F ':[[:space:]]*' '$1 ~ /^[[:space:]]*TYPE$/ && $2 != "" && $2 != "N/A" { print $2; exit }'
}

# "2 sockets x 64 cores x 2 threads (256 CPUs), 2 NUMA nodes"
perf_cpu_topology() {
    local sockets cores threads cpus numa
    sockets=$(perf_host_known cpu_sockets)
    cores=$(perf_host_known cpu_cores_per_socket)
    threads=$(perf_host_known cpu_threads_per_core)
    cpus=$(perf_host_known cpus)
    numa=$(perf_host_known numa_nodes)
    if [ -z "$sockets$cores$threads" ]; then
        echo "unknown"
        return 0
    fi
    echo "${sockets:-?} socket$([ "$sockets" = 1 ] || echo s) x ${cores:-?} cores x ${threads:-?} thread$([ "$threads" = 1 ] || echo s)${cpus:+ ($cpus CPUs)}${numa:+, $numa NUMA node$([ "$numa" = 1 ] || echo s)}"
}

# "3023 GiB usable; installed 3072 GiB Registered-DDR5, 24 channels; DIMMs
# 128 GiB Samsung M321RAJA0MB2-CCPWF, 6400 MT/s, configured 6000 MT/s"
perf_host_memory() {
    local usable installed dimms text=""
    usable=$(perf_host_known memory)
    installed=$(perf_host_known memory_installed)
    dimms=$(perf_host_known memory_dimms)
    [ -n "$usable" ] && text="$usable usable"
    [ -n "$installed" ] && text+="${text:+; }installed $installed"
    [ -n "$dimms" ] && text+="${text:+; }DIMMs $dimms"
    echo "${text:-unknown}"
}

# One sentence naming the platform, the way performance results are quoted
# with it, e.g. "Dual AMD EPYC 9575F 64-Core Processor-based Supermicro
# AS -4126GS-NMR-LCC server with 8x AMD Instinct MI355X (288GB HBM3E 1400W)
# GPUs, 1 NUMA node per socket, System BIOS 1.7, Ubuntu 22.04.5 LTS, Host GPU
# driver amdgpu 30.10.1 (ROCm 7.0.1)". Built from PERF_HOST_SETTINGS plus the
# GPU's marketing name and memory type, which only amd-smi (or rocm-smi) on
# this node knows; whatever is unknown is left out.
perf_system_description() {
    local sockets cpu server gpus name vram type cap spec="" part numa bios os driver text=""
    sockets=$(perf_host_known cpu_sockets)
    case "$sockets" in
        1) text="Single " ;;
        2) text="Dual " ;;
        [1-9]*) text="${sockets}x " ;;
    esac
    cpu=$(perf_host_known cpu_model)
    [ -n "$cpu" ] && text+="${cpu}-based "
    server=$(perf_host_known server)
    text+="${server:+$server }server"

    gpus=$(perf_host_known gpu_count)
    if [ -n "$gpus" ] && [ "$gpus" != 0 ]; then
        name=$(perf_gpu_model)
        [ "$name" != unknown ] || name=$(perf_host_known gpu_product)
        vram=$(perf_host_known gpu_vram)
        [[ "$vram" =~ ^([0-9]+)\ GiB$ ]] && vram="${BASH_REMATCH[1]}GB"
        type=$(perf_gpu_vram_type)
        cap=$(perf_host_known gpu_power_cap)
        [[ "$cap" =~ ^([0-9]+)\ W( \(default [0-9]+ W\))?$ ]] && cap="${BASH_REMATCH[1]}W"
        for part in "$vram" "$type" "$cap"; do
            [ -n "$part" ] && spec+="${spec:+ }$part"
        done
        text+=" with ${gpus}x ${name:-AMD}${spec:+ ($spec)} GPUs"
    fi

    numa=$(perf_host_known numa_nodes)
    if [[ "$numa" =~ ^[1-9][0-9]*$ && "$sockets" =~ ^[1-9][0-9]*$ ]] && [ $((numa % sockets)) -eq 0 ]; then
        numa=$((numa / sockets))
        text+=", $numa NUMA node$([ "$numa" = 1 ] || echo s) per socket"
    elif [ -n "$numa" ]; then
        text+=", $numa NUMA node$([ "$numa" = 1 ] || echo s)"
    fi
    bios=$(perf_host_known bios)
    bios=${bios%% (*}
    [ -n "$bios" ] && [ "$bios" != unknown ] && text+=", System BIOS $bios"
    os=$(perf_host_known os)
    [ -n "$os" ] && text+=", $os"
    driver=$(perf_host_known gpu_driver)
    [ -n "$driver" ] && [ "$driver" != "not loaded" ] && text+=", Host GPU driver $driver"
    echo "$text"
}

# The banner's platform: this node's section of the host snapshot, or the
# first node's when the runner is not one of them (salloc can leave the shell
# on a login node). For multi-node, also whether every node reported the
# same settings; those that differ are appended to the snapshot.
perf_load_host_settings() {
    local host_file="$1" nodes diffs
    PERF_HOST_SETTINGS=$(perf_host_section "$host_file" "$(hostname)")
    [ -n "$PERF_HOST_SETTINGS" ] || PERF_HOST_SETTINGS=$(perf_host_settings 2>/dev/null)
    PERF_SYSTEM_DESCRIPTION=$(perf_system_description)
    [ "${PERF_MULTINODE:-0}" = 1 ] || return 0

    nodes=$(grep -c '^== [^ ]* ==$' "$host_file")
    diffs=$(perf_host_differences "$host_file")
    if [ -n "$diffs" ]; then
        printf '\n== differences between nodes ==\n%s\n' "$diffs" >> "$host_file"
        PERF_HOST_CHECK="differs between nodes: $(cut -d: -f1 <<< "$diffs" | paste -sd ',' | sed 's/,/, /g') (see ${host_file##*/})"
        perf_log "[WARN] The host platform $PERF_HOST_CHECK"
    elif [ "$nodes" -eq 0 ]; then
        PERF_HOST_CHECK="not checked (no node reported its settings)"
    elif [ "$nodes" -eq 1 ]; then
        PERF_HOST_CHECK="one node, nothing to compare"
    else
        PERF_HOST_CHECK="same on all $nodes nodes"
    fi
    if [ "$nodes" -lt "${NNODES:-$nodes}" ]; then
        PERF_HOST_CHECK+="; only $nodes of $NNODES nodes reported"
    fi
}

# Write the batch-level snapshots next to the logs. Sets the PERF_* values
# the banner prints, so it must not run in a pipeline.
perf_write_batch_snapshots() {
    local result_dir="$1" stamp="$2" image="${3:-}"

    perf_log "[INFO] Recording provenance: environment, submodules, GPU state, each node's platform and host settings, and ${image:-the image}'s ROCm/HIP and libraries (the container probe can take up to 3 min)."

    perf_env_redacted                > "$result_dir/batch_env_${stamp}.txt"
    perf_submodule_status            > "$result_dir/batch_submodules_${stamp}.txt"
    perf_gpu_snapshot                > "$result_dir/batch_gpu_${stamp}.txt"
    perf_stack_snapshot "$image"     > "$result_dir/batch_stack_${stamp}.txt"
    perf_host_snapshot               > "$result_dir/batch_host_${stamp}.txt"

    perf_load_stack_versions "$result_dir/batch_stack_${stamp}.txt"
    perf_load_host_settings "$result_dir/batch_host_${stamp}.txt"

    perf_log "Image manifest     -> $(perf_copy_image_manifest "$image" "$result_dir/batch_manifest_${stamp}")"
}

# Banner block shared by both runners. Printed inside the per-run banner, so
# every field lands in the log the extractor reads.
# Set for a run whose container does not inherit the runner's environment.
PERF_ENV_NOTE=""

perf_print_provenance_banner() {
    local image="${1:-}"

    echo "# Docker image     : ${image:-<unset; container mode uses runner/.primus.yaml>}"
    echo "# Image digest     : $(perf_image_digest "$image")"
    echo "# Image ID         : $(perf_image_id "$image")"
    echo "# Primus commit    : $(perf_primus_commit)"
    echo "# Submodule pins   : $(perf_submodule_pins)"
    echo "# GPU model        : $(perf_gpu_model)"
    echo "# ROCm (host)      : $(perf_rocm_version)"
    echo "# Container ROCm   : ${PERF_CONTAINER_ROCM:-unknown}"
    echo "# Container HIP    : ${PERF_CONTAINER_HIP:-unknown}"
    echo "# Torch version    : ${PERF_TORCH_VERSION:-unknown}"
    echo "# JAX version      : ${PERF_JAX_VERSION:-unknown}"
    echo "# World size       : $(( ${NNODES:-1} * ${GPUS_PER_NODE:-8} ))  (NNODES=${NNODES:-1} x GPUS_PER_NODE=${GPUS_PER_NODE:-8})"
    echo "# --------------------------------------------------------------------------------"
    # This node's platform (the first node's when the runner is not one of
    # them); batch_host_<stamp>.txt has every node's, in full.
    echo "# System           : ${PERF_SYSTEM_DESCRIPTION:-unknown}"
    if [ "${PERF_MULTINODE:-0}" = 1 ]; then
        echo "# Host check       : ${PERF_HOST_CHECK:-not checked}"
    fi
    echo "# Server           : $(perf_host_setting server)"
    echo "# BIOS             : $(perf_host_setting bios)"
    echo "# OS               : $(perf_host_setting os)"
    echo "# Kernel           : $(perf_host_setting kernel)"
    echo "# Kernel cmdline   : $(perf_host_setting kernel_cmdline)"
    echo "# CPU              : $(perf_host_setting cpu_model)"
    echo "# CPU topology     : $(perf_cpu_topology)"
    echo "# CPU frequency    : $(perf_host_setting cpu_frequency)"
    echo "# CPU governor     : $(perf_host_setting cpu_governor)"
    echo "# CPU idle states  : $(perf_host_setting cpu_idle)"
    echo "# NUMA balancing   : $(perf_host_setting numa_balancing)"
    echo "# Host memory      : $(perf_host_memory)"
    echo "# Transparent HP   : $(perf_host_setting transparent_hugepages)"
    echo "# IOMMU            : $(perf_host_setting iommu)"
    echo "# GPU VRAM         : $(perf_host_setting gpu_vram)"
    echo "# GPU power cap    : $(perf_host_setting gpu_power_cap)"
    echo "# GPU partition    : $(perf_host_setting gpu_partition)"
    echo "# GPU clocks       : $(perf_host_setting gpu_clocks)"
    echo "# GPU VBIOS        : $(perf_host_setting gpu_vbios)"
    echo "# GPU PCIe         : $(perf_host_setting gpu_pcie)"
    echo "# Host GPU driver  : $(perf_host_setting gpu_driver)"
    echo "# Driver package   : $(perf_host_setting gpu_driver_package)"
    echo "# amdgpu driver    : $(perf_host_setting amdgpu_driver)"
    echo "# RDMA NICs        : $(perf_host_setting rdma_nics)"
    echo "# --------------------------------------------------------------------------------"
    # Distinguished from the block below because these are *guaranteed* to
    # reach the container, whereas a plain export may be filtered out.
    local line extra_env=()
    read -r -a extra_env <<< "${EXTRA_ENV:-}"
    if [ ${#extra_env[@]} -gt 0 ]; then
        echo "# Extra env (--env): $(perf_display_argv "${extra_env[@]}")"
    else
        echo "# Extra env (--env): <none>"
    fi
    # The runner's own shell. Primus prints its main environment settings
    # (and each hook's exports) at the start of the run output; settings from
    # runner/helpers/envs/<GPU>.sh are not all echoed there. PERF_ENV_NOTE
    # qualifies the block for a run that does not inherit it.
    echo "# Perf environment (host shell${PERF_ENV_NOTE:+; $PERF_ENV_NOTE}):"
    while IFS= read -r line; do
        echo "#   $line"
    done < <(perf_env_lines)
}
