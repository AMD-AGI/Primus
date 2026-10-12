#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Shared helpers for the Primus perf batch runners.
#
# This file holds only code that is new to tools/perf (config selection and
# provenance capture). The runners keep their own copies of the pre-existing
# helpers; nothing that already worked was moved here.
#
# Callers must set PRIMUS_PATH (repo root) before sourcing.
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
    # which node you are on; the script can too.
    if [ -z "${GPU:-}" ]; then
        GPU=$(perf_detect_gpu)
        if [ -z "$GPU" ]; then
            echo "[ERROR] GPU is not set and could not be detected from rocm-smi." >&2
            echo "        Set GPU=<key>, one of: $(yq -r 'keys | .[]' "$catalog" 2>/dev/null | tr '\n' ' ')" >&2
            return 1
        fi
        echo "[INFO] GPU not set; detected $GPU from rocm-smi." >&2
    fi

    if [ "$(yq -r "has(\"$GPU\")" "$catalog" 2>/dev/null)" != "true" ]; then
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
    if [ -z "${BACKEND:-}" ]; then
        echo "[ERROR] BACKEND is required; each training image supports one framework stack." >&2
        echo "        Available for $GPU: $(yq -r ".[\"$GPU\"] | keys | .[]" "$catalog" 2>/dev/null | tr '\n' ' ')" >&2
        echo "        PyTorch image: BACKEND=megatron,torchtitan" >&2
        echo "        JAX image    : BACKEND=maxtext,maxdiffusion" >&2
        return 1
    fi

    local backend entry backends=()
    IFS=',' read -r -a backends <<< "$BACKEND"

    for backend in "${backends[@]}"; do
        backend="${backend// /}"
        [ -n "$backend" ] || continue
        if [ "$(yq -r ".[\"$GPU\"] | has(\"$backend\")" "$catalog" 2>/dev/null)" != "true" ]; then
            echo "[ERROR] backend '$backend' is not listed under $GPU in $catalog" >&2
            echo "        Available for $GPU: $(yq -r ".[\"$GPU\"] | keys | .[]" "$catalog" 2>/dev/null | tr '\n' ' ')" >&2
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

    if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
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

    for f in "${CONFIG_FILES[@]}"; do
        if [ ! -f "$f" ]; then
            echo "[ERROR] config does not exist: $f" >&2
            bad=$((bad + 1))
        fi
    done

    seen_dupes=$(printf '%s\n' "${CONFIG_FILES[@]}" | sort | uniq -d)
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
    local kv

    for kv in ${EXTRA_ENV:-}; do
        if [[ "$kv" != *=* ]] || [[ "$kv" != [A-Za-z_]* ]]; then
            echo "[ERROR] EXTRA_ENV entries must look like KEY=VALUE, got: '$kv'" >&2
            echo "        Example: EXTRA_ENV=\"DEBUG_HIP_DYNAMIC_QUEUES=0 GPU_MAX_HW_QUEUES=2\"" >&2
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

perf_gpu_model() {
    command -v rocm-smi >/dev/null 2>&1 || { echo "unknown"; return 0; }
    local model
    model=$(rocm-smi --showproductname 2>/dev/null |
        sed -n 's/.*Card Series:[[:space:]]*//p' | head -1 |
        sed 's/[[:space:]]*$//')
    [ -n "$model" ] && echo "$model" || echo "unknown"
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

perf_rocm_version() {
    if [ -r /opt/rocm/.info/version ]; then
        head -1 /opt/rocm/.info/version
    elif command -v rocminfo >/dev/null 2>&1; then
        rocminfo 2>/dev/null | awk '/ROCm Version/ {print $3; exit}'
    else
        echo "unknown"
    fi
}

perf_gpu_snapshot() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showproductname --showclocks --showpower --showmaxpower 2>&1 || true
        echo
        echo "== topology =="
        rocm-smi --showtopo 2>&1 || true
    elif command -v amd-smi >/dev/null 2>&1; then
        amd-smi static 2>&1 || true
    else
        echo "no rocm-smi / amd-smi on PATH"
    fi
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
    echo
    echo "== container libraries =="
    # Single quotes are deliberate: this script runs inside the container, so
    # nothing may expand in the host shell.
    # shellcheck disable=SC2016
    timeout 180 docker run --rm --entrypoint /bin/bash "$image" -lc '
        rocm=$(cat /opt/rocm/.info/version 2>/dev/null) ||
            rocm=$(hipconfig --version 2>/dev/null) ||
            rocm=$(basename "$(readlink -f /opt/rocm 2>/dev/null)" 2>/dev/null)
        echo "rocm: ${rocm:-unknown}"
        python3 -m pip list 2>/dev/null |
            grep -iE "^(torch|torchvision|torchaudio|jax|jaxlib|libtpu|transformer|transformers|primus|triton|flash|numpy|nvidia-|megatron)" ||
            echo "(pip list unavailable)"
    ' 2>&1 || echo "(container probe failed or timed out; image may need a runtime this host lacks)"
}

# Framework versions lifted out of the stack snapshot so the per-run banner
# can carry them, which is what puts them in the extractor's CSV. Probing the
# container once per batch rather than once per run.
PERF_TORCH_VERSION=""
PERF_JAX_VERSION=""

perf_load_stack_versions() {
    local stack_file="$1"
    [ -r "$stack_file" ] || return 0
    PERF_TORCH_VERSION=$(awk '$1 == "torch" { print $2; exit }' "$stack_file")
    PERF_JAX_VERSION=$(awk '$1 == "jax" { print $2; exit }' "$stack_file")
}

# Write the batch-level snapshots next to the logs.
perf_write_batch_snapshots() {
    local result_dir="$1" stamp="$2" image="${3:-}"

    perf_env_redacted                > "$result_dir/batch_env_${stamp}.txt"
    perf_submodule_status            > "$result_dir/batch_submodules_${stamp}.txt"
    perf_gpu_snapshot                > "$result_dir/batch_gpu_${stamp}.txt"
    perf_stack_snapshot "$image"     > "$result_dir/batch_stack_${stamp}.txt"

    perf_load_stack_versions "$result_dir/batch_stack_${stamp}.txt"
}

# Banner block shared by both runners. Printed inside the per-run banner, so
# every field lands in the log the extractor reads.
perf_print_provenance_banner() {
    local image="${1:-}"

    echo "# Docker image     : ${image:-<unset; container mode uses runner/.primus.yaml>}"
    echo "# Image digest     : $(perf_image_digest "$image")"
    echo "# Primus commit    : $(perf_primus_commit)"
    echo "# Submodule pins   : $(perf_submodule_pins)"
    echo "# GPU model        : $(perf_gpu_model)"
    echo "# ROCm version     : $(perf_rocm_version)"
    echo "# Torch version    : ${PERF_TORCH_VERSION:-unknown}"
    echo "# JAX version      : ${PERF_JAX_VERSION:-unknown}"
    echo "# World size       : $(( ${NNODES:-1} * ${GPUS_PER_NODE:-8} ))  (NNODES=${NNODES:-1} x GPUS_PER_NODE=${GPUS_PER_NODE:-8})"
    echo "# --------------------------------------------------------------------------------"
    # Distinguished from the block below because these are *guaranteed* to
    # reach the container, whereas a plain export may be filtered out.
    echo "# Extra env (--env): ${EXTRA_ENV:-<none>}"
    echo "# Perf environment :"
    local line
    while IFS= read -r line; do
        echo "#   $line"
    done < <(perf_env_lines)
}
