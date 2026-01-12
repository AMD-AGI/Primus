## bnxt rebuild patch (`runner/helpers/rebuild_bnxt.sh`)

Rebuilds the `bnxt` lib from a tarball, mirroring the logic that used to live
in `examples/run_pretrain.sh` (`PATH_TO_BNXT_TAR_PACKAGE`).

- **Environment variable**
  - `PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-*.tar.gz`

- **How to use with `primus-cli direct`**

  ```bash
  export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-xxx.tar.gz

  primus-cli direct \
    --patch runner/helpers/rebuild_bnxt.sh \
    -- train pretrain --config examples/megatron/exp_pretrain.yaml
  ```

- **Exit codes (for `execute_patches.sh`)**
  - `0` — rebuild succeeded
  - `2` — skipped (no tarball found)
  - `other` — error, stops the patch pipeline


## Primus-Turbo rebuild patch (`runner/helpers/rebuild_primus_turbo.sh`)

Rebuilds Primus-Turbo from source, mirroring the logic that used to live in
`examples/run_pretrain.sh` (`REBUILD_PRIMUS_TURBO`).

- **Environment variables**
  - `PRIMUS_TURBO_BUILD_DIR=/path` (optional, default: `/tmp/primus_turbo_$HOSTNAME`)
  - `GPU_ARCHS="gfx942;gfx950"` (optional) — target GPU architectures
  - `PRIMUS_TURBO_REF=<branch-or-sha>` (optional) — git ref (branch/tag/commit) to checkout

- **How to use with `primus-cli direct`**

  ```bash
  # Optional overrides:
  # export PRIMUS_TURBO_BUILD_DIR=/custom/path
  # export GPU_ARCHS="gfx942;gfx90a"
  # export PRIMUS_TURBO_REF="v0.3.0"   # or a commit SHA / branch name

  primus-cli direct \
    --patch runner/helpers/rebuild_primus_turbo.sh \
    -- train pretrain --config examples/megatron/exp_pretrain.yaml
  ```

- **Exit codes (for `execute_patches.sh`)**
  - `0` — rebuild succeeded
  - `other` — error, stops the patch pipeline
