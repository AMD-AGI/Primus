## bnxt rebuild hook (`runner/helpers/hooks/02_rebuild_bnxt.sh`)

Rebuilds the `bnxt` lib from a tarball, mirroring the logic that used to live
in `examples/run_pretrain.sh` (`PATH_TO_BNXT_TAR_PACKAGE`).

- **Environment variable**
  - `PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-*.tar.gz`

- **How to use with `primus-cli direct`**

  ```bash
  export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-xxx.tar.gz

  # The system hook will run automatically before command-specific hooks:
  export REBUILD_BNXT=1
  export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-xxx.tar.gz

  primus-cli direct \
    -- train pretrain --config examples/megatron/exp_pretrain.yaml
  ```

- **Behavior**
  - If `REBUILD_BNXT=1` and the tarball exists, the hook rebuilds bnxt.
  - If the tarball is missing, the hook logs a skip message and continues.


## Primus-Turbo rebuild hook (`runner/helpers/hooks/00_rebuild_primus_turbo.sh`)

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

  export REBUILD_PRIMUS_TURBO=1
  primus-cli direct \
    -- train pretrain --config examples/megatron/exp_pretrain.yaml
  ```

- **Behavior**
  - If `REBUILD_PRIMUS_TURBO=1`, the system hook rebuilds Primus-Turbo.
  - The build workspace defaults to `/tmp/primus_turbo_$HOSTNAME` (override via `PRIMUS_TURBO_BUILD_DIR`).
