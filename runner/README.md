## bnxt rebuild patch (`runner/helpers/rebuild_nbxt.sh`)

Rebuilds the `bnxt` lib from a tarball, mirroring the logic that used to live
in `examples/run_pretrain.sh` (`REBUILD_BNXT` + `PATH_TO_BNXT_TAR_PACKAGE`).

- **Environment variables**
  - `REBUILD_BNXT=1` (default: 0) — enable the rebuild
  - `PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-*.tar.gz`

- **How to use with `primus-cli direct`**

  - Via CLI:

    ```bash
    export REBUILD_BNXT=1
    export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-xxx.tar.gz

    primus-cli direct \
      --patch runner/helpers/rebuild_nbxt.sh \
      -- train pretrain --config examples/megatron/exp_pretrain.yaml
    ```

- **Exit codes (for `execute_patches.sh`)**
  - `0` — rebuild succeeded
  - `2` — skipped (no tarball or `RE/RUN_BUILD_BNXT` not set to 1)
  - other — error, stops the patch pipeline
