## Backend Patch Notes Overview

Primus integrates several large-model backends (Megatron-LM, TorchTitan, JAX MaxText, …) and applies a lightweight patch layer to keep configuration flags consistent with the Primus CLI.
This area of the docs captures those backend-specific switches so they live alongside the rest of the documentation (instead of the historical `primus/README_patch.md` file).

### How to read these docs
- Start with the **Base Module Parameters** table below – every backend module inherits these knobs.
- Jump to the backend-specific page for details on extra CLI/config options and links to the patched source files.
- When editing configs or CLI presets, cross-reference the [Primus CLI Guide](../cli/PRIMUS-CLI-GUIDE.md#configuration) so command examples and backend parameters stay in sync.

### Base Module Parameters
All modules inherit the options defined in [`primus/configs/modules/module_base.yaml`](https://github.com/AMD-AGI/Primus/blob/main/primus/configs/modules/module_base.yaml):

| Argument Name       | Default Value | Description                                                                                |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `trainable`         | `false`       | Whether the module participates in training.                                              |
| `sink_level`        | `null`        | Global sink level for logging. Overrides `file_sink_level` and `stderr_sink_level` if set. |
| `file_sink_level`   | `DEBUG`       | Logging level for file sink (log files).                                                  |
| `stderr_sink_level` | `INFO`        | Logging level for stderr/console output.                                                  |

### Backend Index
- [Megatron-LM Patch Notes](./megatron/patch-notes.md)
- [TorchTitan Patch Notes](./torchtitan/patch-notes.md)
- JAX MaxText – coming soon (tracked here to keep the structure ready)
