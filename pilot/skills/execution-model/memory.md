# Memory Modeling

**Status**: Stub

`Mem = M_param + M_grad + M_optim + M_act + M_buffer`

## TODO

- [ ] Per-component formula (param/grad/optim depend on TP/PP/ZeRO; act depends on seq/hidden/mbs/recompute)
- [ ] γ_act calibration (actual / theoretical, §S1)
- [ ] δ_buffer constant (workspace / fragmentation)
- [ ] OOM safety margin policy
