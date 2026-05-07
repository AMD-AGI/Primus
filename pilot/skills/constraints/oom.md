# OOM Estimation

**Status**: Stub

`Mem_pred = M_param + M_grad + M_optim + γ_act × M_act + δ_buffer` (see `execution-model/memory.md` and §S1 calibration).

Reject the candidate if `Mem_pred > hbm_capacity_gb × safety_margin` (default 0.9).

## TODO

- [ ] safety_margin policy by cluster class
- [ ] How to handle predictor underestimate history (tighten margin)
- [ ] Failure → mark plan dead with predicted_mem / actual_mem in PlanGraph node
