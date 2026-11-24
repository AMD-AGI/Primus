# fix(deep_ep): default to the full fence above 80 CUs

Branch: `fix/deep_ep/cheap-fence-high-cu` @ `394131b3`
Compare: https://github.com/AMD-AGI/Primus-Turbo/compare/main...fix/deep_ep/cheap-fence-high-cu?expand=1

## Summary

- Default DeepEP intranode cheap fence only at `num_sms <= 80` (the #539 validated range).
- Above that, use the full release/acquire fence unless `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE` forces a choice (`1` = full, `0` = cheap).
- Addresses intermittent NaN forward loss on the Qwen3-30B-A3B FP8 full-opt stack at `turbo_deepep_num_cu` 160/192 (~1/13 runs).
- Primus counterpart: `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE=1` in the Qwen3 FP8 MI355X config (works with older Turbo that lacks this default).

## Test plan

- [ ] DeepEP intranode stress at 80 CUs still uses cheap fence when env unset
- [ ] At 160 CUs with env unset, dispatch/combine select the full fence
- [ ] `DISABLE=0` still forces cheap at 160 for A/B
- [ ] Qwen3 FP8 20-iter e2e at 160 CUs with full opt (multiple reps)
