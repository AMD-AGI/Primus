# T_comp Modeling

**Status**: Stub

`T_comp = model_flops / (num_gpus × peak_tflops × η_comp)`

## TODO

- [ ] FLOPs counting per architecture (Dense MLP / Attention / MoE expert)
- [ ] η_comp source (calibration §S1)
- [ ] mbs scaling regime (linear vs sublinear at small mbs)
