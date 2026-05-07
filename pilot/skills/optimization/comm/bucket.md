# Bucket Tuning

**Status**: Stub

DDP / FSDP gradient bucket size. Trade-off: larger buckets → better link utilization but more straggler tail.

## TODO

- [ ] Default range per cluster class (16 / 32 / 64 / 128 MB)
- [ ] Interaction with overlap
- [ ] OOM consideration
