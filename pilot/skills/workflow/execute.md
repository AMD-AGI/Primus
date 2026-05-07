# Execute

**Status**: Stub

Submit selected plans (`submit.run()`), monitor, early-stop. Three modes: FullScale / Sharded / TimeMux (see §S3).

## TODO

- [ ] Mode selection rules (delegate to §S3 Sharded eligibility check)
- [ ] Early-stop policy (mem trend / tps trend)
- [ ] Cross-shard anomaly detection (§S3.3)
- [ ] Extrapolation rules from sharded → full scale (§S3.4)
