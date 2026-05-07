# Config Validity

**Status**: Stub

Static rules every Plan must satisfy before Execute.

## TODO

- [ ] tp must divide hidden, num_heads, num_kv_heads
- [ ] dp × tp × pp must equal world_size
- [ ] mbs × gas × dp = gbs
- [ ] vpp × pp must divide num_layers
- [ ] ep must divide num_experts
