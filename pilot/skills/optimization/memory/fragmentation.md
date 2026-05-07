# Fragmentation Mitigation

**Status**: Stub

`mem_reserved / mem_alloc` ratio high → fragmentation. Likely fix: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (see `skills/env/alloc.md`).

## TODO

- [ ] Detection threshold
- [ ] Variant by allocator (caching / expandable / tcmalloc)
- [ ] When fragmentation is structural (forces a re-PROJECT)
