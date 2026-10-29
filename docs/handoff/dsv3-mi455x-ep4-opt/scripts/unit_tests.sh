#!/usr/bin/env bash
# Runs inside the container: the DeepEP intranode tests that cover the dsv3opt Turbo changes, and the
# gfx1250 variable-K beta=1 wgrad check / benchmark.
set -u
V2=${V2:-/home/boyan/project/deepep/Primus-Turbo-dsv3opt}
D=/home/boyan/project/deepep/dsv3opt
cd "$V2"
export PYTHONPATH=$V2 FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-/tmp/dsv3opt_cache/flydsl}
rc=0
echo "== intranode: worst-case tail + round trips"
timeout 1200 python3 -m pytest -q -x tests/pytorch/deep_ep/test_intranode.py \
  -k "worst_case_tail or (flydsl_v2 and (test_intranode_ or rank_receives_nothing))" || rc=1
echo "== variable-K beta=1 wgrad"
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES%%,*} timeout 900 python3 "$D/bench_vk_accum.py" || rc=1
exit $rc
