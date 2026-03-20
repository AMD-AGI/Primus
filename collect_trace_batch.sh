#!/bin/bash
bash collect_trace_qwen3.sh 16
bash collect_trace_qwen3.sh 8
bash collect_trace_qwen3.sh 4
bash collect_trace_qwen3.sh 2

bash collect_trace_dsv3.sh 32 && \
bash collect_trace_dsv3.sh 16
bash collect_trace_dsv3.sh 8
bash collect_trace_dsv3.sh 4