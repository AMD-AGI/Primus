#!/bin/bash
# Sourced by the EAT harness and by eat/eval.sh. Do not put secrets in this file.
export EAT_GPUS="${EAT_GPUS:-0}"
# This machine has no git identity configured. Keep it in the environment so
# the harness can commit without writing git config.
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-weihuan}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-weihuan@heliosr-1b112-a30-4.mnb.dcgpu}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"
export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin:${HOME}/bin:${PATH}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://llm-api.amd.com/Unified}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}"
if [ -z "${ANTHROPIC_CUSTOM_HEADERS:-}" ] && [ -f "${HOME}/llm_api" ]; then
    export LLM_GATEWAY_KEY="$(tr -d '[:space:]' < "${HOME}/llm_api")"
    export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${LLM_GATEWAY_KEY}"
fi
