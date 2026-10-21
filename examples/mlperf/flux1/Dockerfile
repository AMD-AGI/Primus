# One immutable runtime is used on both DCCS and Crusoe; FLUX_CLUSTER selects
# site networking at launch time. Primus is bind-mounted at /workspace/Primus.
FROM zirui3/primus-v26.7-flux:v0.1@sha256:082afc328d867f79ab5c4a46bc29c909bd891e068a32506a10768e6a6cc7e7c8

# The image packages Primus-Turbo at this source pin. Kernel sources remain in
# that package and are intentionally not vendored into Primus.
LABEL org.opencontainers.image.source="https://github.com/zirui/Primus-Turbo" \
      org.opencontainers.image.revision="e031c5bb3746e8b667676bd20e2697d75fe9f2b2" \
      com.amd.mlperf.flux.cluster-profiles="FLUX_CLUSTER=dccs|crusoe"
