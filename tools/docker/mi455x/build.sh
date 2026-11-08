#!/usr/bin/env bash
# Build the MI455X Kimi K3.1 Pro + DLRMv4 image (CPU only, no GPU needed, ~45 min cold).
#
#   ./build.sh [tag]
#
# DLRMV4_IMAGE      image to take the DLRMv4 `recommendation/` tree from (/workspace/recommendation)
#                   when DLRMV4_SRC is not set; default amdprimus/amdprimus:mi455-training-20261108.
# DLRMV4_SRC        a local DLRMv4 `recommendation/` directory to use instead. If it is inside a
#                   git checkout, its HEAD commit is recorded in the image label.
# DLRMV4_COMMIT     overrides the recorded DLRMv4 commit.
# BASE_IMAGE        default amdprimus/amdprimus:tas-20261019-turbo-4c74631
#
# The Primus context is `git archive HEAD` of this checkout plus third_party/Megatron-LM, so only
# committed files go into the image.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${1:-primus-mi455x:kimi-k3.1-pro-dlrmv4}
DLRMV4_IMAGE=${DLRMV4_IMAGE:-amdprimus/amdprimus:mi455-training-20261108}
PRIMUS_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
PRIMUS_COMMIT=$(git -C "$PRIMUS_ROOT" rev-parse HEAD)

if [ -n "$(git -C "$PRIMUS_ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "warning: uncommitted changes in $PRIMUS_ROOT are NOT in the image" >&2
fi

CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT

if [ -z "${DLRMV4_SRC:-}" ]; then
  echo "DLRMv4 tree from $DLRMV4_IMAGE"
  docker image inspect "$DLRMV4_IMAGE" >/dev/null 2>&1 || docker pull "$DLRMV4_IMAGE"
  cid=$(docker create "$DLRMV4_IMAGE")
  docker cp "$cid:/workspace/recommendation" "$CTX/dlrmv4"
  docker rm "$cid" >/dev/null
  DLRMV4_SRC=$CTX/dlrmv4
  DLRMV4_COMMIT=${DLRMV4_COMMIT:-$(docker image inspect "$DLRMV4_IMAGE" \
    --format '{{index .Config.Labels "org.amd.dlrmv4.commit"}}')}
fi
DLRMV4_COMMIT=${DLRMV4_COMMIT:-$(git -C "$DLRMV4_SRC" rev-parse HEAD 2>/dev/null || echo unknown)}

# Only what training, data preparation and the image build use goes into the image.
DLRMV4_FILES=(
  generative_recommenders
  docker/mi455/fbgemm_cta_per_row_gfx1250_rebuild.sh
  docker/mi455/fbgemm_host_warp32_gfx1250_rebuild.sh
  docker/mi455/fbgemm_codegen_warp32_gfx1250_rebuild.sh
  scripts/ts_index_tool.py
  tools/mi455/run_kit/analyze.py
  download_dataset.sh
  verify_dataset.sh
  md5sums_yambda_5b_processed.txt
  sha256sums_yambda_5b_cache.txt
  setup.py
  requirements.txt
  LICENSE
)
for f in "${DLRMV4_FILES[@]}"; do
  [ -e "$DLRMV4_SRC/$f" ] || { echo "$DLRMV4_SRC is missing $f (not a DLRMv4 recommendation/ tree?)" >&2; exit 1; }
done
mkdir -p "$CTX/dlrmv4_files"
tar -C "$DLRMV4_SRC" --exclude=__pycache__ --exclude='*.pyc' -cf - "${DLRMV4_FILES[@]}" \
  | tar -C "$CTX/dlrmv4_files" -xf -
DLRMV4_SRC=$CTX/dlrmv4_files

mkdir -p "$CTX/primus/third_party/Megatron-LM"
git -C "$PRIMUS_ROOT" archive HEAD | tar -x -C "$CTX/primus"
git -C "$PRIMUS_ROOT/third_party/Megatron-LM" archive HEAD | tar -x -C "$CTX/primus/third_party/Megatron-LM"
echo "$PRIMUS_COMMIT" > "$CTX/primus/.primus_commit"

echo "Primus $PRIMUS_COMMIT, Megatron-LM $(git -C "$PRIMUS_ROOT/third_party/Megatron-LM" rev-parse --short HEAD), DLRMv4 $DLRMV4_COMMIT -> $TAG"
docker build --progress=plain \
  --build-context primus="$CTX/primus" \
  --build-context dlrmv4="$DLRMV4_SRC" \
  ${BASE_IMAGE:+--build-arg BASE_IMAGE="$BASE_IMAGE"} \
  --build-arg PRIMUS_COMMIT="$PRIMUS_COMMIT" \
  --build-arg DLRMV4_COMMIT="$DLRMV4_COMMIT" \
  -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
echo "built $TAG"
